import torch
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from typing import Type, Tuple, Union
import cuda.bindings.driver as cuda
from inspect import isclass

class Sm100GroupedSwiGLUBlockscaledKernel:
    bytes_per_tensormap = 128
    tensor_memory_management_bytes = 1048576
    num_tensormaps = 7

    def __init__(self, sf_vec_size, mma_tiler_mn, cluster_shape_mn):
        self.sf_vec_size = sf_vec_size
        self.use_2cta_instrs = False
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tensormap_update_mode = utils.TensorMapUpdateMode.SMEM
        self.delegate_tensormap_ab_init = True
        self.occupancy = 1
        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        
        self.a_dtype = cutlass.Float8E4M3FN
        self.b_dtype = cutlass.Float8E4M3FN
        self.c_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32

        self.tma_warp_id = 5
        self.threads_per_warp = 32
        self.threads_per_cta = 32 * len((self.mma_warp_id, self.tma_warp_id, *self.epilog_warp_id))
        self.epilog_sync_barrier = pipeline.NamedBarrier(1, 32 * len(self.epilog_warp_id))
        self.tmem_alloc_barrier = pipeline.NamedBarrier(2, 32 * len((self.mma_warp_id, *self.epilog_warp_id)))
        self.tensormap_ab_init_barrier = pipeline.NamedBarrier(3, 32 * (len(self.epilog_warp_id) + 1))
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

    def _compute_stages(self, tiled_mma, mma_tiler, a_dt, b_dt, epi_tile, c_dt, c_layout, sf_dt, sf_vec, smem_cap, occ):
        # Delegate to blockscaled logic
        a_smem_staged = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler, a_dt, 1)
        b_smem_staged = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler, b_dt, 1)
        sfa_smem_staged = blockscaled_utils.make_smem_layout_sfa(tiled_mma, mma_tiler, sf_vec, 1)
        sfb_smem_staged = blockscaled_utils.make_smem_layout_sfb(tiled_mma, mma_tiler, sf_vec, 1)
        c_smem_staged = sm100_utils.make_smem_layout_epi(c_dt, c_layout, epi_tile, 1)
        
        a_bytes = cute.size_in_bytes(a_dt, a_smem_staged)
        b_bytes = cute.size_in_bytes(b_dt, b_smem_staged)
        sfa_bytes = cute.size_in_bytes(sf_dt, sfa_smem_staged)
        sfb_bytes = cute.size_in_bytes(sf_dt, sfb_smem_staged)
        c_bytes = cute.size_in_bytes(c_dt, c_smem_staged)
        
        mbar_bytes_per_stage = 16
        tensormap_bytes = self.num_tensormaps * self.bytes_per_tensormap
        max_smem_bytes = smem_cap // occ - tensormap_bytes
        
        num_ab_stage = 2
        while ((a_bytes + 2*b_bytes + sfa_bytes + 2*sfb_bytes) * num_ab_stage + c_bytes + 2 * mbar_bytes_per_stage * num_ab_stage <= max_smem_bytes):
            num_ab_stage += 1
        num_ab_stage -= 1
        
        num_c_stage = 1
        num_acc_stage = 1
        return num_acc_stage, num_ab_stage, num_c_stage

    def _setup_attributes(self):
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (self.mma_inst_shape_mn[0], cute.round_up(self.mma_inst_shape_mn[1], 128))

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.a_major_mode, self.b_major_mode, self.sf_dtype, self.sf_vec_size, self.cta_group, self.mma_inst_shape_mn
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.a_major_mode, self.b_major_mode, self.sf_dtype, self.sf_vec_size, cute.nvgpu.tcgen05.CtaGroup.ONE, self.mma_inst_shape_mn_sfb
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (self.mma_inst_shape_mn[0], self.mma_inst_shape_mn[1], mma_inst_shape_k * mma_inst_tile_k)
        self.mma_tiler_sfb = (self.mma_inst_shape_mn_sfb[0], self.mma_inst_shape_mn_sfb[1], mma_inst_shape_k * mma_inst_tile_k)
        
        self.cta_tile_shape_mnk = (self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape), self.mma_tiler[1], self.mma_tiler[2])
        self.cta_tile_shape_mnk_sfb = (self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape), self.mma_tiler_sfb[1], self.mma_tiler_sfb[2])
        self.cluster_tile_shape_mnk = tuple(x * y for x, y in zip(self.cta_tile_shape_mnk, (*self.cluster_shape_mn, 1)))

        self.cluster_layout_vmnk = cute.tiled_divide(cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,))
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma_sfb.thr_id.shape,))

        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(self.cta_tile_shape_mnk, self.use_2cta_instrs, self.c_layout, self.c_dtype)
        self.epi_tile_n = cute.size(self.epi_tile[1])

        self.num_acc_stage, self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma, self.mma_tiler, self.a_dtype, self.b_dtype, self.epi_tile, self.c_dtype, self.c_layout, self.sf_dtype, self.sf_vec_size, self.smem_capacity, self.occupancy
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage)
        self.b_w1_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage)
        self.b_w3_smem_layout_staged = sm100_utils.make_smem_layout_b(tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage)
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage)
        self.sfb_w1_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage)
        self.sfb_w3_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage)
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(self.c_dtype, self.c_layout, self.epi_tile, self.num_c_stage)

        self.overlapping_accum = self.num_acc_stage == 1
        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // sf_atom_mn) * mma_inst_tile_k
        self.num_sfb_tmem_cols = (self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn) * mma_inst_tile_k
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + 2 * self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = (
            self.cta_tile_shape_mnk[1] * self.num_acc_stage
            if not self.overlapping_accum
            else self.cta_tile_shape_mnk[1] * 2 - self.num_sf_tmem_cols
        )
        self.iter_acc_early_release_in_epilogue = self.num_sf_tmem_cols // self.epi_tile_n

    def _compute_grid(self, total_num_clusters, max_active_clusters):
        import cutlass

        problem_shape_ntile_mnl = (
            self.cluster_shape_mn[0],
            self.cluster_shape_mn[1],
            total_num_clusters
        )

        tile_sched_params = utils.PersistentTileSchedulerParams(
            problem_shape_ntile_mnl, (*self.cluster_shape_mn, 1)
        )

        grid = utils.StaticPersistentGroupTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    @cute.jit
    def __call__(
        self,
        initial_a, initial_b_w1, initial_b_w3, initial_sfa, initial_sfb_w1, initial_sfb_w3, initial_c,
        group_count, problem_shape_mnkl, ptrs_absfsfbc, strides_abc,
        total_num_clusters, tensormaps, max_active_clusters, stream
    ):
        self.a_dtype = initial_a.element_type
        self.b_dtype = initial_b_w1.element_type
        self.sf_dtype = initial_sfa.element_type
        self.c_dtype = initial_c.element_type

        self.a_major_mode = utils.LayoutEnum.from_tensor(initial_a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(initial_b_w1).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(initial_c)
        
        self._setup_attributes()

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.a_major_mode, self.b_major_mode, self.sf_dtype, self.sf_vec_size, self.cta_group, self.mma_inst_shape_mn
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.a_major_mode, self.b_major_mode, self.sf_dtype, self.sf_vec_size, cute.nvgpu.tcgen05.CtaGroup.ONE, self.mma_inst_shape_mn_sfb
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(a_op, initial_a, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)

        b_w1_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_w1_smem_layout = cute.slice_(self.b_w1_smem_layout_staged, (None, None, None, 0))
        tma_atom_b_w1, tma_tensor_b_w1 = cute.nvgpu.make_tiled_tma_atom_B(b_w1_op, initial_b_w1, b_w1_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)

        b_w3_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_w3_smem_layout = cute.slice_(self.b_w3_smem_layout_staged, (None, None, None, 0))
        tma_atom_b_w3, tma_tensor_b_w3 = cute.nvgpu.make_tiled_tma_atom_B(b_w3_op, initial_b_w3, b_w3_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)

        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, None, 0))
        sfa_m_dummy = 16384 * 128
        sfa_k_dummy = 16384 * 32
        dummy_sfa_lyt = blockscaled_utils.tile_atom_to_shape_SF((sfa_m_dummy, sfa_k_dummy, 1), self.sf_vec_size)
        dummy_sfa_tensor = cute.make_tensor(initial_sfa.iterator, dummy_sfa_lyt)
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op, dummy_sfa_tensor, sfa_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )

        sfb_w1_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_w1_smem_layout = cute.slice_(self.sfb_w1_smem_layout_staged, (None, None, None, 0))
        sfb_n_dummy = 16384 * 128
        sfb_k_dummy = 16384 * 32
        dummy_sfb_lyt = blockscaled_utils.tile_atom_to_shape_SF((sfb_n_dummy, sfb_k_dummy, 1), self.sf_vec_size)
        dummy_sfb_w1_tensor = cute.make_tensor(initial_sfb_w1.iterator, dummy_sfb_lyt)
        tma_atom_sfb_w1, tma_tensor_sfb_w1 = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_w1_op, dummy_sfb_w1_tensor, sfb_w1_smem_layout, self.mma_tiler_sfb, tiled_mma_sfb, self.cluster_layout_sfb_vmnk.shape
        )

        sfb_w3_op = sm100_utils.cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, tiled_mma.thr_id)
        sfb_w3_smem_layout = cute.slice_(self.sfb_w3_smem_layout_staged, (None, None, None, 0))
        dummy_sfb_w3_tensor = cute.make_tensor(initial_sfb_w3.iterator, dummy_sfb_lyt)
        tma_atom_sfb_w3, tma_tensor_sfb_w3 = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_w3_op, dummy_sfb_w3_tensor, sfb_w3_smem_layout, self.mma_tiler_sfb, tiled_mma_sfb, self.cluster_layout_sfb_vmnk.shape
        )

        if cutlass.const_expr(self.cta_tile_shape_mnk[1] == 192):
            x = tma_tensor_sfb_w1.stride[0][1]
            y = cute.ceil_div(tma_tensor_sfb_w1.shape[0][1], 4)
            new_shape = ((tma_tensor_sfb_w1.shape[0][0], ((2, 2), y)), tma_tensor_sfb_w1.shape[1], tma_tensor_sfb_w1.shape[2])
            new_stride = ((tma_tensor_sfb_w1.stride[0][0], ((x, x), 3 * x)), tma_tensor_sfb_w1.stride[1], tma_tensor_sfb_w1.stride[2])
            tma_tensor_sfb_w1 = cute.make_tensor(tma_tensor_sfb_w1.iterator, cute.make_layout(new_shape, stride=new_stride))
            tma_tensor_sfb_w3 = cute.make_tensor(tma_tensor_sfb_w3.iterator, cute.make_layout(new_shape, stride=new_stride))

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_w1_copy_size = cute.size_in_bytes(self.b_dtype, b_w1_smem_layout)
        b_w3_copy_size = cute.size_in_bytes(self.b_dtype, b_w3_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_w1_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_w1_smem_layout)
        sfb_w3_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_w3_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_w1_copy_size + b_w3_copy_size + sfa_copy_size + sfb_w1_copy_size + sfb_w3_copy_size) * atom_thr_size

        epi_smem_layout = cute.slice_(self.c_smem_layout_staged, (None, None, 0))
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), initial_c, epi_smem_layout, self.epi_tile
        )

        self.tile_sched_params, grid = self._compute_grid(total_num_clusters, max_active_clusters)
        self.buffer_align_bytes = 1024
        self.size_tensormap_in_i64 = self.num_tensormaps * self.bytes_per_tensormap // 8

        @cute.struct
        class SharedStorage:
            tensormap_buffer: cute.struct.MemRange[cutlass.Int64, self.size_tensormap_in_i64]
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            ab_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            acc_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sC: cute.struct.Align[
                cute.struct.MemRange[self.c_dtype, cute.cosize(self.c_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(self.a_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB_w1: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_w1_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sB_w3: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(self.b_w3_smem_layout_staged.outer)],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB_w1: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_w1_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sSFB_w3: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(self.sfb_w3_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
        self.shared_storage = SharedStorage
        self.epilogue_op = lambda x: x

        args = (
            tiled_mma, tiled_mma_sfb,
            tma_atom_a, tma_tensor_a, 
            tma_atom_b_w1, tma_tensor_b_w1, tma_atom_b_w3, tma_tensor_b_w3,
            tma_atom_sfa, tma_tensor_sfa, 
            tma_atom_sfb_w1, tma_tensor_sfb_w1, tma_atom_sfb_w3, tma_tensor_sfb_w3,
            tma_atom_c, tma_tensor_c,
            self.cluster_layout_vmnk, self.cluster_layout_sfb_vmnk,
            self.a_smem_layout_staged, self.b_w1_smem_layout_staged, self.b_w3_smem_layout_staged,
            self.sfa_smem_layout_staged, self.sfb_w1_smem_layout_staged, self.sfb_w3_smem_layout_staged,
            self.c_smem_layout_staged, self.epi_tile, self.tile_sched_params,
            group_count, problem_shape_mnkl, ptrs_absfsfbc, strides_abc, tensormaps
        )
        
        self.kernel(*args).launch(
            grid=grid, block=[self.threads_per_cta, 1, 1], cluster=(*self.cluster_shape_mn, 1),
            stream=stream, min_blocks_per_mp=1
        )


    @cute.kernel
    def kernel(
        self,
        tiled_mma, tiled_mma_sfb,
        tma_atom_a, tma_tensor_a, 
        tma_atom_b_w1, tma_tensor_b_w1, tma_atom_b_w3, tma_tensor_b_w3,
        tma_atom_sfa, tma_tensor_sfa, 
        tma_atom_sfb_w1, tma_tensor_sfb_w1, tma_atom_sfb_w3, tma_tensor_sfb_w3,
        tma_atom_c, tma_tensor_c,
        cluster_layout_vmnk, cluster_layout_sfb_vmnk,
        a_smem_layout_staged, b_w1_smem_layout_staged, b_w3_smem_layout_staged,
        sfa_smem_layout_staged, sfb_w1_smem_layout_staged, sfb_w3_smem_layout_staged,
        epi_smem_layout_staged, epi_tile, tile_sched_params,
        group_count, problem_sizes_mnkl, ptrs_abc, strides_abc, tensormaps
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b_w1)
            cpasync.prefetch_descriptor(tma_atom_b_w3)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb_w1)
            cpasync.prefetch_descriptor(tma_atom_sfb_w3)
            cpasync.prefetch_descriptor(tma_atom_c)

        use_2cta_instrs = False
        bid = cute.arch.block_idx()
        mma_tile_coord_v = bid[0] % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tensormap_smem_ptr = storage.tensormap_buffer.data_ptr()
        tensormap_a_smem_ptr = tensormap_smem_ptr
        tensormap_b_w1_smem_ptr = tensormap_a_smem_ptr + self.bytes_per_tensormap // 8
        tensormap_b_w3_smem_ptr = tensormap_b_w1_smem_ptr + self.bytes_per_tensormap // 8
        tensormap_sfa_smem_ptr = tensormap_b_w3_smem_ptr + self.bytes_per_tensormap // 8
        tensormap_sfb_w1_smem_ptr = tensormap_sfa_smem_ptr + self.bytes_per_tensormap // 8
        tensormap_sfb_w3_smem_ptr = tensormap_sfb_w1_smem_ptr + self.bytes_per_tensormap // 8
        tensormap_c_smem_ptr = tensormap_sfb_w3_smem_ptr + self.bytes_per_tensormap // 8

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilog_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sC = storage.sC.get_tensor(epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB_w1 = storage.sB_w1.get_tensor(b_w1_smem_layout_staged.outer, swizzle=b_w1_smem_layout_staged.inner)
        sB_w3 = storage.sB_w3.get_tensor(b_w3_smem_layout_staged.outer, swizzle=b_w3_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB_w1 = storage.sSFB_w1.get_tensor(sfb_w1_smem_layout_staged)
        sSFB_w3 = storage.sSFB_w3.get_tensor(sfb_w3_smem_layout_staged)

        a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2) if self.is_a_mcast else None
        b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1) if self.is_b_mcast else None
        sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2) if self.is_a_mcast else None
        sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1) if self.is_sfb_mcast else None

        gA_mkl = cute.local_tile(tma_tensor_a, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB_w1_nkl = cute.local_tile(tma_tensor_b_w1, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gB_w3_nkl = cute.local_tile(tma_tensor_b_w3, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gSFA_mkl = cute.local_tile(tma_tensor_sfa, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gSFB_w1_nkl = cute.local_tile(tma_tensor_sfb_w1, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None))
        gSFB_w3_nkl = cute.local_tile(tma_tensor_sfb_w3, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None))
        gC_mnl = cute.local_tile(tma_tensor_c, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB_w1 = thr_mma.partition_B(gB_w1_nkl)
        tCgB_w3 = thr_mma.partition_B(gB_w3_nkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)
        tCgSFB_w1 = thr_mma_sfb.partition_B(gSFB_w1_nkl)
        tCgSFB_w3 = thr_mma_sfb.partition_B(gSFB_w3_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(tma_atom_a, block_in_cluster_coord_vmnk[2], a_cta_layout, cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB_w1, tBgB_w1 = cpasync.tma_partition(tma_atom_b_w1, block_in_cluster_coord_vmnk[1], b_cta_layout, cute.group_modes(sB_w1, 0, 3), cute.group_modes(tCgB_w1, 0, 3))
        tBsB_w3, tBgB_w3 = cpasync.tma_partition(tma_atom_b_w3, block_in_cluster_coord_vmnk[1], b_cta_layout, cute.group_modes(sB_w3, 0, 3), cute.group_modes(tCgB_w3, 0, 3))
        
        sfa_cta_layout = a_cta_layout
        tAsSFA, tAgSFA = cute.nvgpu.cpasync.tma_partition(tma_atom_sfa, block_in_cluster_coord_vmnk[2], sfa_cta_layout, cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3))
        tAsSFA, tAgSFA = cute.filter_zeros(tAsSFA), cute.filter_zeros(tAgSFA)
        
        sfb_cta_layout = cute.make_layout(cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
        tBsSFB_w1, tBgSFB_w1 = cute.nvgpu.cpasync.tma_partition(tma_atom_sfb_w1, block_in_cluster_coord_sfb_vmnk[1], sfb_cta_layout, cute.group_modes(sSFB_w1, 0, 3), cute.group_modes(tCgSFB_w1, 0, 3))
        tBsSFB_w1, tBgSFB_w1 = cute.filter_zeros(tBsSFB_w1), cute.filter_zeros(tBgSFB_w1)
        tBsSFB_w3, tBgSFB_w3 = cute.nvgpu.cpasync.tma_partition(tma_atom_sfb_w3, block_in_cluster_coord_sfb_vmnk[1], sfb_cta_layout, cute.group_modes(sSFB_w3, 0, 3), cute.group_modes(tCgSFB_w3, 0, 3))
        tBsSFB_w3, tBgSFB_w3 = cute.filter_zeros(tBsSFB_w3), cute.filter_zeros(tBgSFB_w3)

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB_w1 = tiled_mma.make_fragment_B(sB_w1)
        tCrB_w3 = tiled_mma.make_fragment_B(sB_w3)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        grid_dim = cute.arch.grid_dim()
        tensormap_workspace_idx = bid[2] * grid_dim[1] * grid_dim[0] + bid[1] * grid_dim[0] + bid[0]

        tensormap_manager = utils.TensorMapManager(self.tensormap_update_mode, self.bytes_per_tensormap)
        tensormap_a_ptr = tensormap_manager.get_tensormap_ptr(tensormaps[(tensormap_workspace_idx, 0, None)].iterator)
        tensormap_b_w1_ptr = tensormap_manager.get_tensormap_ptr(tensormaps[(tensormap_workspace_idx, 1, None)].iterator)
        tensormap_b_w3_ptr = tensormap_manager.get_tensormap_ptr(tensormaps[(tensormap_workspace_idx, 2, None)].iterator)
        tensormap_sfa_ptr = tensormap_manager.get_tensormap_ptr(tensormaps[(tensormap_workspace_idx, 3, None)].iterator)
        tensormap_sfb_w1_ptr = tensormap_manager.get_tensormap_ptr(tensormaps[(tensormap_workspace_idx, 4, None)].iterator)
        tensormap_sfb_w3_ptr = tensormap_manager.get_tensormap_ptr(tensormaps[(tensormap_workspace_idx, 5, None)].iterator)
        tensormap_c_ptr = tensormap_manager.get_tensormap_ptr(tensormaps[(tensormap_workspace_idx, 6, None)].iterator)

        tile_sched = utils.StaticPersistentGroupTileScheduler.create(
            tile_sched_params, bid, grid_dim, self.cluster_tile_shape_mnk,
            utils.create_initial_search_state(), group_count, problem_sizes_mnkl
        )
        initial_work_tile_info = tile_sched.initial_work_tile_info()

        if warp_idx == self.tma_warp_id and initial_work_tile_info.is_valid_tile:
            tensormap_init_done = cutlass.Boolean(False)
            last_group_idx = cutlass.Int32(-1)
            work_tile = initial_work_tile_info
            ab_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_ab_stage)

            while work_tile.is_valid_tile:
                grouped_gemm_cta_tile_info = work_tile.group_search_result
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                is_k_tile_cnt_zero = cur_k_tile_cnt == 0
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx

                if not is_k_tile_cnt_zero:
                    is_group_changed = cur_group_idx != last_group_idx
                    if is_group_changed:
                        pshape = (grouped_gemm_cta_tile_info.problem_shape_m, grouped_gemm_cta_tile_info.problem_shape_n, grouped_gemm_cta_tile_info.problem_shape_k)
                        real_tensor_a = self.make_tensor_for_tensormap_update(cur_group_idx, self.a_dtype, pshape, strides_abc, ptrs_abc, 0)
                        real_tensor_b_w1 = self.make_tensor_for_tensormap_update(cur_group_idx, self.b_dtype, pshape, strides_abc, ptrs_abc, 1)
                        real_tensor_b_w3 = self.make_tensor_for_tensormap_update(cur_group_idx, self.b_dtype, pshape, strides_abc, ptrs_abc, 2)
                        real_tensor_sfa = self.make_tensor_for_tensormap_update(cur_group_idx, self.sf_dtype, pshape, strides_abc, ptrs_abc, 3)
                        real_tensor_sfb_w1 = self.make_tensor_for_tensormap_update(cur_group_idx, self.sf_dtype, pshape, strides_abc, ptrs_abc, 4)
                        real_tensor_sfb_w3 = self.make_tensor_for_tensormap_update(cur_group_idx, self.sf_dtype, pshape, strides_abc, ptrs_abc, 5)

                        # wait tensormap initialization complete before update
                        if not tensormap_init_done:
                            self.tensormap_ab_init_barrier.arrive_and_wait()
                            tensormap_manager.fence_tensormap_initialization()
                            tensormap_init_done = True

                        tensormap_manager.update_tensormap(
                            (real_tensor_a, real_tensor_b_w1, real_tensor_b_w3, real_tensor_sfa, real_tensor_sfb_w1, real_tensor_sfb_w3),
                            (tma_atom_a, tma_atom_b_w1, tma_atom_b_w3, tma_atom_sfa, tma_atom_sfb_w1, tma_atom_sfb_w3),
                            (tensormap_a_ptr, tensormap_b_w1_ptr, tensormap_b_w3_ptr, tensormap_sfa_ptr, tensormap_sfb_w1_ptr, tensormap_sfb_w3_ptr),
                            self.tma_warp_id,
                            (tensormap_a_smem_ptr, tensormap_b_w1_smem_ptr, tensormap_b_w3_smem_ptr, tensormap_sfa_smem_ptr, tensormap_sfb_w1_smem_ptr, tensormap_sfb_w3_smem_ptr),
                        )

                    mma_tile_coord_mnl = (
                        grouped_gemm_cta_tile_info.cta_tile_idx_m // cute.size(tiled_mma.thr_id.shape),
                        grouped_gemm_cta_tile_info.cta_tile_idx_n,
                        0,
                    )

                    tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                    tBgB_w1_slice = tBgB_w1[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
                    tBgB_w3_slice = tBgB_w3[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
                    tAgSFA_slice = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                    tBgSFB_w1_slice = tBgSFB_w1[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
                    tBgSFB_w3_slice = tBgSFB_w3[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                    ab_producer_state.reset_count()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < cur_k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)

                    # ensure tensormap update has completed before using it
                    if is_group_changed:
                        tensormap_manager.fence_tensormap_update(tensormap_a_ptr)
                        tensormap_manager.fence_tensormap_update(tensormap_b_w1_ptr)
                        tensormap_manager.fence_tensormap_update(tensormap_b_w3_ptr)
                        tensormap_manager.fence_tensormap_update(tensormap_sfa_ptr)
                        tensormap_manager.fence_tensormap_update(tensormap_sfb_w1_ptr)
                        tensormap_manager.fence_tensormap_update(tensormap_sfb_w3_ptr)

                    for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                        ab_pipeline.producer_acquire(ab_producer_state, peek_ab_empty_status)

                        bar_ptr = ab_pipeline.producer_get_barrier(ab_producer_state)
                        cute.copy(tma_atom_a, tAgA_slice[(None, ab_producer_state.count)], tAsA[(None, ab_producer_state.index)], tma_bar_ptr=bar_ptr, mcast_mask=a_full_mcast_mask, tma_desc_ptr=tensormap_manager.get_tensormap_ptr(tensormap_a_ptr, cute.AddressSpace.generic))
                        cute.copy(tma_atom_b_w1, tBgB_w1_slice[(None, ab_producer_state.count)], tBsB_w1[(None, ab_producer_state.index)], tma_bar_ptr=bar_ptr, mcast_mask=b_full_mcast_mask, tma_desc_ptr=tensormap_manager.get_tensormap_ptr(tensormap_b_w1_ptr, cute.AddressSpace.generic))
                        cute.copy(tma_atom_b_w3, tBgB_w3_slice[(None, ab_producer_state.count)], tBsB_w3[(None, ab_producer_state.index)], tma_bar_ptr=bar_ptr, mcast_mask=b_full_mcast_mask, tma_desc_ptr=tensormap_manager.get_tensormap_ptr(tensormap_b_w3_ptr, cute.AddressSpace.generic))
                        cute.copy(tma_atom_sfa, tAgSFA_slice[(None, ab_producer_state.count)], tAsSFA[(None, ab_producer_state.index)], tma_bar_ptr=bar_ptr, mcast_mask=sfa_full_mcast_mask, tma_desc_ptr=tensormap_manager.get_tensormap_ptr(tensormap_sfa_ptr, cute.AddressSpace.generic))
                        cute.copy(tma_atom_sfb_w1, tBgSFB_w1_slice[(None, ab_producer_state.count)], tBsSFB_w1[(None, ab_producer_state.index)], tma_bar_ptr=bar_ptr, mcast_mask=sfb_full_mcast_mask, tma_desc_ptr=tensormap_manager.get_tensormap_ptr(tensormap_sfb_w1_ptr, cute.AddressSpace.generic))
                        cute.copy(tma_atom_sfb_w3, tBgSFB_w3_slice[(None, ab_producer_state.count)], tBsSFB_w3[(None, ab_producer_state.index)], tma_bar_ptr=bar_ptr, mcast_mask=sfb_full_mcast_mask, tma_desc_ptr=tensormap_manager.get_tensormap_ptr(tensormap_sfb_w3_ptr, cute.AddressSpace.generic))

                        ab_producer_state.advance()
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if ab_producer_state.count < cur_k_tile_cnt:
                            peek_ab_empty_status = ab_pipeline.producer_try_acquire(ab_producer_state)
                else:
                    # If tensormap initialization is not done, wait for it to complete
                    if not tensormap_init_done:
                        self.tensormap_ab_init_barrier.arrive_and_wait()
                        tensormap_manager.fence_tensormap_initialization()
                        tensormap_init_done = True

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx


            ab_pipeline.producer_tail(ab_producer_state)

        if warp_idx == self.mma_warp_id and initial_work_tile_info.is_valid_tile:
            # Bar sync for retrieve tmem ptr from shared mem
            tmem.wait_for_alloc()

            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            
            acc1_tmem_ptr = acc_tmem_ptr
            tCtAcc1_base = cute.make_tensor(acc1_tmem_ptr, tCtAcc_fake.layout)

            acc2_tmem_ptr = acc_tmem_ptr + self.num_accumulator_tmem_cols
            tCtAcc2_base = cute.make_tensor(acc2_tmem_ptr, tCtAcc_fake.layout)

            sfa_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + 2 * self.num_accumulator_tmem_cols, dtype=self.sf_dtype)
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(tiled_mma, self.mma_tiler, self.sf_vec_size, cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)))
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            sfb_w1_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + 2 * self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols, dtype=self.sf_dtype)
            tCtSFB_w1_layout = blockscaled_utils.make_tmem_layout_sfb(tiled_mma, self.mma_tiler, self.sf_vec_size, cute.slice_(sfb_w1_smem_layout_staged, (None, None, None, 0)))
            tCtSFB_w1 = cute.make_tensor(sfb_w1_tmem_ptr, tCtSFB_w1_layout)

            sfb_w3_tmem_ptr = cute.recast_ptr(acc_tmem_ptr + 2 * self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols + self.num_sfb_tmem_cols, dtype=self.sf_dtype)
            tCtSFB_w3_layout = blockscaled_utils.make_tmem_layout_sfb(tiled_mma, self.mma_tiler, self.sf_vec_size, cute.slice_(sfb_w3_smem_layout_staged, (None, None, None, 0)))
            tCtSFB_w3 = cute.make_tensor(sfb_w3_tmem_ptr, tCtSFB_w3_layout)

            tiled_copy_s2t_sfa, tCsSFA_compact_s2t, tCtSFA_compact_s2t = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            tiled_copy_s2t_sfb_w1, tCsSFB_w1_compact_s2t, tCtSFB_w1_compact_s2t = self.mainloop_s2t_copy_and_partition(sSFB_w1, tCtSFB_w1)
            tiled_copy_s2t_sfb_w3, tCsSFB_w3_compact_s2t, tCtSFB_w3_compact_s2t = self.mainloop_s2t_copy_and_partition(sSFB_w3, tCtSFB_w3)

            work_tile = initial_work_tile_info
            ab_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_ab_stage)
            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)

            while work_tile.is_valid_tile:
                grouped_gemm_cta_tile_info = work_tile.group_search_result
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                is_k_tile_cnt_zero = cur_k_tile_cnt == 0
                
                acc_stage_index = acc_producer_state.index
                tCtAcc1 = tCtAcc1_base[(None, None, None, acc_stage_index)]
                tCtAcc2 = tCtAcc2_base[(None, None, None, acc_stage_index)]

                # Peek (try_wait) AB buffer full for k_tile = 0
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    if ab_consumer_state.count < cur_k_tile_cnt:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                    # Wait for accumulator buffer empty
                    if not is_k_tile_cnt_zero:
                        acc_pipeline.producer_acquire(acc_producer_state)

                    tCtSFB_w1_mma = tCtSFB_w1
                    tCtSFB_w3_mma = tCtSFB_w3
                    accumulate_flag = cutlass.Boolean(0)

                    # Mma mainloop
                    for k_tile in cutlass.range(0, cur_k_tile_cnt, 1, unroll=1):
                        ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)

                        s2t_stage_coord = (None, None, None, None, ab_consumer_state.index)
                        cute.copy(tiled_copy_s2t_sfa, tCsSFA_compact_s2t[s2t_stage_coord], tCtSFA_compact_s2t)
                        cute.copy(tiled_copy_s2t_sfb_w1, tCsSFB_w1_compact_s2t[s2t_stage_coord], tCtSFB_w1_compact_s2t)
                        cute.copy(tiled_copy_s2t_sfb_w3, tCsSFB_w3_compact_s2t[s2t_stage_coord], tCtSFB_w3_compact_s2t)

                        num_kblocks = cute.size(tCrA, mode=[2])
                        
                        # Pass 1: W1
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, accumulate_flag)
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
                            tiled_mma.set(tcgen05.Field.SFB, tCtSFB_w1_mma[sf_kblock_coord].iterator)

                            cute.gemm(tiled_mma, tCtAcc1, tCrA[kblock_coord], tCrB_w1[kblock_coord], tCtAcc1)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                        # Pass 2: W3
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, accumulate_flag)
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_coord = (None, None, kblock_idx, ab_consumer_state.index)
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator)
                            tiled_mma.set(tcgen05.Field.SFB, tCtSFB_w3_mma[sf_kblock_coord].iterator)

                            cute.gemm(tiled_mma, tCtAcc2, tCrA[kblock_coord], tCrB_w3[kblock_coord], tCtAcc2)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        
                        accumulate_flag = cutlass.Boolean(1)
                        
                        ab_pipeline.consumer_release(ab_consumer_state)

                        ab_consumer_state.advance()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if ab_consumer_state.count < cur_k_tile_cnt:
                            peek_ab_full_status = ab_pipeline.consumer_try_wait(ab_consumer_state)

                    # Async arrive accumulator buffer full
                    if not is_k_tile_cnt_zero:
                        acc_pipeline.producer_commit(acc_producer_state)
                        acc_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()


            acc_pipeline.producer_tail(acc_producer_state)
        if warp_idx < self.mma_warp_id and initial_work_tile_info.is_valid_tile:
            # Initialize tensormaps for A, B_w1, B_w3, SFA, SFB_w1, SFB_w3 from TMA atoms
            tensormap_manager.init_tensormap_from_atom(tma_atom_a, tensormap_a_smem_ptr, self.epilog_warp_id[0])
            tensormap_manager.init_tensormap_from_atom(tma_atom_b_w1, tensormap_b_w1_smem_ptr, self.epilog_warp_id[0])
            tensormap_manager.init_tensormap_from_atom(tma_atom_b_w3, tensormap_b_w3_smem_ptr, self.epilog_warp_id[0])
            tensormap_manager.init_tensormap_from_atom(tma_atom_sfa, tensormap_sfa_smem_ptr, self.epilog_warp_id[0])
            tensormap_manager.init_tensormap_from_atom(tma_atom_sfb_w1, tensormap_sfb_w1_smem_ptr, self.epilog_warp_id[0])
            tensormap_manager.init_tensormap_from_atom(tma_atom_sfb_w3, tensormap_sfb_w3_smem_ptr, self.epilog_warp_id[0])
            # Signal tensormap initialization has finished
            self.tensormap_ab_init_barrier.arrive_and_wait()

            # Initialize tensormap for C
            tensormap_manager.init_tensormap_from_atom(tma_atom_c, tensormap_c_smem_ptr, self.epilog_warp_id[0])

            # Alloc tensor memory buffer
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            
            acc1_tmem_ptr = acc_tmem_ptr
            tCtAcc1_base = cute.make_tensor(acc1_tmem_ptr, tCtAcc_fake.layout)

            acc2_tmem_ptr = acc_tmem_ptr + self.num_accumulator_tmem_cols
            tCtAcc2_base = cute.make_tensor(acc2_tmem_ptr, tCtAcc_fake.layout)

            epi_tidx = tidx
            tiled_copy_t2r_1, tTR_tAcc1_base, tTR_rAcc1 = self.epilog_tmem_copy_and_partition(epi_tidx, tCtAcc1_base, tCgC, epi_tile)
            tiled_copy_t2r_2, tTR_tAcc2_base, tTR_rAcc2 = self.epilog_tmem_copy_and_partition(epi_tidx, tCtAcc2_base, tCgC, epi_tile)

            tTR_rC = cute.make_rmem_tensor(tTR_rAcc1.shape, self.c_dtype)
            tiled_copy_r2s, tRS_rC, tRS_sC = self.epilog_smem_copy_and_partition(tiled_copy_t2r_1, tTR_rC, epi_tidx, sC)
            tma_atom_c, bSG_sC, bSG_gC_partitioned = self.epilog_gmem_copy_and_partition(epi_tidx, tma_atom_c, tCgC, epi_tile, sC)

            work_tile = initial_work_tile_info

            # wait tensormap initialization complete before update
            tensormap_manager.fence_tensormap_initialization()

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)

            c_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_warp * len(self.epilog_warp_id))
            c_pipeline = pipeline.PipelineTmaStore.create(num_stages=self.num_c_stage, producer_group=c_producer_group)

            last_group_idx = cutlass.Int32(-1)

            while work_tile.is_valid_tile:
                grouped_gemm_cta_tile_info = work_tile.group_search_result
                cur_group_idx = grouped_gemm_cta_tile_info.group_idx
                cur_k_tile_cnt = grouped_gemm_cta_tile_info.cta_tile_count_k
                is_k_tile_cnt_zero = cur_k_tile_cnt == 0
                is_group_changed = cur_group_idx != last_group_idx

                # We still need to store 0s when k_tile_cnt is 0
                if is_group_changed:
                    pshape = (grouped_gemm_cta_tile_info.problem_shape_m, grouped_gemm_cta_tile_info.problem_shape_n, grouped_gemm_cta_tile_info.problem_shape_k)
                    real_tensor_c = self.make_tensor_for_tensormap_update(cur_group_idx, self.c_dtype, pshape, strides_abc, ptrs_abc, 6)
                    tensormap_manager.update_tensormap(
                        (real_tensor_c,), (tma_atom_c,), (tensormap_c_ptr,), self.epilog_warp_id[0], (tensormap_c_smem_ptr,)
                    )

                mma_tile_coord_mnl = (
                    grouped_gemm_cta_tile_info.cta_tile_idx_m // cute.size(tiled_mma.thr_id.shape),
                    grouped_gemm_cta_tile_info.cta_tile_idx_n,
                    0,
                )

                bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]
                tTR_tAcc1 = tTR_tAcc1_base[(None, None, None, None, None, acc_consumer_state.index)]
                tTR_tAcc2 = tTR_tAcc2_base[(None, None, None, None, None, acc_consumer_state.index)]

                if not is_k_tile_cnt_zero:
                    acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc1 = cute.group_modes(tTR_tAcc1, 3, cute.rank(tTR_tAcc1))
                tTR_tAcc2 = cute.group_modes(tTR_tAcc2, 3, cute.rank(tTR_tAcc2))
                bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

                # ensure the update to tensormap has completed before using it
                if is_group_changed:
                    if warp_idx == self.epilog_warp_id[0]:
                        tensormap_manager.fence_tensormap_update(tensormap_c_ptr)

                subtile_cnt = cute.size(tTR_tAcc1.shape, mode=[3])
                num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt

                for subtile_idx in range(subtile_cnt):
                    epi_buffer = (num_prev_subtiles + subtile_idx) % self.num_c_stage
                    tTR_tAcc1_mn = tTR_tAcc1[(None, None, None, subtile_idx)]
                    tTR_tAcc2_mn = tTR_tAcc2[(None, None, None, subtile_idx)]
                    
                    if not is_k_tile_cnt_zero:
                        cute.copy(tiled_copy_t2r_1, tTR_tAcc1_mn, tTR_rAcc1)
                        cute.copy(tiled_copy_t2r_2, tTR_tAcc2_mn, tTR_rAcc2)
                        acc_vec1 = tiled_copy_r2s.retile(tTR_rAcc1).load()
                        acc_vec2 = tiled_copy_r2s.retile(tTR_rAcc2).load()
                        
                        # match Triton kernel's SwiGLU order: silu(acc2) * acc1
                        silu_vec = acc_vec2 / (1.0 + cute.exp(0.0 - acc_vec2))
                        acc_vec = silu_vec * acc_vec1
                        
                        tRS_rC.store(acc_vec.to(self.c_dtype))
                    else:
                        tRS_rC.fill(0)

                    cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, epi_buffer)])
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()

                    if warp_idx == self.epilog_warp_id[0]:
                        cute.copy(tma_atom_c, bSG_sC[(None, epi_buffer)], bSG_gC[(None, subtile_idx)], tma_desc_ptr=tensormap_manager.get_tensormap_ptr(tensormap_c_ptr, cute.AddressSpace.generic))
                        c_pipeline.producer_commit()
                        c_pipeline.producer_acquire()
                    self.epilog_sync_barrier.arrive_and_wait()

                if not is_k_tile_cnt_zero:
                    with cute.arch.elect_one():
                        acc_pipeline.consumer_release(acc_consumer_state)
                    acc_consumer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                last_group_idx = cur_group_idx
                last_group_idx = cur_group_idx

            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(acc_tmem_ptr)
            c_pipeline.producer_tail()

    @cute.jit
    def make_tensor_for_tensormap_update(self, group_idx, dtype, problem_shape_mnk, strides_abc, tensor_address_abc, tensor_index):
        ptr_i64 = tensor_address_abc[(group_idx, tensor_index)]
        #if not isclass(dtype) or not issubclass(dtype, cutlass.Numeric):
        #    raise TypeError("dtype error")
        tensor_gmem_ptr = cute.make_ptr(dtype, ptr_i64, cute.AddressSpace.gmem, assumed_align=16)

        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)
        
        # A, B, C strides are passed in strides_abc. SFA/SFB are dense per expert.
        if cutlass.const_expr(tensor_index == 0):
            strides_tensor_gmem = strides_abc[(group_idx, 0, None)]
            strides_tensor_reg = cute.make_rmem_tensor(cute.make_layout(2), strides_abc.element_type)
            cute.autovec_copy(strides_tensor_gmem, strides_tensor_reg)
            stride_mn = strides_tensor_reg[0]
            stride_k = strides_tensor_reg[1]
            return cute.make_tensor(tensor_gmem_ptr, cute.make_layout((problem_shape_mnk[0], problem_shape_mnk[2], c1), stride=(stride_mn, stride_k, c0)))
        elif cutlass.const_expr(tensor_index == 1):
            strides_tensor_gmem = strides_abc[(group_idx, tensor_index, None)]
            strides_tensor_reg = cute.make_rmem_tensor(cute.make_layout(2), strides_abc.element_type)
            cute.autovec_copy(strides_tensor_gmem, strides_tensor_reg)
            stride_mn = strides_tensor_reg[0]
            stride_k = strides_tensor_reg[1]
            return cute.make_tensor(tensor_gmem_ptr, cute.make_layout((problem_shape_mnk[1], problem_shape_mnk[2], c1), stride=(stride_mn, stride_k, c0)))
        elif cutlass.const_expr(tensor_index == 2):
            strides_tensor_gmem = strides_abc[(group_idx, tensor_index, None)]
            strides_tensor_reg = cute.make_rmem_tensor(cute.make_layout(2), strides_abc.element_type)
            cute.autovec_copy(strides_tensor_gmem, strides_tensor_reg)
            stride_mn = strides_tensor_reg[0]
            stride_k = strides_tensor_reg[1]
            return cute.make_tensor(tensor_gmem_ptr, cute.make_layout((problem_shape_mnk[1], problem_shape_mnk[2], c1), stride=(stride_mn, stride_k, c0)))
        elif cutlass.const_expr(tensor_index == 3):  # SFA
            sfa_layout = blockscaled_utils.tile_atom_to_shape_SF((problem_shape_mnk[0], problem_shape_mnk[2], c1), self.sf_vec_size)
            return cute.make_tensor(tensor_gmem_ptr, sfa_layout)
        elif cutlass.const_expr(tensor_index == 4):  # SFB_W1
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF((problem_shape_mnk[1], problem_shape_mnk[2], c1), self.sf_vec_size)
            return cute.make_tensor(tensor_gmem_ptr, sfb_layout)
        elif cutlass.const_expr(tensor_index == 5):  # SFB_W3
            sfb_layout = blockscaled_utils.tile_atom_to_shape_SF((problem_shape_mnk[1], problem_shape_mnk[2], c1), self.sf_vec_size)
            return cute.make_tensor(tensor_gmem_ptr, sfb_layout)
        elif cutlass.const_expr(tensor_index == 6):  # C
            strides_tensor_gmem = strides_abc[(group_idx, 6, None)]
            strides_tensor_reg = cute.make_rmem_tensor(cute.make_layout(2), strides_abc.element_type)
            cute.autovec_copy(strides_tensor_gmem, strides_tensor_reg)
            stride_mn = strides_tensor_reg[0]
            stride_k = strides_tensor_reg[1]
            return cute.make_tensor(tensor_gmem_ptr, cute.make_layout((problem_shape_mnk[0], problem_shape_mnk[1], c1), stride=(stride_mn, stride_k, c0)))
        else:
            return cute.make_tensor(tensor_gmem_ptr, cute.make_layout((problem_shape_mnk[0], problem_shape_mnk[1], c1)))

    def mainloop_s2t_copy_and_partition(self, sSF, tSF):
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(self, tidx, tAcc, gC_mnl, epi_tile):
        copy_atom_t2r = sm100_utils.get_tmem_load_op(self.cta_tile_shape_mnk, self.c_layout, self.c_dtype, self.acc_dtype, epi_tile, False)
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        gC_mnl_epi = cute.flat_divide(gC_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tTR_gC = thr_copy_t2r.partition_D(gC_mnl_epi)
        tTR_rAcc = cute.make_rmem_tensor(tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    def epilog_smem_copy_and_partition(self, tiled_copy_t2r, tTR_rC, tidx, sC):
        copy_atom_r2s = sm100_utils.get_smem_store_op(
            self.c_layout, self.c_dtype, self.acc_dtype, tiled_copy_t2r
        )
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sC = thr_copy_r2s.partition_D(sC)
        tRS_rC = tiled_copy_r2s.retile(tTR_rC)
        return tiled_copy_r2s, tRS_rC, tRS_sC

    def epilog_gmem_copy_and_partition(self, tidx, tma_atom_c, tCgC, epi_tile, sC):
        gC_epi = cute.flat_divide(
            tCgC[((None, None), 0, 0, None, None, None)], epi_tile
        )
        sC_for_tma_partition = cute.group_modes(sC, 0, 2)
        gC_for_tma_partition = cute.group_modes(gC_epi, 0, 2)
        bSG_sC, bSG_gC = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            sC_for_tma_partition,
            gC_for_tma_partition,
        )
        return tma_atom_c, bSG_sC, bSG_gC
