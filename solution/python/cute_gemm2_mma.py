import os

import torch


_COMPILED = None
_COMPILE_ERROR = None
_STATE = None
_B_FP16_CACHE = {}
_METADATA_CACHE = {}

TARGET_BLOCK_M = {
    14107: 128,
}
H = 7168
I_SIZE = 2048
E_LOCAL = 32
TOP_K = 8
MAX_PADDED = max(t * TOP_K + E_LOCAL * block_m for t, block_m in TARGET_BLOCK_M.items())
EXPERT_OUT_DTYPE = torch.bfloat16


def _load_cute_stack():
    # The vendored grouped GEMM follows the official CuTe example path and
    # passes CuTe runtime tensors directly. Do not force TVM-FFI here; doing so
    # makes the compiled call reject runtime._Tensor arguments.
    os.environ.pop("CUTE_DSL_ENABLE_TVM_FFI", None)
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cutlass.utils as utils

    import cute_grouped_gemm_sm100 as grouped_gemm

    return cutlass, cute, cutlass_torch, utils, grouped_gemm


def _get_b_fp16(gemm2_weights, gemm2_weights_scale):
    key = (
        str(gemm2_weights.device),
        int(gemm2_weights.data_ptr()),
        int(gemm2_weights_scale.data_ptr()),
        tuple(gemm2_weights.shape),
    )
    cached = _B_FP16_CACHE.get(key)
    if cached is not None:
        return cached

    pieces = []
    for expert in range(E_LOCAL):
        scale = gemm2_weights_scale[expert].repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        pieces.append((gemm2_weights[expert].float() * scale).half())
    packed = torch.stack(pieces, dim=0).contiguous()
    _B_FP16_CACHE[key] = packed
    return packed


def _compute_total_num_clusters(problem_sizes_mnkl, cluster_tile_shape_mn):
    total = 0
    for m, n, _, _ in problem_sizes_mnkl:
        total += ((m + cluster_tile_shape_mn[0] - 1) // cluster_tile_shape_mn[0]) * (
            (n + cluster_tile_shape_mn[1] - 1) // cluster_tile_shape_mn[1]
        )
    return total


def _build_or_get_state():
    global _COMPILED, _COMPILE_ERROR, _STATE

    if _COMPILED is not None:
        return _STATE
    if _COMPILE_ERROR is not None:
        raise RuntimeError(f"cached CuTe grouped GEMM2 compile failure: {_COMPILE_ERROR!r}")

    try:
        cutlass, cute, cutlass_torch, utils, grouped_gemm = _load_cute_stack()

        ab_dtype = cutlass.Float16
        c_dtype = cutlass.BFloat16
        acc_dtype = cutlass.Float32
        a_major = "k"
        b_major = "k"
        c_major = "n"
        mma_tiler_mn = (128, 128)
        cluster_shape_mn = (1, 1)
        use_2cta_instrs = False
        num_groups = E_LOCAL

        initial_a = grouped_gemm.create_tensor_and_stride(1, 8, 8, a_major == "m", ab_dtype)[2]
        initial_b = grouped_gemm.create_tensor_and_stride(1, 8, 8, b_major == "n", ab_dtype)[2]
        initial_c = grouped_gemm.create_tensor_and_stride(1, 4, 4, c_major == "m", c_dtype)[2]

        hardware_info = utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1])
        sm_count = hardware_info.get_max_active_clusters(1)
        tensormap_shape = (
            sm_count,
            grouped_gemm.GroupedGemmKernel.num_tensormaps,
            grouped_gemm.GroupedGemmKernel.bytes_per_tensormap // 8,
        )
        tensor_of_tensormap, tensor_of_tensormap_torch = cutlass_torch.cute_tensor_like(
            torch.empty(tensormap_shape, dtype=torch.int64),
            cutlass.Int64,
            is_dynamic_layout=False,
        )

        tensor_of_problem_sizes, problem_sizes_torch = cutlass_torch.cute_tensor_like(
            torch.empty((num_groups, 4), dtype=torch.int32),
            cutlass.Int32,
            is_dynamic_layout=False,
            assumed_align=16,
        )
        tensor_of_strides, strides_torch = cutlass_torch.cute_tensor_like(
            torch.empty((num_groups, 3, 2), dtype=torch.int32),
            cutlass.Int32,
            is_dynamic_layout=False,
            assumed_align=16,
        )
        tensor_of_ptrs, ptrs_torch = cutlass_torch.cute_tensor_like(
            torch.empty((num_groups, 3), dtype=torch.int64),
            cutlass.Int64,
            is_dynamic_layout=False,
            assumed_align=16,
        )

        gemm = grouped_gemm.GroupedGemmKernel(
            acc_dtype,
            use_2cta_instrs,
            mma_tiler_mn,
            cluster_shape_mn,
            utils.TensorMapUpdateMode.SMEM,
        )
        cluster_tile_shape_mn = mma_tiler_mn
        total_num_clusters = ((MAX_PADDED + cluster_tile_shape_mn[0] - 1) // cluster_tile_shape_mn[0]) * (
            (H + cluster_tile_shape_mn[1] - 1) // cluster_tile_shape_mn[1]
        )
        current_stream = cutlass_torch.default_stream()

        _COMPILED = cute.compile(
            gemm,
            initial_a,
            initial_b,
            initial_c,
            num_groups,
            tensor_of_problem_sizes,
            tensor_of_strides,
            tensor_of_ptrs,
            total_num_clusters,
            tensor_of_tensormap,
            max_active_clusters,
            current_stream,
            options="--opt-level 3",
        )
        _STATE = {
            "cutlass_torch": cutlass_torch,
            "initial_a": initial_a,
            "initial_b": initial_b,
            "initial_c": initial_c,
            "problem_sizes_torch": problem_sizes_torch,
            "strides_torch": strides_torch,
            "ptrs_torch": ptrs_torch,
            "tensor_of_problem_sizes": tensor_of_problem_sizes,
            "tensor_of_strides": tensor_of_strides,
            "tensor_of_ptrs": tensor_of_ptrs,
            "tensor_of_tensormap": tensor_of_tensormap,
            "current_stream": current_stream,
        }
        return _STATE
    except Exception as exc:
        _COMPILE_ERROR = exc
        raise


def _update_group_metadata(
    state,
    intermediate,
    b_fp16,
    expert_out,
    block_offsets,
    block_m,
    block_offsets_host=None,
):
    offsets = block_offsets_host if block_offsets_host is not None else block_offsets.detach().cpu().tolist()
    offset_key = tuple(int(x) for x in offsets)
    layout_key = (
        int(block_m),
        int(intermediate.data_ptr()),
        int(b_fp16.data_ptr()),
        int(expert_out.data_ptr()),
        tuple(intermediate.stride()),
        tuple(b_fp16.stride()),
        tuple(expert_out.stride()),
    )

    meta_key = (
        offset_key,
        layout_key,
    )
    if state.get("metadata_key") == meta_key:
        return

    cached = _METADATA_CACHE.get(meta_key)
    if cached is not None:
        problems_t, strides_t, ptrs_t = cached
        state["problem_sizes_torch"].copy_(problems_t, non_blocking=True)
        state["strides_torch"].copy_(strides_t, non_blocking=True)
        state["ptrs_torch"].copy_(ptrs_t, non_blocking=True)
        state["metadata_key"] = meta_key
        return

    problems = []
    strides = []
    ptrs = []
    a_elem = intermediate.element_size()
    b_elem = b_fp16.element_size()
    c_elem = expert_out.element_size()

    for expert in range(E_LOCAL):
        row_begin = int(offsets[expert]) * block_m
        row_end = int(offsets[expert + 1]) * block_m
        m = max(row_end - row_begin, 0)
        if m == 0:
            m = block_m
            row_begin = int(offsets[expert]) * block_m
        problems.append((m, H, I_SIZE, 1))
        strides.append(
            (
                (intermediate.stride(0), intermediate.stride(1)),
                (b_fp16.stride(1), b_fp16.stride(2)),
                (expert_out.stride(0), expert_out.stride(1)),
            )
        )
        ptrs.append(
            (
                int(intermediate.data_ptr()) + row_begin * intermediate.stride(0) * a_elem,
                int(b_fp16.data_ptr()) + expert * b_fp16.stride(0) * b_elem,
                int(expert_out.data_ptr()) + row_begin * expert_out.stride(0) * c_elem,
            )
        )

    problems_t = torch.tensor(problems, dtype=torch.int32, device=intermediate.device)
    strides_t = torch.tensor(strides, dtype=torch.int32, device=intermediate.device)
    ptrs_t = torch.tensor(ptrs, dtype=torch.int64, device=intermediate.device)
    state["problem_sizes_torch"].copy_(problems_t, non_blocking=True)
    state["strides_torch"].copy_(strides_t, non_blocking=True)
    state["ptrs_torch"].copy_(ptrs_t, non_blocking=True)
    state["metadata_key"] = meta_key
    if len(_METADATA_CACHE) > 8:
        _METADATA_CACHE.clear()
    _METADATA_CACHE[meta_key] = (problems_t, strides_t, ptrs_t)


def run(
    intermediate,
    gemm2_weights,
    gemm2_weights_scale,
    expert_out,
    block_offsets,
    total_tokens,
    block_m,
    block_offsets_host=None,
):
    expected_block_m = TARGET_BLOCK_M.get(int(total_tokens))
    if expected_block_m is None:
        raise ValueError(f"CuTe GEMM2 expects T in {tuple(TARGET_BLOCK_M)}, got {int(total_tokens)}")
    if int(block_m) != expected_block_m:
        raise ValueError(f"CuTe GEMM2 expects block_m={expected_block_m}, got {int(block_m)}")
    if intermediate.dtype != torch.float16:
        raise TypeError(f"CuTe GEMM2 expects fp16 intermediate, got {intermediate.dtype}")
    if expert_out.dtype != EXPERT_OUT_DTYPE:
        raise TypeError(f"CuTe GEMM2 expects {EXPERT_OUT_DTYPE} expert_out, got {expert_out.dtype}")

    state = _build_or_get_state()
    b_fp16 = _get_b_fp16(gemm2_weights, gemm2_weights_scale)
    _update_group_metadata(
        state,
        intermediate,
        b_fp16,
        expert_out,
        block_offsets,
        int(block_m),
        block_offsets_host=block_offsets_host,
    )

    _COMPILED(
        state["initial_a"],
        state["initial_b"],
        state["initial_c"],
        state["tensor_of_problem_sizes"],
        state["tensor_of_strides"],
        state["tensor_of_ptrs"],
        state["tensor_of_tensormap"],
        state["current_stream"],
    )
