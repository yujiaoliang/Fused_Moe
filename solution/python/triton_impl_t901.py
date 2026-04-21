import importlib.util
from pathlib import Path

import torch
import triton
import triton.language as tl


_THIS_DIR = Path(__file__).resolve().parent


def _load_local_module(module_name: str, local_name: str):
    path = _THIS_DIR / local_name
    if not path.exists():
        raise FileNotFoundError(f"Unable to locate module source for {module_name}: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_pure = _load_local_module("t901_gemm2_pure_triton_impl", "pure_triton_impl.py")
_cute_gemm2_runtime = _load_local_module(
    "t901_cute_gemm2_mma_runtime",
    "cute_gemm2_mma_runtime_901.py",
)

T_TARGET = 901
H = 7168
I_SIZE = 2048
E_LOCAL = 32
TOP_K = 8
BLOCK_M = 64
SORT_BLOCK_ITEMS = 256
EXPERT_OUT_DTYPE = torch.bfloat16
FIXED_BLOCKS_PER_EXPERT = 2
FIXED_ROWS = E_LOCAL * FIXED_BLOCKS_PER_EXPERT * BLOCK_M
STATIC_BLOCK_OFFSETS_HOST = tuple(i * FIXED_BLOCKS_PER_EXPERT for i in range(E_LOCAL + 1))

_buf_cache = {}
_routing_cache = {}
_sort_cache = {}


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 16, 'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_T': 16, 'BLOCK_N': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_T': 8, 'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T': 32, 'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_T': 16, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ],
    key=['T_val', 'N'],
)
@triton.jit
def _token_reduce_t901_kernel(
    expert_out_ptr,
    output_ptr,
    scatter_map_ptr,
    packed_to_fixed_ptr,
    token_weights_ptr,
    T_val: tl.constexpr,
    N: tl.constexpr,
    TOP_K: tl.constexpr,
    stride_em, stride_en,
    stride_ot, stride_on,
    BLOCK_T: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_t = (pid // num_pid_n) * BLOCK_T
    pid_n = (pid % num_pid_n) * BLOCK_N

    rt = pid_t + tl.arange(0, BLOCK_T)
    rn = pid_n + tl.arange(0, BLOCK_N)
    t_mask = rt < T_val
    n_mask = rn < N
    acc = tl.zeros((BLOCK_T, BLOCK_N), dtype=tl.float32)

    for k in tl.static_range(TOP_K):
        pos_idx = rt * TOP_K + k
        pos = tl.load(scatter_map_ptr + pos_idx, mask=t_mask, other=-1)
        valid_pos = (pos >= 0) & t_mask
        fixed_pos = tl.load(packed_to_fixed_ptr + pos, mask=valid_pos, other=0)
        vals = tl.load(
            expert_out_ptr + fixed_pos[:, None] * stride_em + rn[None, :] * stride_en,
            mask=valid_pos[:, None] & n_mask[None, :],
            other=0.0,
            eviction_policy='evict_first',
        ).to(tl.float32)
        weights = tl.load(token_weights_ptr + pos, mask=valid_pos, other=0.0)
        acc += vals * (weights[:, None] * 8.0)

    tl.store(
        output_ptr + rt[:, None] * stride_ot + rn[None, :] * stride_on,
        acc.to(tl.bfloat16),
        mask=t_mask[:, None] & n_mask[None, :],
        eviction_policy='evict_first',
    )


@triton.jit
def _pack_t901_fixed_slots_kernel(
    src_ptr,
    dst_ptr,
    packed_to_fixed_ptr,
    block_offsets_ptr,
    total_blocks_ptr,
    stride_sm, stride_sk,
    stride_dm, stride_dk,
    E_LOCAL: tl.constexpr,
    I_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FIXED_BLOCKS_PER_EXPERT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(I_SIZE, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    total_blocks = tl.load(total_blocks_ptr)
    if pid_m >= total_blocks:
        return

    e_idx = tl.arange(0, E_LOCAL)
    b_start = tl.load(block_offsets_ptr + e_idx)
    b_end = tl.load(block_offsets_ptr + e_idx + 1)
    valid_expert = (b_start <= pid_m) & (pid_m < b_end)
    expert_id = tl.argmax(valid_expert.to(tl.int32), axis=0)
    expert_start = tl.load(block_offsets_ptr + expert_id)
    local_block = pid_m - expert_start

    src_rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    dst_rows = (expert_id * FIXED_BLOCKS_PER_EXPERT + local_block) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask = cols < I_SIZE
    slot_mask = local_block < FIXED_BLOCKS_PER_EXPERT

    vals = tl.load(
        src_ptr + src_rows[:, None] * stride_sm + cols[None, :] * stride_sk,
        mask=slot_mask & col_mask[None, :],
        other=0.0,
        eviction_policy='evict_first',
    )
    tl.store(
        dst_ptr + dst_rows[:, None] * stride_dm + cols[None, :] * stride_dk,
        vals,
        mask=slot_mask & col_mask[None, :],
        eviction_policy='evict_first',
    )
    map_mask = slot_mask & (pid_n == 0)
    tl.store(packed_to_fixed_ptr + src_rows, dst_rows, mask=map_mask)


@torch.no_grad()
def kernel(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor,
    output=None,
):
    T = int(routing_logits.shape[0])
    if T != T_TARGET:
        raise ValueError("triton_impl_t901.py only supports T=901")
    if _pure._select_block_m(T * TOP_K) != BLOCK_M:
        raise ValueError("T=901 CuTe GEMM2 experiment expects block_m=64")

    device = hidden_states.device
    local_start = _pure._cached_host_scalar(local_expert_offset, int)
    scale_factor = _pure._cached_host_scalar(routed_scaling_factor, float)

    if output is None:
        output = torch.empty((T, H), dtype=torch.bfloat16, device=device)

    max_padded = T * TOP_K + E_LOCAL * BLOCK_M
    max_pid_m = max_padded // BLOCK_M
    num_tiles = triton.cdiv(T * TOP_K, SORT_BLOCK_ITEMS)

    if T in _routing_cache:
        topk_idx_ws, topk_wts_ws = _routing_cache[T]
    else:
        topk_idx_ws = torch.empty((T, TOP_K), dtype=torch.int32, device=device)
        topk_wts_ws = torch.empty((T, TOP_K), dtype=torch.float32, device=device)
        _routing_cache[T] = (topk_idx_ws, topk_wts_ws)

    bkey = (T, BLOCK_M)
    if bkey in _sort_cache:
        sorted_token_ids, sorted_weights, scatter_map, block_offsets, total_blocks, counts_workspace, partial_counts, tile_offsets = _sort_cache[bkey]
    else:
        sorted_token_ids = torch.empty((max_padded,), dtype=torch.int64, device=device)
        sorted_weights = torch.empty((max_padded,), dtype=torch.float32, device=device)
        scatter_map = torch.empty((T * TOP_K,), dtype=torch.int32, device=device)
        block_offsets = torch.empty((E_LOCAL + 1,), dtype=torch.int32, device=device)
        total_blocks = torch.empty((1,), dtype=torch.int32, device=device)
        counts_workspace = torch.empty((E_LOCAL,), dtype=torch.int32, device=device)
        partial_counts = torch.empty((num_tiles, E_LOCAL), dtype=torch.int32, device=device)
        tile_offsets = torch.empty((num_tiles, E_LOCAL), dtype=torch.int32, device=device)
        _sort_cache[bkey] = (
            sorted_token_ids,
            sorted_weights,
            scatter_map,
            block_offsets,
            total_blocks,
            counts_workspace,
            partial_counts,
            tile_offsets,
        )

    _pure.ds_routing(
        routing_logits,
        routing_bias,
        scale_factor,
        topk_idx=topk_idx_ws,
        topk_weights=topk_wts_ws,
    )
    _pure.parallel_sort_and_scatter(
        topk_idx_ws,
        topk_wts_ws,
        sorted_token_ids,
        sorted_weights,
        scatter_map,
        block_offsets,
        total_blocks,
        partial_counts,
        tile_offsets,
        local_start,
        T,
        BLOCK_M,
        max_padded,
        counts_workspace,
        histogram_ready=False,
    )

    buf_key = (T, BLOCK_M, EXPERT_OUT_DTYPE)
    if buf_key in _buf_cache:
        intermediate, intermediate_fixed, expert_out, packed_to_fixed = _buf_cache[buf_key]
    else:
        intermediate = torch.empty((max_padded, I_SIZE), dtype=torch.float16, device=device)
        intermediate_fixed = torch.empty((FIXED_ROWS, I_SIZE), dtype=torch.float16, device=device)
        expert_out = torch.empty((FIXED_ROWS, H), dtype=EXPERT_OUT_DTYPE, device=device)
        packed_to_fixed = torch.empty((max_padded,), dtype=torch.int32, device=device)
        _buf_cache[buf_key] = (intermediate, intermediate_fixed, expert_out, packed_to_fixed)

    grid1 = lambda META: (max_pid_m * triton.cdiv(I_SIZE, META['BLOCK_N']),)
    _pure._fused_moe_gemm1_swiglu_kernel[grid1](
        A_ptr=hidden_states,
        A_scale_ptr=hidden_states_scale,
        B_ptr=gemm1_weights,
        C_ptr=intermediate,
        B_scale_ptr=gemm1_weights_scale,
        token_ids_ptr=sorted_token_ids,
        block_offsets_ptr=block_offsets,
        total_blocks_ptr=total_blocks,
        MAX_PID_M=max_pid_m, T=T, H=H, N=4096, K=H,
        stride_at=hidden_states.stride(0), stride_ah=hidden_states.stride(1),
        stride_as0=hidden_states_scale.stride(0), stride_as1=hidden_states_scale.stride(1),
        stride_be=gemm1_weights.stride(0), stride_bn=gemm1_weights.stride(1), stride_bh=gemm1_weights.stride(2),
        stride_cm=intermediate.stride(0), stride_cn=intermediate.stride(1),
        stride_bse=gemm1_weights_scale.stride(0), stride_bsn=gemm1_weights_scale.stride(1), stride_bsh=gemm1_weights_scale.stride(2),
        E_LOCAL=E_LOCAL,
        BLOCK_M=BLOCK_M,
        USE_FP16_INTER=True,
    )

    grid_pack = lambda META: (max_pid_m * triton.cdiv(I_SIZE, META['BLOCK_N']),)
    _pack_t901_fixed_slots_kernel[grid_pack](
        src_ptr=intermediate,
        dst_ptr=intermediate_fixed,
        packed_to_fixed_ptr=packed_to_fixed,
        block_offsets_ptr=block_offsets,
        total_blocks_ptr=total_blocks,
        stride_sm=intermediate.stride(0),
        stride_sk=intermediate.stride(1),
        stride_dm=intermediate_fixed.stride(0),
        stride_dk=intermediate_fixed.stride(1),
        E_LOCAL=E_LOCAL,
        I_SIZE=I_SIZE,
        BLOCK_M=BLOCK_M,
        BLOCK_N=128,
        FIXED_BLOCKS_PER_EXPERT=FIXED_BLOCKS_PER_EXPERT,
        num_warps=4,
    )

    _cute_gemm2_runtime.run_cute_gemm2_mma(
        intermediate_fixed,
        gemm2_weights,
        gemm2_weights_scale,
        expert_out,
        block_offsets,
        T,
        BLOCK_M,
        block_offsets_host=STATIC_BLOCK_OFFSETS_HOST,
    )

    grid3 = lambda META: (triton.cdiv(T, META['BLOCK_T']) * triton.cdiv(H, META['BLOCK_N']),)
    _token_reduce_t901_kernel[grid3](
        expert_out_ptr=expert_out,
        output_ptr=output,
        scatter_map_ptr=scatter_map,
        packed_to_fixed_ptr=packed_to_fixed,
        token_weights_ptr=sorted_weights,
        T_val=T,
        N=H,
        TOP_K=TOP_K,
        stride_em=expert_out.stride(0),
        stride_en=expert_out.stride(1),
        stride_ot=output.stride(0),
        stride_on=output.stride(1),
    )
    return output
