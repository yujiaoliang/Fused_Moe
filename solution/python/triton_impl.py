import importlib.util
import os
from pathlib import Path

os.environ["DISABLE_LLVM_OPT"] = "disable-lsr"
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


_pure = _load_local_module("hybrid_pure_triton_impl", "pure_triton_impl.py")
_cute_gemm2_mma_runtime = _load_local_module(
    "hybrid_cute_gemm2_mma_runtime",
    "cute_gemm2_mma_runtime.py",
)
_cute_gemm1_mma_runtime = _load_local_module(
    "hybrid_cute_gemm1_mma_runtime",
    "cute_gemm1_mma_runtime.py",
)

H = 7168
I_SIZE = 2048
E_LOCAL = 32
TOP_K = 8
SORT_BLOCK_ITEMS = 256
TOKEN_SCATTER_BLOCK_TOKENS = 8
BLOCK_M = 128
TARGET_TS = (11948,)
TARGET_BLOCK_M = {11948: BLOCK_M}
CUTE_GEMM1_RAW_DTYPE = torch.bfloat16
T11948_CUTE_GEMM2_EXPERT_OUT_DTYPE = torch.bfloat16


@triton.jit
def _dequantize_gemm1_sorted_a_kernel(
    hidden_states_ptr,
    hidden_states_scale_ptr,
    sorted_token_ids_ptr,
    a_sorted_ptr,
    NUM_ROWS: tl.constexpr,
    T_val: tl.constexpr,
    H: tl.constexpr,
    stride_at, stride_ah,
    stride_as0, stride_as1,
    stride_sm, stride_sh,
    BLOCK_M: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    token = tl.load(sorted_token_ids_ptr + rm, mask=rm < NUM_ROWS, other=T_val)
    valid = (rm < NUM_ROWS) & (token < T_val)
    safe_token = tl.where(token < T_val, token, 0)
    h_mask = rh < H

    vals = tl.load(
        hidden_states_ptr + safe_token[:, None] * stride_at + rh[None, :] * stride_ah,
        mask=valid[:, None] & h_mask[None, :],
        other=0.0,
        eviction_policy='evict_first',
    ).to(tl.float32)
    scales = tl.load(
        hidden_states_scale_ptr + pid_h * stride_as0 + safe_token * stride_as1,
        mask=valid,
        other=0.0,
        eviction_policy='evict_first',
    )
    tl.store(
        a_sorted_ptr + rm[:, None] * stride_sm + rh[None, :] * stride_sh,
        (vals * scales[:, None]).to(tl.float16),
        mask=(rm[:, None] < NUM_ROWS) & h_mask[None, :],
        eviction_policy='evict_first',
    )


@triton.jit
def _cute_gemm1_swiglu_epilogue_kernel(
    raw_ptr,
    intermediate_ptr,
    NUM_ROWS: tl.constexpr,
    I_SIZE: tl.constexpr,
    stride_rm, stride_rn,
    stride_im, stride_in,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < NUM_ROWS) & (rn[None, :] < I_SIZE)

    w1 = tl.load(
        raw_ptr + rm[:, None] * stride_rm + rn[None, :] * stride_rn,
        mask=mask,
        other=0.0,
        eviction_policy='evict_first',
    ).to(tl.float32)
    w3 = tl.load(
        raw_ptr + rm[:, None] * stride_rm + (rn[None, :] + I_SIZE) * stride_rn,
        mask=mask,
        other=0.0,
        eviction_policy='evict_first',
    ).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-w3))
    out = (w3 * sig) * w1 * 0.125
    tl.store(
        intermediate_ptr + rm[:, None] * stride_im + rn[None, :] * stride_in,
        out.to(tl.float16),
        mask=mask,
        eviction_policy='evict_first',
    )


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
def _token_reduce_weighted_kernel(
    expert_out_ptr,
    output_ptr,
    scatter_map_ptr,
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
        load_mask = valid_pos[:, None] & n_mask[None, :]

        e_ptrs = expert_out_ptr + pos[:, None] * stride_em + rn[None, :] * stride_en
        vals = tl.load(e_ptrs, mask=load_mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
        weights = tl.load(token_weights_ptr + pos, mask=valid_pos, other=0.0)
        acc += vals * (weights[:, None] * 8.0)

    o_ptrs = output_ptr + rt[:, None] * stride_ot + rn[None, :] * stride_on
    tl.store(
        o_ptrs,
        acc.to(tl.bfloat16),
        mask=(t_mask[:, None] & n_mask[None, :]),
        eviction_policy='evict_first',
    )


_buf_cache = {}
_cute_gemm1_buf_cache = {}
_routing_cache = {}
_routing_hist_cache = {}
_sort_cache = {}
_token_sort_cache = {}


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
    block_m = TARGET_BLOCK_M.get(T)
    if block_m is None:
        raise ValueError(f"hybrid CuTe path only supports T in {TARGET_TS}, got {T}")
    if _pure._select_block_m(T * TOP_K) != block_m:
        raise ValueError(f"hybrid CuTe path expects block_m={block_m} for T={T}")

    device = hidden_states.device
    local_start = _pure._cached_host_scalar(local_expert_offset, int)
    scale_factor = _pure._cached_host_scalar(routed_scaling_factor, float)

    if output is None:
        output = torch.empty((T, H), dtype=torch.bfloat16, device=device)

    max_padded = T * TOP_K + E_LOCAL * block_m
    max_pid_m = max_padded // block_m
    num_tiles = triton.cdiv(T * TOP_K, SORT_BLOCK_ITEMS)

    rkey = T
    if rkey in _routing_cache:
        topk_idx_ws, topk_wts_ws = _routing_cache[rkey]
    else:
        topk_idx_ws = torch.empty((T, TOP_K), dtype=torch.int32, device=device)
        topk_wts_ws = torch.empty((T, TOP_K), dtype=torch.float32, device=device)
        _routing_cache[rkey] = (topk_idx_ws, topk_wts_ws)

    bkey = (T, block_m)
    if bkey in _sort_cache:
        sorted_token_ids, sorted_weights, scatter_map, block_offsets, total_blocks, counts_workspace, write_offsets_ws = _sort_cache[bkey]
    else:
        sorted_token_ids = torch.empty((max_padded,), dtype=torch.int64, device=device)
        sorted_weights = torch.empty((max_padded,), dtype=torch.float32, device=device)
        scatter_map = torch.empty((T * TOP_K,), dtype=torch.int32, device=device)
        block_offsets = torch.empty((E_LOCAL + 1,), dtype=torch.int32, device=device)
        total_blocks = torch.empty((1,), dtype=torch.int32, device=device)
        counts_workspace = torch.empty((E_LOCAL,), dtype=torch.int32, device=device)
        write_offsets_ws = torch.empty((E_LOCAL,), dtype=torch.int32, device=device)
        _sort_cache[bkey] = (
            sorted_token_ids,
            sorted_weights,
            scatter_map,
            block_offsets,
            total_blocks,
            counts_workspace,
            write_offsets_ws,
        )

    if rkey in _routing_hist_cache:
        local_topk_idx_ws = _routing_hist_cache[rkey]
    else:
        local_topk_idx_ws = torch.empty((T, TOP_K), dtype=torch.int32, device=device)
        _routing_hist_cache[rkey] = local_topk_idx_ws

    _pure.ds_routing_with_histogram(
        routing_logits,
        routing_bias,
        scale_factor,
        local_topk_idx_ws,
        counts_workspace,
        local_start,
        topk_idx=topk_idx_ws,
        topk_weights=topk_wts_ws,
    )
    _pure.token_sort_and_scatter(
        local_topk_idx_ws,
        topk_wts_ws,
        sorted_token_ids,
        sorted_weights,
        scatter_map,
        block_offsets,
        total_blocks,
        counts_workspace,
        write_offsets_ws,
        T,
        block_m,
        max_padded,
    )

    block_offsets_host = block_offsets.detach().cpu().tolist()
    exact_pid_m = int(block_offsets_host[-1])
    if exact_pid_m <= 0:
        output.zero_()
        return output
    num_rows = exact_pid_m * block_m

    buf_key = (T, block_m, T11948_CUTE_GEMM2_EXPERT_OUT_DTYPE)
    if buf_key in _buf_cache:
        intermediate, expert_out = _buf_cache[buf_key]
    else:
        intermediate = torch.empty((max_padded, I_SIZE), dtype=torch.float16, device=device)
        expert_out = torch.empty((max_padded, H), dtype=T11948_CUTE_GEMM2_EXPERT_OUT_DTYPE, device=device)
        _buf_cache[buf_key] = (intermediate, expert_out)

    cute_gemm1_key = (T, block_m, CUTE_GEMM1_RAW_DTYPE)
    cute_gemm1_buffers = _cute_gemm1_buf_cache.get(cute_gemm1_key)
    if cute_gemm1_buffers is None:
        a_sorted_fp16 = torch.empty((max_padded, H), dtype=torch.float16, device=device)
        gemm1_raw = torch.empty((max_padded, I_SIZE * 2), dtype=CUTE_GEMM1_RAW_DTYPE, device=device)
        cute_gemm1_buffers = (a_sorted_fp16, gemm1_raw)
        _cute_gemm1_buf_cache[cute_gemm1_key] = cute_gemm1_buffers
    a_sorted_fp16, gemm1_raw = cute_gemm1_buffers

    _dequantize_gemm1_sorted_a_kernel[(triton.cdiv(num_rows, 16), triton.cdiv(H, 128))](
        hidden_states,
        hidden_states_scale,
        sorted_token_ids,
        a_sorted_fp16,
        NUM_ROWS=num_rows,
        T_val=T,
        H=H,
        stride_at=hidden_states.stride(0),
        stride_ah=hidden_states.stride(1),
        stride_as0=hidden_states_scale.stride(0),
        stride_as1=hidden_states_scale.stride(1),
        stride_sm=a_sorted_fp16.stride(0),
        stride_sh=a_sorted_fp16.stride(1),
        BLOCK_M=16,
        BLOCK_H=128,
        num_warps=8,
    )
    _cute_gemm1_mma_runtime.run_cute_gemm1_mma(
        a_sorted_fp16,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_raw,
        block_offsets,
        T,
        block_m,
        block_offsets_host=block_offsets_host,
    )
    _cute_gemm1_swiglu_epilogue_kernel[(triton.cdiv(num_rows, 16), triton.cdiv(I_SIZE, 128))](
        gemm1_raw,
        intermediate,
        NUM_ROWS=num_rows,
        I_SIZE=I_SIZE,
        stride_rm=gemm1_raw.stride(0),
        stride_rn=gemm1_raw.stride(1),
        stride_im=intermediate.stride(0),
        stride_in=intermediate.stride(1),
        BLOCK_M=16,
        BLOCK_N=128,
        num_warps=8,
    )

    _cute_gemm2_mma_runtime.run_cute_gemm2_mma(
        intermediate,
        gemm2_weights,
        gemm2_weights_scale,
        expert_out,
        block_offsets,
        T,
        block_m,
        block_offsets_host=block_offsets_host,
    )

    grid_weighted = lambda META: (triton.cdiv(T, META['BLOCK_T']) * triton.cdiv(H, META['BLOCK_N']),)
    _token_reduce_weighted_kernel[grid_weighted](
        expert_out_ptr=expert_out,
        output_ptr=output,
        scatter_map_ptr=scatter_map,
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
