import importlib.util
from pathlib import Path

import torch
import triton


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


_pure = _load_local_module("t901_static_cap_pure_triton_impl", "pure_triton_impl.py")

T_TARGET = 901
H = 7168
I_SIZE = 2048
E_LOCAL = 32
TOP_K = 8
BLOCK_M = 64
STATIC_PID_M = 64
SORT_BLOCK_ITEMS = 256
INTER_DTYPE = torch.float16
EXPERT_OUT_DTYPE = torch.bfloat16

_buf_cache = {}
_routing_cache = {}
_sort_cache = {}


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

    device = hidden_states.device
    local_start = _pure._cached_host_scalar(local_expert_offset, int)
    scale_factor = _pure._cached_host_scalar(routed_scaling_factor, float)

    if output is None:
        output = torch.empty((T, H), dtype=torch.bfloat16, device=device)

    max_padded = T * TOP_K + E_LOCAL * BLOCK_M
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

    buf_key = (T, BLOCK_M, INTER_DTYPE, EXPERT_OUT_DTYPE)
    if buf_key in _buf_cache:
        intermediate, expert_out = _buf_cache[buf_key]
    else:
        intermediate = torch.empty((max_padded, I_SIZE), dtype=INTER_DTYPE, device=device)
        expert_out = torch.empty((max_padded, H), dtype=EXPERT_OUT_DTYPE, device=device)
        _buf_cache[buf_key] = (intermediate, expert_out)

    grid1 = lambda META: (STATIC_PID_M * triton.cdiv(I_SIZE, META['BLOCK_N']),)
    _pure._fused_moe_gemm1_swiglu_kernel[grid1](
        A_ptr=hidden_states,
        A_scale_ptr=hidden_states_scale,
        B_ptr=gemm1_weights,
        C_ptr=intermediate,
        B_scale_ptr=gemm1_weights_scale,
        token_ids_ptr=sorted_token_ids,
        block_offsets_ptr=block_offsets,
        total_blocks_ptr=total_blocks,
        MAX_PID_M=STATIC_PID_M, T=T, H=H, N=4096, K=H,
        stride_at=hidden_states.stride(0), stride_ah=hidden_states.stride(1),
        stride_as0=hidden_states_scale.stride(0), stride_as1=hidden_states_scale.stride(1),
        stride_be=gemm1_weights.stride(0), stride_bn=gemm1_weights.stride(1), stride_bh=gemm1_weights.stride(2),
        stride_cm=intermediate.stride(0), stride_cn=intermediate.stride(1),
        stride_bse=gemm1_weights_scale.stride(0), stride_bsn=gemm1_weights_scale.stride(1), stride_bsh=gemm1_weights_scale.stride(2),
        E_LOCAL=E_LOCAL,
        BLOCK_M=BLOCK_M,
        USE_FP16_INTER=True,
    )

    grid2 = lambda META: (STATIC_PID_M * triton.cdiv(H, META['BLOCK_N']),)
    _pure._fused_moe_gemm2_t901_kernel[grid2](
        A_ptr=intermediate,
        B_ptr=gemm2_weights,
        C_ptr=expert_out,
        B_scale_ptr=gemm2_weights_scale,
        token_weights_ptr=sorted_weights,
        block_offsets_ptr=block_offsets,
        total_blocks_ptr=total_blocks,
        MAX_PID_M=STATIC_PID_M, N=H, K=I_SIZE,
        stride_am=intermediate.stride(0), stride_ak=intermediate.stride(1),
        stride_be=gemm2_weights.stride(0), stride_bn=gemm2_weights.stride(1), stride_bk=gemm2_weights.stride(2),
        stride_cm=expert_out.stride(0), stride_cn=expert_out.stride(1),
        stride_bse=gemm2_weights_scale.stride(0), stride_bsn=gemm2_weights_scale.stride(1), stride_bsk=gemm2_weights_scale.stride(2),
        E_LOCAL=E_LOCAL,
        BLOCK_M=BLOCK_M,
        USE_FP16_INTER=True,
        USE_BF16_EXPERT_OUT=True,
    )

    grid3 = lambda META: (triton.cdiv(T, META['BLOCK_T']) * triton.cdiv(H, META['BLOCK_N']),)
    _pure._token_reduce_kernel[grid3](
        expert_out_ptr=expert_out,
        output_ptr=output,
        scatter_map_ptr=scatter_map,
        T_val=T,
        N=H,
        TOP_K=TOP_K,
        stride_em=expert_out.stride(0),
        stride_en=expert_out.stride(1),
        stride_ot=output.stride(0),
        stride_on=output.stride(1),
    )
    return output
