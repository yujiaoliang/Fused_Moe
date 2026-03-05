"""
CUDA MoE Kernel — Optimized PyTorch/cuBLAS baseline for Track A.

Optimizations:
  1. Gather-first routing: normalize on [T,8] instead of [T,256]
  2. Bool group mask: avoid float→bool temporary allocation
  3. In-place masked_fill_ to avoid copy
  4. Pre-allocated output buffer cache
  5. torch.compile routing (fuses element-wise ops)
  6. Skip empty experts early

Architecture:
  1. Routing: sigmoid → group-filter → top-8 → normalized weights
  2. Per-expert loop: dequant(fp32) → cuBLAS matmul → SwiGLU → matmul → scatter-add
  3. Cast output FP32 → BF16

Testing: python scripts/pack_solution_simple.py --cuda && python check_modal.py
"""

import torch
import torch.nn.functional as F

# ── Constants (DeepSeek-V3 / R1) ──
H = 7168
I_SIZE = 2048
E_GLOBAL = 256
E_LOCAL = 32
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
QBLOCK = 128


# ═══════════════════════════════════════════════════════════════
# Routing (PyTorch) — Optimized DeepSeek-V3 no-aux routing
# ═══════════════════════════════════════════════════════════════

def _ds_routing_impl(logits, bias, scale_factor):
    T = logits.shape[0]
    s = torch.sigmoid(logits.float())
    b = bias.float().reshape(-1)
    sb = s + b

    gs = E_GLOBAL // N_GROUP
    sb_g = sb.view(T, N_GROUP, gs)
    top2, _ = torch.topk(sb_g, k=2, dim=2, largest=True, sorted=False)
    g_scores = top2.sum(dim=2)

    _, g_idx = torch.topk(g_scores, k=TOPK_GROUP, dim=1, largest=True, sorted=False)
    g_mask = torch.zeros(T, N_GROUP, dtype=torch.bool, device=logits.device)
    g_mask.scatter_(1, g_idx, True)
    s_mask = g_mask.unsqueeze(2).expand(T, N_GROUP, gs).reshape(T, E_GLOBAL)

    neg_inf = torch.finfo(torch.float32).min
    sb.masked_fill_(~s_mask, neg_inf)
    _, topk_idx = torch.topk(sb, k=TOP_K, dim=1, largest=True, sorted=False)

    topk_s = torch.gather(s, 1, topk_idx)
    topk_weights = topk_s / (topk_s.sum(dim=1, keepdim=True) + 1e-20) * scale_factor

    return topk_idx, topk_weights


try:
    ds_routing = torch.compile(_ds_routing_impl, mode="reduce-overhead", dynamic=True)
except Exception:
    ds_routing = _ds_routing_impl


# ═══════════════════════════════════════════════════════════════
# FP8 Block-Scale Dequantization
# ═══════════════════════════════════════════════════════════════

def dequant_hidden_states(hs_fp8, hs_scale):
    """[T, 7168] fp8 + [56, T] fp32 → [T, 7168] fp32"""
    T = hs_fp8.shape[0]
    return (hs_fp8.float().view(T, 56, 128) * hs_scale.T.unsqueeze(2)).reshape(T, H)


def dequant_weight(w_fp8, w_scale, out_rows, out_cols, scale_rows, scale_cols):
    """Generic fp8 block-scale dequant → fp32"""
    return (w_fp8.float().view(scale_rows, 128, scale_cols, 128)
            * w_scale[:, None, :, None]).reshape(out_rows, out_cols)


# ═══════════════════════════════════════════════════════════════
# Buffer cache
# ═══════════════════════════════════════════════════════════════
_buf_cache = {}


# ═══════════════════════════════════════════════════════════════
# Main Kernel Entry Point
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def kernel(
    routing_logits,           # [T, 256]          float32
    routing_bias,             # [256]             bfloat16
    hidden_states,            # [T, 7168]         float8_e4m3fn
    hidden_states_scale,      # [56, T]           float32
    gemm1_weights,            # [32, 4096, 7168]  float8_e4m3fn
    gemm1_weights_scale,      # [32, 32, 56]      float32
    gemm2_weights,            # [32, 7168, 2048]  float8_e4m3fn
    gemm2_weights_scale,      # [32, 56, 16]      float32
    local_expert_offset,      # int32 scalar
    routed_scaling_factor,    # float32 scalar
    output,                   # [T, 7168]         bfloat16
):
    T = routing_logits.shape[0]
    device = hidden_states.device
    local_start = int(local_expert_offset)

    # ── 1. Routing ──
    topk_idx, topk_weights = ds_routing(
        routing_logits, routing_bias, float(routed_scaling_factor)
    )

    # ── 2. Pre-allocated FP32 accumulation buffer ──
    bkey = T
    if bkey in _buf_cache:
        output_fp32 = _buf_cache[bkey]
        output_fp32.zero_()
    else:
        output_fp32 = torch.zeros((T, H), dtype=torch.float32, device=device)
        _buf_cache[bkey] = output_fp32

    # ── 3. Dequant hidden_states once ──
    hs_fp32 = dequant_hidden_states(hidden_states, hidden_states_scale)

    # ── 4. Per-expert loop ──
    local_end = local_start + E_LOCAL

    all_token_ids = torch.arange(T, device=device).unsqueeze(1).expand(T, TOP_K).reshape(-1)
    all_expert_ids = topk_idx.reshape(-1)
    all_weights = topk_weights.reshape(-1)

    mask = (all_expert_ids >= local_start) & (all_expert_ids < local_end)
    loc_tokens = all_token_ids[mask]
    loc_experts = (all_expert_ids[mask] - local_start).to(torch.int32)
    loc_weights = all_weights[mask]

    if loc_tokens.numel() == 0:
        output.copy_(output_fp32)
        return

    sort_idx = loc_experts.argsort(stable=True)
    loc_tokens = loc_tokens[sort_idx]
    loc_experts = loc_experts[sort_idx]
    loc_weights = loc_weights[sort_idx]

    counts = torch.zeros(E_LOCAL, dtype=torch.int32, device=device)
    counts.scatter_add_(0, loc_experts.long(), torch.ones_like(loc_experts, dtype=torch.int32))
    counts_cpu = counts.cpu().tolist()
    offset = 0

    for e in range(E_LOCAL):
        cnt = counts_cpu[e]
        if cnt == 0:
            continue

        tok_ids = loc_tokens[offset:offset + cnt]
        tok_wts = loc_weights[offset:offset + cnt]
        offset += cnt

        A = hs_fp32[tok_ids]  # [cnt, 7168] fp32

        # ── GEMM1: A @ W13.T → [cnt, 4096] ──
        W13 = dequant_weight(gemm1_weights[e], gemm1_weights_scale[e], 4096, H, 32, 56)
        gemm1_out = torch.matmul(A, W13.T)

        # ── SwiGLU ──
        W1_out = gemm1_out[:, :I_SIZE]
        W3_out = gemm1_out[:, I_SIZE:]
        intermediate = F.silu(W3_out) * W1_out

        # ── GEMM2: intermediate @ W2.T → [cnt, 7168] ──
        W2 = dequant_weight(gemm2_weights[e], gemm2_weights_scale[e], H, I_SIZE, 56, 16)
        gemm2_out = torch.matmul(intermediate, W2.T)

        # ── Weighted scatter-add ──
        output_fp32.index_add_(0, tok_ids, gemm2_out * tok_wts[:, None])

    # ── 5. Cast to bfloat16 ──
    output.copy_(output_fp32)
