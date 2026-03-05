"""
CUDA MoE Kernel — PyTorch/cuBLAS baseline for Track A.

Uses PyTorch ops (matmul = cuBLAS) for all GEMMs. Serves as:
  1. A correct, runnable reference implementation
  2. A starting point for teammates to replace with fused CUDA kernels from kernel.cu

Architecture:
  1. Routing: sigmoid → group-filter → top-8 → normalized weights (same as Triton)
  2. Per-expert loop:
     a. Dequant hidden_states FP8 → FP32 (block-scale)
     b. Dequant W13 FP8 → FP32 (block-scale)
     c. GEMM1: tokens @ W13.T → [n, 4096] via torch.matmul (cuBLAS)
     d. SwiGLU: silu(W3_out) * W1_out
     e. Dequant W2 FP8 → FP32 (block-scale)
     f. GEMM2: intermediate @ W2.T → [n, 7168] via torch.matmul (cuBLAS)
     g. Weighted scatter-add to output
  3. Cast output FP32 → BF16

Testing this baseline:
  The benchmark's CUDA builder (TVMFFIBuilder) compiles .cu files only — .py files
  are ignored. To test this PyTorch baseline for correctness:
    1. Copy this file to solution/triton/binding.py
    2. Set config.toml: language = "triton", entry_point = "binding.py::kernel"
    3. Pack and run: pack_solution_simple.py → test_modal.py
    4. Restore config.toml afterwards

  For the real CUDA submission, replace the per-expert PyTorch loop with
  launches of the fused kernels from kernel.cu (via ctypes or pybind11).
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
QBLOCK = 128  # FP8 quantization block size


# ═══════════════════════════════════════════════════════════════
# Routing (PyTorch) — DeepSeek-V3 no-aux routing
# ═══════════════════════════════════════════════════════════════

def ds_routing(logits, bias, scale_factor):
    """
    logits: [T, 256] f32,  bias: [256] bf16,  scale_factor: float
    returns: topk_idx [T, 8] int64, topk_weights [T, 8] f32
    """
    T = logits.shape[0]
    s = torch.sigmoid(logits.float())                          # [T, 256]
    b = bias.float().reshape(-1)
    sb = s + b

    gs = E_GLOBAL // N_GROUP  # 32
    sb_g = sb.view(T, N_GROUP, gs)
    top2, _ = torch.topk(sb_g, k=2, dim=2, largest=True, sorted=False)
    g_scores = top2.sum(dim=2)                                 # [T, 8]

    _, g_idx = torch.topk(g_scores, k=TOPK_GROUP, dim=1,
                          largest=True, sorted=False)
    g_mask = torch.zeros(T, N_GROUP, dtype=torch.bool, device=logits.device)
    g_mask.scatter_(1, g_idx, True)
    s_mask = g_mask.unsqueeze(2).expand(T, N_GROUP, gs).reshape(T, E_GLOBAL)

    neg_inf = torch.finfo(torch.float32).min
    sb.masked_fill_(~s_mask, neg_inf)
    _, topk_idx = torch.topk(sb, k=TOP_K, dim=1,
                             largest=True, sorted=False)       # [T, 8]

    # Gather first, normalize on [T, 8] instead of [T, 256]
    topk_s = torch.gather(s, 1, topk_idx)                      # [T, 8]
    topk_weights = topk_s / (topk_s.sum(dim=1, keepdim=True) + 1e-20) * scale_factor

    return topk_idx, topk_weights


# ═══════════════════════════════════════════════════════════════
# FP8 Block-Scale Dequantization Helpers
# ═══════════════════════════════════════════════════════════════

def dequant_hidden_states(hs_fp8, hs_scale):
    """
    hs_fp8:   [T, 7168] float8_e4m3fn
    hs_scale: [56, T] float32  (quantization block = 128, 7168/128 = 56)
    returns:  [T, 7168] float32
    """
    T = hs_fp8.shape[0]
    # [T, 56, 128] * [T, 56, 1] → [T, 7168]
    return (hs_fp8.float().view(T, 56, 128) * hs_scale.T.unsqueeze(2)).reshape(T, H)


def dequant_gemm1_weight(w_fp8, w_scale):
    """
    w_fp8:   [4096, 7168] float8_e4m3fn  (one expert)
    w_scale: [32, 56] float32  (4096/128=32, 7168/128=56)
    returns: [4096, 7168] float32
    """
    # [32, 128, 56, 128] * [32, 1, 56, 1] → [4096, 7168]
    return (w_fp8.float().view(32, 128, 56, 128) * w_scale[:, None, :, None]).reshape(4096, H)


def dequant_gemm2_weight(w_fp8, w_scale):
    """
    w_fp8:   [7168, 2048] float8_e4m3fn  (one expert)
    w_scale: [56, 16] float32  (7168/128=56, 2048/128=16)
    returns: [7168, 2048] float32
    """
    # [56, 128, 16, 128] * [56, 1, 16, 1] → [7168, 2048]
    return (w_fp8.float().view(56, 128, 16, 128) * w_scale[:, None, :, None]).reshape(H, I_SIZE)


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
    output,                   # [T, 7168]         bfloat16  (destination-passing)
):
    T = routing_logits.shape[0]
    device = hidden_states.device
    local_start = int(local_expert_offset)

    # ── 1. Routing ──
    topk_idx, topk_weights = ds_routing(
        routing_logits, routing_bias, float(routed_scaling_factor)
    )

    # Accumulate in FP32 to avoid bf16 rounding across experts
    output_fp32 = torch.zeros((T, H), dtype=torch.float32, device=device)

    # ── 2. Dequant hidden_states once (shared across all experts) ──
    hs_fp32 = dequant_hidden_states(hidden_states, hidden_states_scale)  # [T, 7168]

    # ── 3. Per-expert loop ──
    # Build per-expert token lists: for each local expert, find which (token, slot)
    # pairs are routed to it. No BLOCK_M padding needed for matmul path.
    local_end = local_start + E_LOCAL

    # Flatten [T, TOP_K] → [T*TOP_K]
    all_token_ids = torch.arange(T, device=device).unsqueeze(1).expand(T, TOP_K).reshape(-1)
    all_expert_ids = topk_idx.reshape(-1)
    all_weights = topk_weights.reshape(-1)

    # Filter to local experts
    mask = (all_expert_ids >= local_start) & (all_expert_ids < local_end)
    loc_tokens = all_token_ids[mask]       # token indices
    loc_experts = (all_expert_ids[mask] - local_start).to(torch.int32)  # local expert 0..31
    loc_weights = all_weights[mask]

    if loc_tokens.numel() == 0:
        output.copy_(output_fp32)
        return

    # Sort by expert
    sort_idx = loc_experts.argsort(stable=True)
    loc_tokens = loc_tokens[sort_idx]
    loc_experts = loc_experts[sort_idx]
    loc_weights = loc_weights[sort_idx]

    # Count tokens per expert
    counts = torch.zeros(E_LOCAL, dtype=torch.int32, device=device)
    counts.scatter_add_(0, loc_experts.long(),
                        torch.ones_like(loc_experts, dtype=torch.int32))

    # Use CPU path for the expert loop (simple, avoids many small kernel launches)
    counts_cpu = counts.cpu().tolist()
    offset = 0

    for e in range(E_LOCAL):
        cnt = counts_cpu[e]
        if cnt == 0:
            continue

        # Token indices for this expert
        tok_ids = loc_tokens[offset:offset + cnt]   # [cnt] int64
        tok_wts = loc_weights[offset:offset + cnt]  # [cnt] f32
        offset += cnt

        # Gather hidden states for these tokens: [cnt, 7168] fp32
        A = hs_fp32[tok_ids]

        # ── GEMM1: A @ W13.T → [cnt, 4096] ──
        W13 = dequant_gemm1_weight(gemm1_weights[e], gemm1_weights_scale[e])  # [4096, 7168]
        gemm1_out = torch.matmul(A, W13.T)  # [cnt, 4096]

        # ── SwiGLU: silu(W3_out) * W1_out ──
        W1_out = gemm1_out[:, :I_SIZE]      # [cnt, 2048]
        W3_out = gemm1_out[:, I_SIZE:]      # [cnt, 2048]
        intermediate = F.silu(W3_out) * W1_out  # [cnt, 2048]

        # ── GEMM2: intermediate @ W2.T → [cnt, 7168] ──
        W2 = dequant_gemm2_weight(gemm2_weights[e], gemm2_weights_scale[e])  # [7168, 2048]
        gemm2_out = torch.matmul(intermediate, W2.T)  # [cnt, 7168]

        # ── Weighted scatter-add ──
        output_fp32.index_add_(0, tok_ids, gemm2_out * tok_wts[:, None])

    # ── 4. Cast to bfloat16 ──
    output.copy_(output_fp32)
