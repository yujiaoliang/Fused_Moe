"""
Fused MoE Kernel — Track A (FlashInfer AI Kernel Generation Contest)

FP8 block-scale MoE with DeepSeek-V3 no-aux routing.
Target hardware: NVIDIA B200 (Blackwell).

Architecture:
  1. Routing:  sigmoid → group-filter → top-8 → normalized weights
  2. Per-expert: FP8-dequant → GEMM1(W13) → SwiGLU → GEMM2(W2)
  3. Accumulate weighted expert outputs → bfloat16
"""

import torch
import triton
import triton.language as tl

# ──────────────────────────────────────────────────────────────
# Constants (DeepSeek-V3 / R1)
# ──────────────────────────────────────────────────────────────
H = 7168
I = 2048
E_GLOBAL = 256
E_LOCAL = 32
TOP_K = 8
N_GROUP = 8
TOPK_GROUP = 4
BLOCK = 128  # FP8 quantization block size


# ──────────────────────────────────────────────────────────────
# Triton kernel: fused SwiGLU
# ──────────────────────────────────────────────────────────────
@triton.jit
def _swiglu_fwd(
    G1_ptr,       # [N, 2*I_SIZE]  input (GEMM1 output)
    OUT_ptr,      # [N, I_SIZE]    output
    N,
    I_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_i = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ri = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = (rn[:, None] < N) & (ri[None, :] < I_SIZE)

    stride = 2 * I_SIZE
    # gate = G1[:, :I],  up = G1[:, I:]
    gate = tl.load(G1_ptr + rn[:, None] * stride + ri[None, :],
                   mask=mask, other=0.0).to(tl.float32)
    up   = tl.load(G1_ptr + rn[:, None] * stride + (ri[None, :] + I_SIZE),
                   mask=mask, other=0.0).to(tl.float32)

    silu_up = up * tl.sigmoid(up)
    result = silu_up * gate

    tl.store(OUT_ptr + rn[:, None] * I_SIZE + ri[None, :],
             result.to(tl.bfloat16), mask=mask)


def swiglu(g1: torch.Tensor) -> torch.Tensor:
    """g1: [N, 2*I] → [N, I]  (gate, up) → silu(up)*gate"""
    N = g1.shape[0]
    out = torch.empty((N, I), dtype=torch.bfloat16, device=g1.device)
    if N == 0:
        return out
    BN, BI = 32, min(I, 1024)
    grid = (triton.cdiv(N, BN), triton.cdiv(I, BI))
    _swiglu_fwd[grid](g1, out, N, I, BLOCK_N=BN, BLOCK_I=BI)
    return out


# ──────────────────────────────────────────────────────────────
# FP8 block-scale dequantization helpers
# ──────────────────────────────────────────────────────────────
def dequant_hidden(x_fp8: torch.Tensor, scale: torch.Tensor, T: int):
    """
    x_fp8:  [T, H]         float8_e4m3fn
    scale:  [H/128, T]     float32   ← NOTE transposed layout
    return: [T, H]          bfloat16
    """
    x = x_fp8.view(T, H // BLOCK, BLOCK).float()          # [T, 56, 128]
    s = scale.permute(1, 0).contiguous().unsqueeze(-1)     # [T, 56, 1]
    return (x * s).reshape(T, H).to(torch.bfloat16)


def dequant_weight(w_fp8: torch.Tensor, scale: torch.Tensor):
    """
    w_fp8:  [out, inp]                  float8_e4m3fn
    scale:  [out/128, inp/128]          float32
    return: [out, inp]                  bfloat16
    """
    out_dim, in_dim = w_fp8.shape
    ob, ib = out_dim // BLOCK, in_dim // BLOCK
    w = w_fp8.view(ob, BLOCK, ib, BLOCK).float()           # [ob, 128, ib, 128]
    s = scale.float().view(ob, 1, ib, 1)                   # [ob,   1, ib,   1]
    return (w * s).reshape(out_dim, in_dim).to(torch.bfloat16)


# ──────────────────────────────────────────────────────────────
# DeepSeek-V3 no-aux routing
# ──────────────────────────────────────────────────────────────
def ds_routing(logits, bias, scale_factor):
    """
    logits: [T, 256] f32,  bias: [256] bf16,  scale_factor: float
    returns: topk_idx [T,8], weights [T, 256]
    """
    T = logits.shape[0]
    s = torch.sigmoid(logits.float())                      # [T, 256]
    b = bias.float().reshape(-1)
    sb = s + b                                             # with bias

    # group filtering: 8 groups × 32 experts
    gs = 32  # group_size = E_GLOBAL // N_GROUP
    sb_g = sb.view(T, N_GROUP, gs)
    top2, _ = torch.topk(sb_g, k=2, dim=2, largest=True, sorted=False)
    g_scores = top2.sum(dim=2)                             # [T, 8]

    _, g_idx = torch.topk(g_scores, k=TOPK_GROUP, dim=1,
                          largest=True, sorted=False)       # [T, 4]
    g_mask = torch.zeros_like(g_scores)
    g_mask.scatter_(1, g_idx, 1.0)
    s_mask = g_mask.unsqueeze(2).expand(T, N_GROUP, gs).reshape(T, E_GLOBAL)

    # global top-k in kept groups
    neg_inf = torch.finfo(torch.float32).min
    pruned = sb.masked_fill(s_mask == 0, neg_inf)
    _, topk_idx = torch.topk(pruned, k=TOP_K, dim=1,
                             largest=True, sorted=False)    # [T, 8]

    # combination weights (from s without bias)
    m = torch.zeros_like(s)
    m.scatter_(1, topk_idx, 1.0)
    w = s * m
    w = (w / (w.sum(dim=1, keepdim=True) + 1e-20)) * scale_factor
    return topk_idx, w


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────
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
):
    T = routing_logits.shape[0]
    device = hidden_states.device
    local_start = int(local_expert_offset)

    # 1 ── routing
    topk_idx, weights = ds_routing(
        routing_logits, routing_bias, float(routed_scaling_factor)
    )

    # 2 ── dequantize hidden states  →  [T, H] bf16
    A = dequant_hidden(hidden_states, hidden_states_scale, T)

    # 3 ── per-expert compute
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    for le in range(E_LOCAL):
        ge = local_start + le
        if ge < 0 or ge >= E_GLOBAL:
            continue

        # tokens that selected this expert
        sel = (topk_idx == ge).any(dim=1)
        if not sel.any():
            continue
        tok = torch.nonzero(sel, as_tuple=False).squeeze(1)

        A_e = A[tok]                                       # [Tk, H] bf16

        # dequant weights for this expert
        W13 = dequant_weight(gemm1_weights[le],
                             gemm1_weights_scale[le])      # [2I, H] bf16
        W2  = dequant_weight(gemm2_weights[le],
                             gemm2_weights_scale[le])      # [H, I]  bf16

        # GEMM1: [Tk, H] @ [H, 2I] → [Tk, 2I]
        G1 = torch.matmul(A_e, W13.t())                   # bf16 Tensor Core

        # SwiGLU (Triton-fused): silu(up) * gate → [Tk, I]
        C = swiglu(G1)

        # GEMM2: [Tk, I] @ [I, H] → [Tk, H]
        O = torch.matmul(C, W2.t())                       # bf16 Tensor Core

        # weighted accumulation
        w_tok = weights[tok, ge]                           # [Tk]
        output.index_add_(0, tok, O.float() * w_tok.unsqueeze(1))

    return output.to(torch.bfloat16)
