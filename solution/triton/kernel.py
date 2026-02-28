"""
Fused MoE Kernel — Track A (FlashInfer AI Kernel Generation Contest)

FP8 block-scale MoE with DeepSeek-V3 no-aux routing.
Target hardware: NVIDIA B200 (Blackwell).

Architecture:
  1. Routing (PyTorch):  sigmoid → group-filter → top-8 → normalized weights
  2. Token sorting (PyTorch): group tokens by expert, pad to BLOCK_SIZE_M
  3. GEMM1 (Triton): FP8×FP8 grouped GEMM with block-scale dequant
  4. SwiGLU (Triton): silu(up) * gate
  5. GEMM2 (Triton): BF16×FP8 grouped GEMM with block-scale dequant
  6. Weighted reduce (PyTorch): scatter-add expert outputs
"""

import torch
import triton
import triton.language as tl

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
    g_mask = torch.zeros_like(g_scores)
    g_mask.scatter_(1, g_idx, 1.0)
    s_mask = g_mask.unsqueeze(2).expand(T, N_GROUP, gs).reshape(T, E_GLOBAL)

    neg_inf = torch.finfo(torch.float32).min
    pruned = sb.masked_fill(s_mask == 0, neg_inf)
    _, topk_idx = torch.topk(pruned, k=TOP_K, dim=1,
                             largest=True, sorted=False)       # [T, 8]

    # Weights from s (without bias), normalized
    m = torch.zeros_like(s)
    m.scatter_(1, topk_idx, 1.0)
    w = s * m
    w = (w / (w.sum(dim=1, keepdim=True) + 1e-20)) * scale_factor
    topk_weights = torch.gather(w, 1, topk_idx)               # [T, 8]

    return topk_idx, topk_weights


# ═══════════════════════════════════════════════════════════════
# Token Sorting (PyTorch)
# ═══════════════════════════════════════════════════════════════

def moe_sort_tokens(topk_idx, topk_weights, local_start, BLOCK_M, T, device):
    """
    Sort tokens by expert, filter to local experts, pad to BLOCK_M.

    Returns:
        sorted_token_ids [num_padded] int64 — token index for gather
        block_expert_ids [num_blocks] int32 — expert for each M-block
        sorted_weights   [num_padded] f32   — routing weight per slot
        num_padded       int                — total padded length
    Returns (None, ...) if no local experts selected.
    """
    # Flatten [T, TOP_K] → [T*TOP_K]
    all_token_ids = torch.arange(T, device=device).unsqueeze(1).expand(T, TOP_K).reshape(-1)
    all_expert_ids = topk_idx.reshape(-1)
    all_weights = topk_weights.reshape(-1)

    # Filter to local experts
    local_end = local_start + E_LOCAL
    mask = (all_expert_ids >= local_start) & (all_expert_ids < local_end)
    loc_tokens = all_token_ids[mask]
    loc_experts = (all_expert_ids[mask] - local_start).to(torch.int32)
    loc_weights = all_weights[mask]

    if loc_tokens.numel() == 0:
        return None, None, None, 0

    # Sort by expert
    sort_idx = loc_experts.argsort(stable=True)
    loc_tokens = loc_tokens[sort_idx]
    loc_experts = loc_experts[sort_idx]
    loc_weights = loc_weights[sort_idx]

    # Count per expert
    counts = torch.zeros(E_LOCAL, dtype=torch.int32, device=device)
    counts.scatter_add_(0, loc_experts.long(), torch.ones_like(loc_experts, dtype=torch.int32))

    # Pad each expert's tokens to BLOCK_M alignment
    padded_tokens_list = []
    padded_weights_list = []
    block_experts_list = []
    offset = 0

    for e in range(E_LOCAL):
        cnt = counts[e].item()
        if cnt == 0:
            continue
        num_blocks = (cnt + BLOCK_M - 1) // BLOCK_M
        pad_cnt = num_blocks * BLOCK_M - cnt

        padded_tokens_list.append(loc_tokens[offset:offset + cnt])
        padded_weights_list.append(loc_weights[offset:offset + cnt])

        if pad_cnt > 0:
            # Pad with T (out-of-range, will be masked)
            padded_tokens_list.append(torch.full((pad_cnt,), T, device=device, dtype=torch.int64))
            padded_weights_list.append(torch.zeros(pad_cnt, device=device))

        block_experts_list.extend([e] * num_blocks)
        offset += cnt

    sorted_token_ids = torch.cat(padded_tokens_list)
    sorted_weights = torch.cat(padded_weights_list)
    block_expert_ids = torch.tensor(block_experts_list, device=device, dtype=torch.int32)
    num_padded = sorted_token_ids.shape[0]

    return sorted_token_ids, block_expert_ids, sorted_weights, num_padded


# ═══════════════════════════════════════════════════════════════
# Triton Kernel: Grouped GEMM (FP8 × FP8) with block-scale dequant
# For GEMM1: hidden_states[T,H]fp8 × W13[E,4096,7168]fp8 → [EM,4096]
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _grouped_gemm_fp8_kernel(
    # Data pointers
    A_ptr,            # [T, K] fp8 (hidden_states)
    B_ptr,            # [E, N, K] fp8 (weights)
    C_ptr,            # [EM, N] output buffer
    A_scale_ptr,      # [K//128, T] fp32 (transposed layout)
    B_scale_ptr,      # [E, N//128, K//128] fp32
    token_ids_ptr,    # [EM] sorted token indices
    expert_ids_ptr,   # [num_M_blocks] expert per block
    # Dimensions
    T,                # NOT constexpr — varies per workload
    K: tl.constexpr,
    N: tl.constexpr,
    # Strides for A [T, K]
    stride_at, stride_ak,
    # Strides for B [E, N, K]
    stride_be, stride_bn, stride_bk,
    # Strides for C [EM, N]
    stride_cm, stride_cn,
    # Strides for A_scale [K//128, T]
    stride_as0, stride_as1,
    # Strides for B_scale [E, N//128, K//128]
    stride_bs0, stride_bs1, stride_bs2,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n: tl.constexpr = N // BLOCK_N
    num_pid_m = tl.num_programs(0) // num_pid_n

    # Grouped ordering for L2 reuse
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Load token indices for this M-block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    token_ids = tl.load(token_ids_ptr + offs_m)
    token_mask = token_ids < T

    # Get expert for this block
    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    # Offsets
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # A pointers: gather rows by token_ids. A[token_id, k]
    a_ptrs = A_ptr + token_ids[:, None].to(tl.int64) * stride_at + offs_k[None, :] * stride_ak

    # B pointers: B[expert].T accessed as B[expert, n, k] -> [K, N]
    b_ptrs = (B_ptr + expert_id * stride_be
              + offs_k[:, None] * stride_bk
              + offs_n[None, :].to(tl.int64) * stride_bn)

    # A_scale base: A_scale[k_block, token_id]
    a_scale_base = A_scale_ptr + token_ids.to(tl.int64) * stride_as1

    # B_scale base: B_scale[expert, n_block, k_block]
    n_block_idx = pid_n  # BLOCK_N == QBLOCK assumed
    b_scale_base = (B_scale_ptr + expert_id * stride_bs0
                    + n_block_idx * stride_bs1)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_iters: tl.constexpr = K // BLOCK_K
    for k in range(num_k_iters):
        # Load FP8 data
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)

        # Load scales
        a_scale = tl.load(a_scale_base + k * stride_as0,
                          mask=token_mask, other=0.0)           # [BLOCK_M]
        b_scale = tl.load(b_scale_base + k * stride_bs2)       # scalar

        # FP8 dot + block-scale dequant
        acc += tl.dot(a, b) * a_scale[:, None] * b_scale

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Store result as bf16
    c_ptrs = (C_ptr + offs_m[:, None].to(tl.int64) * stride_cm
              + offs_n[None, :].to(tl.int64) * stride_cn)
    c_mask = token_mask[:, None] & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


# ═══════════════════════════════════════════════════════════════
# Triton Kernel: SwiGLU
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _swiglu_kernel(
    IN_ptr,      # [M, 2*I_SIZE] input (GEMM1 output)
    OUT_ptr,     # [M, I_SIZE]   output
    M,
    I_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = (offs_m[:, None] < M) & (offs_i[None, :] < I_SIZE)

    stride_in = 2 * I_SIZE
    # gate = IN[:, :I_SIZE],  up = IN[:, I_SIZE:]
    gate = tl.load(IN_ptr + offs_m[:, None] * stride_in + offs_i[None, :],
                   mask=mask, other=0.0).to(tl.float32)
    up = tl.load(IN_ptr + offs_m[:, None] * stride_in + offs_i[None, :] + I_SIZE,
                 mask=mask, other=0.0).to(tl.float32)

    silu_up = up * tl.sigmoid(up)
    result = silu_up * gate

    tl.store(OUT_ptr + offs_m[:, None] * I_SIZE + offs_i[None, :],
             result.to(tl.bfloat16), mask=mask)


# ═══════════════════════════════════════════════════════════════
# Triton Kernel: Grouped GEMM (BF16 × FP8) with weight block-scale
# For GEMM2: swiglu_out[EM,2048]bf16 × W2[E,7168,2048]fp8 → [EM,7168]
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _grouped_gemm_bf16xfp8_kernel(
    A_ptr,            # [EM, K] bf16 (SwiGLU output, contiguous)
    B_ptr,            # [E, N, K] fp8 (weights)
    C_ptr,            # [EM, N] output buffer
    B_scale_ptr,      # [E, N//128, K//128] fp32
    expert_ids_ptr,   # [num_M_blocks] expert per block
    weights_ptr,      # [EM] routing weights (f32)
    # Dimensions
    EM,
    K: tl.constexpr,
    N: tl.constexpr,
    # Strides for A [EM, K]
    stride_am, stride_ak,
    # Strides for B [E, N, K]
    stride_be, stride_bn, stride_bk,
    # Strides for C [EM, N]
    stride_cm, stride_cn,
    # Strides for B_scale [E, N//128, K//128]
    stride_bs0, stride_bs1, stride_bs2,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_WEIGHT: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_m = tl.num_programs(0) // num_pid_n

    # Grouped ordering for L2 reuse
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Row offsets (contiguous — no gather needed)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = offs_m < EM

    # Expert for this block
    expert_id = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_k = tl.arange(0, BLOCK_K)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # A pointers: contiguous rows
    a_ptrs = A_ptr + offs_m[:, None].to(tl.int64) * stride_am + offs_k[None, :] * stride_ak

    # B pointers: B[expert].T → B[expert, n, k] read as [K, N]
    b_ptrs = (B_ptr + expert_id * stride_be
              + offs_k[:, None] * stride_bk
              + offs_n[None, :].to(tl.int64) * stride_bn)

    # B_scale base
    n_block_idx = pid_n  # BLOCK_N == QBLOCK
    b_scale_base = B_scale_ptr + expert_id * stride_bs0 + n_block_idx * stride_bs1

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    num_k_blocks: tl.constexpr = K // BLOCK_K
    for k in range(num_k_blocks):
        a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)

        b_scale = tl.load(b_scale_base + k * stride_bs2)

        # Dequant FP8 weight to bf16 (fp8 -> fp32 * scale -> bf16 for dot)
        b_dequant = (b.to(tl.float32) * b_scale).to(tl.bfloat16)
        acc += tl.dot(a, b_dequant)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Multiply by routing weight
    if MUL_WEIGHT:
        w = tl.load(weights_ptr + offs_m, mask=row_mask, other=0.0)
        acc *= w[:, None]

    c_ptrs = (C_ptr + offs_m[:, None].to(tl.int64) * stride_cm
              + offs_n[None, :].to(tl.int64) * stride_cn)
    c_mask = row_mask[:, None] & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


# ═══════════════════════════════════════════════════════════════
# Entry Point
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
):
    T = routing_logits.shape[0]
    device = hidden_states.device
    local_start = int(local_expert_offset)

    # ── 1. Routing ──
    topk_idx, topk_weights = ds_routing(
        routing_logits, routing_bias, float(routed_scaling_factor)
    )

    # ── 2. Token sorting ──
    BLOCK_M = 64
    sorted_token_ids, block_expert_ids, sorted_weights, num_padded = \
        moe_sort_tokens(topk_idx, topk_weights, local_start, BLOCK_M, T, device)

    if sorted_token_ids is None:
        return torch.zeros((T, H), dtype=torch.bfloat16, device=device)

    num_blocks = num_padded // BLOCK_M

    # ── 3. GEMM1: [T, H=7168]fp8 × [E, 4096, 7168]fp8 → [EM, 4096]bf16 ──
    GEMM1_N = 2 * I_SIZE  # 4096
    gemm1_out = torch.empty((num_padded, GEMM1_N), dtype=torch.bfloat16, device=device)

    BLOCK_N = QBLOCK  # 128
    BLOCK_K = QBLOCK  # 128
    grid_gemm1 = (num_blocks * (GEMM1_N // BLOCK_N),)

    _grouped_gemm_fp8_kernel[grid_gemm1](
        hidden_states, gemm1_weights, gemm1_out,
        hidden_states_scale, gemm1_weights_scale,
        sorted_token_ids, block_expert_ids,
        T=T, K=H, N=GEMM1_N,
        stride_at=hidden_states.stride(0), stride_ak=hidden_states.stride(1),
        stride_be=gemm1_weights.stride(0), stride_bn=gemm1_weights.stride(1),
        stride_bk=gemm1_weights.stride(2),
        stride_cm=gemm1_out.stride(0), stride_cn=gemm1_out.stride(1),
        stride_as0=hidden_states_scale.stride(0), stride_as1=hidden_states_scale.stride(1),
        stride_bs0=gemm1_weights_scale.stride(0), stride_bs1=gemm1_weights_scale.stride(1),
        stride_bs2=gemm1_weights_scale.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=8,
    )

    # ── 4. SwiGLU: [EM, 4096] → [EM, 2048] ──
    swiglu_out = torch.empty((num_padded, I_SIZE), dtype=torch.bfloat16, device=device)
    SW_BLOCK_M = 32
    SW_BLOCK_I = min(I_SIZE, 1024)
    swiglu_grid = (triton.cdiv(num_padded, SW_BLOCK_M), triton.cdiv(I_SIZE, SW_BLOCK_I))
    _swiglu_kernel[swiglu_grid](
        gemm1_out, swiglu_out, num_padded,
        I_SIZE=I_SIZE,
        BLOCK_M=SW_BLOCK_M, BLOCK_I=SW_BLOCK_I,
    )

    # ── 5. GEMM2: [EM, 2048]bf16 × [E, 7168, 2048]fp8 → [EM, 7168]bf16 ──
    gemm2_out = torch.empty((num_padded, H), dtype=torch.bfloat16, device=device)
    GEMM2_K = I_SIZE   # 2048
    GEMM2_N = H        # 7168

    grid_gemm2 = (num_blocks * (GEMM2_N // BLOCK_N),)

    _grouped_gemm_bf16xfp8_kernel[grid_gemm2](
        swiglu_out, gemm2_weights, gemm2_out,
        gemm2_weights_scale, block_expert_ids, sorted_weights,
        EM=num_padded, K=GEMM2_K, N=GEMM2_N,
        stride_am=swiglu_out.stride(0), stride_ak=swiglu_out.stride(1),
        stride_be=gemm2_weights.stride(0), stride_bn=gemm2_weights.stride(1),
        stride_bk=gemm2_weights.stride(2),
        stride_cm=gemm2_out.stride(0), stride_cn=gemm2_out.stride(1),
        stride_bs0=gemm2_weights_scale.stride(0), stride_bs1=gemm2_weights_scale.stride(1),
        stride_bs2=gemm2_weights_scale.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=8,
        MUL_WEIGHT=True,
    )

    # ── 6. Weighted reduce: scatter-add back to original tokens ──
    output = torch.zeros((T, H), dtype=torch.float32, device=device)

    # Only accumulate valid (non-padded) entries
    valid_mask = sorted_token_ids < T
    valid_token_ids = sorted_token_ids[valid_mask]
    valid_out = gemm2_out[valid_mask].float()
    # Weight already applied in GEMM2 kernel

    output.index_add_(0, valid_token_ids, valid_out)

    return output.to(torch.bfloat16)
