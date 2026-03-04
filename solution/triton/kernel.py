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

# ── Triton tuning parameters ──
BLOCK_M = 64   # tokens per M-block (for token sorting & GEMM)
BLOCK_K = 128  # K-block (must equal QBLOCK for scale alignment)
GROUP_SIZE_M = 8  # L2 cache reuse grouping


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

    # Vectorized padding: count per expert and compute padded offsets
    counts = torch.zeros(E_LOCAL, dtype=torch.int32, device=device)
    counts.scatter_add_(0, loc_experts.long(), torch.ones_like(loc_experts, dtype=torch.int32))

    # Compute padded lengths and offsets per expert (on CPU for indexing)
    counts_cpu = counts.cpu().tolist()
    padded_tokens_list = []
    padded_weights_list = []
    block_experts_list = []
    offset = 0

    for e in range(E_LOCAL):
        cnt = counts_cpu[e]
        if cnt == 0:
            continue
        num_blocks = (cnt + BLOCK_M - 1) // BLOCK_M
        pad_cnt = num_blocks * BLOCK_M - cnt

        padded_tokens_list.append(loc_tokens[offset:offset + cnt])
        padded_weights_list.append(loc_weights[offset:offset + cnt])

        if pad_cnt > 0:
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
# Triton Kernel 1: Fused GEMM1 + SwiGLU
# Computes: SwiGLU( A[sorted_idx] @ W13 )
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _fused_moe_gemm1_swiglu_kernel(
    # Data pointers
    A_ptr,            # [T, H] fp8 (hidden_states)
    A_scale_ptr,      # [H//128, T] fp32 (hidden_states_scale)
    B_ptr,            # [E, N, H] fp8 (W13 weights, transposed physically or logically)
    C_ptr,            # [num_padded, N//2] bf16 (intermediate SwiGLU output)
    B_scale_ptr,      # [E, N//128, H//128] fp32 (weight scales)
    token_ids_ptr,    # [num_padded] int64 (sorted token indices)
    expert_ids_ptr,   # [num_blocks] int32 (expert for each M-block)
    
    # Dimensions
    num_padded, T, H: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    
    # Strides
    stride_at, stride_ah,
    stride_as0, stride_as1,
    stride_be, stride_bn, stride_bh,
    stride_cm, stride_cn,
    stride_bse, stride_bsn, stride_bsh,
    
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    """
    Computes GEMM1 and SwiGLU.
    A: [T, H] (fp8)
    W13: [E, 4096, 7168] (fp8) -> loaded as [E, N, H] logically
    C: [num_padded, 2048] (bf16) after SwiGLU
    """
    pid = tl.program_id(0)
    # We want to output N_OUT = N // 2 channels (2048)
    N_OUT = N // 2
    num_pid_m = tl.cdiv(num_padded, BLOCK_M)
    num_pid_n = tl.cdiv(N_OUT, BLOCK_N)
    
    # Grouped Swizzle
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Load expert ID for this M block
    expert_id = tl.load(expert_ids_ptr + pid_m)
    
    # Offsets for M (tokens) and N (output channels)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # SwiGLU needs both rn (W1) and rn + N_OUT (W3)
    rn_1 = rn
    rn_3 = rn + N_OUT
    
    # Load token indices
    token_idx = tl.load(token_ids_ptr + rm, mask=rm < num_padded, other=T)
    # Mask out padding tokens (T is the pad index)
    m_mask = token_idx < T

    # Accumulators for W1 and W3
    acc_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K loop
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        
        safe_token_idx = tl.where(token_idx < T, token_idx, 0)
        
        # Load A: fp8
        a_ptrs = A_ptr + (safe_token_idx[:, None] * stride_at + rk[None, :] * stride_ah)
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        
        # Load A scale: fp32
        # A_scale is [K//128, T] -> transposed logically
        a_scale_ptrs = A_scale_ptr + (k // 128) * stride_as0 + safe_token_idx * stride_as1
        a_scale = tl.load(a_scale_ptrs, mask=m_mask, other=0.0) # shape: [BLOCK_M]
        
        # Load B for W1: fp8
        b_ptrs_1 = B_ptr + expert_id * stride_be + rn_1[None, :] * stride_bn + rk[:, None] * stride_bh
        b_1 = tl.load(b_ptrs_1)
        
        # Load B scale for W1 (fp32)
        # B_scale is [E, N//128, K//128]
        # We broadcast it over the K dimension (which is size 1 since BLOCK_K=128)
        # Let's load it as [1, BLOCK_N] so it broadcasts over BLOCK_M
        b_scale_ptrs_1 = B_scale_ptr + expert_id * stride_bse + (rn_1[None, :] // 128) * stride_bsn + (k // 128) * stride_bsh
        b_scale_1 = tl.load(b_scale_ptrs_1) # shape: [1, BLOCK_N]
        
        # Load B for W3: fp8
        b_ptrs_3 = B_ptr + expert_id * stride_be + rn_3[None, :] * stride_bn + rk[:, None] * stride_bh
        b_3 = tl.load(b_ptrs_3)
        
        # Load B scale for W3
        b_scale_ptrs_3 = B_scale_ptr + expert_id * stride_bse + (rn_3[None, :] // 128) * stride_bsn + (k // 128) * stride_bsh
        b_scale_3 = tl.load(b_scale_ptrs_3) # shape: [1, BLOCK_N]
        
        # Cast A to fp32 and scale directly
        a_fp32 = a.to(tl.float32) * a_scale[:, None]
        
        # Scale B natively in fp32
        b_1_fp32 = b_1.to(tl.float32) * b_scale_1
        b_3_fp32 = b_3.to(tl.float32) * b_scale_3
        
        # Dot natively via fp32 tensor cores (TF32) for precise verification matching
        dot_out_1 = tl.dot(a_fp32, b_1_fp32, acc=None, out_dtype=tl.float32)
        acc_1 += dot_out_1 # (BLOCK_M, BLOCK_N)
        
        dot_out_3 = tl.dot(a_fp32, b_3_fp32, acc=None, out_dtype=tl.float32)
        acc_3 += dot_out_3

    # Epilogue: SwiGLU (Note: PyTorch baseline does silu(W3) * W1)
    sig_out = 1.0 / (1.0 + tl.exp(-acc_3))
    swiglu_out = (acc_3 * sig_out) * acc_1
    
    # Store to C (store directly as fp32 to match eager precision exactly without bf16 mantissa clip)
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, swiglu_out, mask=m_mask[:, None])


# ═══════════════════════════════════════════════════════════════
# Triton Kernel 2: Fused GEMM2 + Routing Weight + Scatter Add
# Computes: Output[token_id] += (Intermediate[sorted_idx] @ W2) * routing_weight
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _fused_moe_gemm2_scatter_kernel(
    # Data pointers
    A_ptr,            # [num_padded, K] bf16 (intermediate SwiGLU output)
    B_ptr,            # [E, N, K] fp8 (W2 weights trans)
    C_ptr,            # [T, N] bf16 (final output tensor, modified in place)
    B_scale_ptr,      # [E, N//16, K//128] fp32 (W2 block scales)
    token_weights_ptr,# [num_padded] fp32 (routing weights for each slot)
    token_ids_ptr,    # [num_padded] int64 (original token indices to scatter to)
    expert_ids_ptr,   # [num_blocks] int32 (expert for each M-block)
    
    # Dimensions
    T, num_padded: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    
    # Strides
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_ct, stride_cn,
    stride_bse, stride_bsn, stride_bsk,
    
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(num_padded, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    # Grouped Swizzle
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Load expert ID
    expert_id = tl.load(expert_ids_ptr + pid_m)
    
    # Offsets
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Masks
    m_mask = rm < num_padded
    n_mask = rn < N
    
    # Load routing token IDs and Weights
    token_idx = tl.load(token_ids_ptr + rm, mask=m_mask, other=T)
    # T is the pad token. So actual valid tokens have index < T
    valid_mask = token_idx < T
    
    token_weights = tl.load(token_weights_ptr + rm, mask=m_mask, other=0.0)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        k_mask = rk < K
        
        # Load A: bf16 -> cast to fp32 or keep bf16
        # A is [num_padded, K]
        safe_rm = tl.where(rm < num_padded, rm, 0)
        a_ptrs = A_ptr + safe_rm[:, None] * stride_am + rk[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        
        # Load B: fp8
        # B is [E, N, K] -> loaded as transposed W2
        b_ptrs = B_ptr + expert_id * stride_be + rn[None, :] * stride_bn + rk[:, None] * stride_bk
        b = tl.load(b_ptrs, mask=n_mask[None, :] & k_mask[:, None], other=0.0)
        
        # Load B scale: fp32
        # b_scale is [E, K//128, N//128] (which is [32, 56, 16] for K=7168, N=2048)
        # b_scale_ptrs receives transposed mapping accurately
        b_scale_ptrs = B_scale_ptr + expert_id * stride_bse + (k // 128) * stride_bsk + (rn[None, :] // 128) * stride_bsn
        b_scale = tl.load(b_scale_ptrs, mask=n_mask[None, :], other=0.0) # shape: [1, BLOCK_N]
        
        # We need mixed precision dot. tl.dot supports A=fp32, B=fp8 in some Triton versions?
        # PyTorch baseline retains tf32 precision across GEMM2. We do NOT downcast to bf16!
        # Load B natively as fp32 multiplied by scale.
        b_fp32 = b.to(tl.float32) * b_scale
        
        # A is natively fp32 (from Intermediate buffer). Use TF32 dot accumulation to match PyTorch
        dot_out = tl.dot(a, b_fp32, acc=None, out_dtype=tl.float32)
        acc += dot_out
        
    # Scale by routing weights
    out = acc * token_weights[:, None]

    # Scatter Add directly to output tensor C[T, N]
    # Cap token_idx to 0 to prevent out-of-bounds pointer crashes even when masked
    safe_token_idx = tl.where(token_idx < T, token_idx, 0)
    c_ptrs = C_ptr + safe_token_idx[:, None] * stride_ct + rn[None, :] * stride_cn
    
    # Triton atomic_add into FP32 buffer
    tl.atomic_add(c_ptrs, out.to(tl.float32), mask=valid_mask[:, None] & n_mask[None, :], sem='relaxed')

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

        # Dequant fp8 → fp32 * scale → bf16 BEFORE dot
        a_dequant = (a.to(tl.float32) * a_scale[:, None]).to(tl.bfloat16)
        b_dequant = (b.to(tl.float32) * b_scale).to(tl.bfloat16)
        acc += tl.dot(a_dequant, b_dequant)

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
# Per-expert compute: FP8 native matmul on B200
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# Triton Kernel 3: FULLY FUSED GEMM1 + SwiGLU + GEMM2
# (Phase 3 Optimization: Zero Intermediate HBM Reads/Writes)
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _fully_fused_moe_kernel(
    # Data pointers
    A_ptr,            # [T, H] fp8 (hidden_states)
    A_scale_ptr,      # [H//128, T] fp32 (hidden_states_scale)
    B1_ptr,           # [E, N1, H] fp8 (W13 weights trans)
    B1_scale_ptr,     # [E, N1//128, H//128] fp32 (W13 scales)
    B2_ptr,           # [E, H, N1//2] fp8 (W2 weights trans)
    B2_scale_ptr,     # [E, H//16, (N1//2)//128] fp32 (W2 scales)
    C_ptr,            # [T, H] float32 (Output Accumulation Buffer)
    token_weights_ptr,# [num_padded] fp32
    token_ids_ptr,    # [num_padded] int64
    expert_ids_ptr,   # [num_blocks] int32
    
    # Dimensions
    T, num_padded: tl.constexpr, 
    H: tl.constexpr, N1: tl.constexpr, N1_HALF: tl.constexpr,
    
    # Strides
    stride_at, stride_ah,
    stride_as0, stride_as1,
    stride_b1e, stride_b1n, stride_b1h,
    stride_b1se, stride_b1sn, stride_b1sh,
    stride_b2e, stride_b2h, stride_b2n,
    stride_b2se, stride_b2sh, stride_b2sn,
    stride_ct, stride_ch,
    
    # Block sizes (Critical for SRAM limits)
    BLOCK_M: tl.constexpr, BLOCK_N1: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    
    num_pid_m = tl.cdiv(num_padded, BLOCK_M)
    num_pid_h = tl.cdiv(H, BLOCK_H)
    
    pid_m = pid % num_pid_m
    pid_h = pid // num_pid_m
    
    # Sorting upstream uses BLOCK_M=64 globally, but fully fused kernel runs at BLOCK_M=16
    # So every 4 `pid_m` blocks belong to the same expert!
    # To fetch the correct expert ID from `expert_ids_ptr` which is sized for BLOCK_M=64:
    sorter_block_m = 64
    stride_pid_m = sorter_block_m // BLOCK_M
    expert_id = tl.load(expert_ids_ptr + (pid_m // stride_pid_m))
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = rm < num_padded
    
    # Safe bounds indexing
    token_idx = tl.load(token_ids_ptr + rm, mask=m_mask, other=T)
    safe_token_idx = tl.where(token_idx < T, token_idx, 0)
    valid_mask = token_idx < T
    token_weights = tl.load(token_weights_ptr + rm, mask=m_mask, other=0.0)

    # ────────────────────────────────────────────────────────
    # STAGE 1: Compute GEMM1 (A @ W13) directly into SRAM
    # ────────────────────────────────────────────────────────
    
    # We must operate loop over H (7168) for A @ W13 
    # Since we need the FULL SwiGLU activation [BLOCK_M, 2048] to multiply with W2,
    # we must compute SwiGLU fully *before* starting GEMM2!
    
    # Since SwiGLU output is 2048, and B2 is [2048, 7168], 
    # A single Triton block cannot easily `for h_out in range(0, 7168)` while re-computing 
    # SwiGLU every time.
    
    # Let's use two loops:
    # Outer Loop: computes SwiGLU outputs in blocks of size BLOCK_MID (e.g. 128)
    # But wait, W2 expects the dot product across ALL 2048 SwiGLU outputs to produce 1 output channel.
    # Therefore, we MUST compute ALL 2048 SwiGLU outputs and hold them in SRAM, OR we fuse the 
    # loops by accumulating partial W2 dots as we stream SwiGLU outputs!
    
    # ────────────────────────────────────────────────────────
    # STAGE 2: Compute Full GEMM1 -> SwiGLU -> partial GEMM2
    # ────────────────────────────────────────────────────────
    
    # We output a slice of H (size BLOCK_H) per Triton thread block.
    # Grid: (num_pid_m, num_pid_h)
    
    rh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = rh < H
    
    acc_out = tl.zeros((BLOCK_M, BLOCK_H), dtype=tl.float32)
    
    # We chunk the N1 dimension (4096) into BLOCK_N1 size to fit into SRAM
    for n1_idx in range(0, N1_HALF, BLOCK_N1):
        rn1 = n1_idx + tl.arange(0, BLOCK_N1)
        rn1_3 = rn1 + N1_HALF
        
        acc_1 = tl.zeros((BLOCK_M, BLOCK_N1), dtype=tl.float32)
        acc_3 = tl.zeros((BLOCK_M, BLOCK_N1), dtype=tl.float32)
        
        # Inner loop over H (7168) for A @ W13 in this specific N1 slice
        for h_in in range(0, H, BLOCK_H):
            rh_in = h_in + tl.arange(0, BLOCK_H)
            
            # Load A
            a_ptrs = A_ptr + safe_token_idx[:, None] * stride_at + rh_in[None, :] * stride_ah
            a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
            a_scale_ptrs = A_scale_ptr + (h_in // 128) * stride_as0 + safe_token_idx[:, None] * stride_as1
            a_scale = tl.load(a_scale_ptrs, mask=m_mask[:, None], other=0.0)
            a_fp32 = a.to(tl.float32) * a_scale
            
            # Load B1 (W1)
            b1_ptrs_1 = B1_ptr + expert_id * stride_b1e + rn1[None, :] * stride_b1n + rh_in[:, None] * stride_b1h
            b1_1 = tl.load(b1_ptrs_1)
            b1_scale_ptrs_1 = B1_scale_ptr + expert_id * stride_b1se + (rn1[None, :] // 128) * stride_b1sn + ((h_in + tl.arange(0, BLOCK_H))[:, None] // 128) * stride_b1sh
            b1_scale_1 = tl.load(b1_scale_ptrs_1) # [BLOCK_H, BLOCK_N1]
            b1_1_fp32 = b1_1.to(tl.float32) * b1_scale_1
            
            # Load B1 (W3)
            b1_ptrs_3 = B1_ptr + expert_id * stride_b1e + rn1_3[None, :] * stride_b1n + rh_in[:, None] * stride_b1h
            b1_3 = tl.load(b1_ptrs_3)
            b1_scale_ptrs_3 = B1_scale_ptr + expert_id * stride_b1se + (rn1_3[None, :] // 128) * stride_b1sn + ((h_in + tl.arange(0, BLOCK_H))[:, None] // 128) * stride_b1sh
            b1_scale_3 = tl.load(b1_scale_ptrs_3) # [BLOCK_H, BLOCK_N1]
            b1_3_fp32 = b1_3.to(tl.float32) * b1_scale_3
            
            a_dequant = a_fp32.to(tl.bfloat16)
            b1_1_dequant = b1_1_fp32.to(tl.bfloat16)
            b1_3_dequant = b1_3_fp32.to(tl.bfloat16)
            acc_1 += tl.dot(a_dequant, b1_1_dequant, out_dtype=tl.float32)
            acc_3 += tl.dot(a_dequant, b1_3_dequant, out_dtype=tl.float32)

        # Apply SwiGLU over the computed block
        sig_out = 1.0 / (1.0 + tl.exp(-acc_3))
        swiglu_out = (acc_3 * sig_out) * acc_1 # [BLOCK_M, BLOCK_N1]
        
        # Immediately multiply this SwiGLU chunk with its counterpart in W2!
        # W2 is [2048, 7168], so we load W2[rn1, rh]
        # Wait, GEMM2 input is `swiglu_out` [BLOCK_M, BLOCK_N1]. 
        # B2 is W2 transposed -> loaded as [H, N1//2]
        
        # Here we dot `[BLOCK_M, BLOCK_N1]` with `W2^T [BLOCK_N1, BLOCK_H]`.
        # Transpose B2 memory fetch mapping: B2 is [H, N1//2]
        # We need it as [N1//2, H] logically
        # B2 is initially [E, 7168, 2048]. 
        # So we fetch `[rh, rn1]` from B2. 
        # Wait, GEMM1's loops were over H, so `rh` is the `H` dimension (7168). `rn1` is `N_OUT` (2048).
        # We need b2_fp32 as `[BLOCK_N1, BLOCK_H]` to multiply `swiglu_out` which is `[BLOCK_M, BLOCK_N1]`.
        # That means `tl.load` of B2 must be fetched as `[rn1[:, None], rh[None, :]]`.
        
        n1_mask = rn1 < N1_HALF
        
        b2_ptrs = B2_ptr + expert_id * stride_b2e + rh[None, :] * stride_b2h + rn1[:, None] * stride_b2n
        b2_mask = h_mask[None, :] & n1_mask[:, None]
        b2 = tl.load(b2_ptrs, mask=b2_mask, other=0.0)
        
        # B2_scale is [E, 56, 16] -> [E, H//128, N_OUT//128]
        # We load it as [rn1, 1] to match b2
        b2_scale_ptrs = B2_scale_ptr + expert_id * stride_b2se + ((pid_h * BLOCK_H) // 128) * stride_b2sh + (rn1[:, None] // 128) * stride_b2sn
        b2_scale_mask = n1_mask[:, None]
        b2_scale = tl.load(b2_scale_ptrs, mask=b2_scale_mask, other=0.0)
        
        b2_fp32 = b2.to(tl.float32) * b2_scale
        
        swiglu_bf16 = swiglu_out.to(tl.bfloat16)
        b2_dequant = b2_fp32.to(tl.bfloat16)
        acc_out += tl.dot(swiglu_bf16, b2_dequant, out_dtype=tl.float32)

    # Scale by routing weight
    acc_out *= token_weights[:, None]
    
    # Store atomically to output
    c_ptrs = C_ptr + safe_token_idx[:, None] * stride_ct + rh[None, :] * stride_ch
    tl.atomic_add(c_ptrs, acc_out, mask=valid_mask[:, None] & h_mask[None, :], sem='relaxed')

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
    output,                   # [T, 7168]         bfloat16  (destination-passing, last)
):
    T = routing_logits.shape[0]
    device = hidden_states.device
    local_start = int(local_expert_offset)
    # ── 1. Routing ──
    topk_idx, topk_weights = ds_routing(
        routing_logits, routing_bias, float(routed_scaling_factor)
    )

    # ── 2. Token sorting ──
    sorted_token_ids, block_expert_ids, sorted_weights, num_padded = \
        moe_sort_tokens(topk_idx, topk_weights, local_start, BLOCK_M, T, device)

    # Accumulate into FP32 buffer to prevent low-bit `atomic_add` rounding error loss across experts
    output_fp32 = torch.zeros((T, 7168), dtype=torch.float32, device=device)
    
    if sorted_token_ids is None or num_padded == 0:
        output.copy_(output_fp32)
        return

    # ── 3. Fused GEMM1 + SwiGLU ──
    # W13 is [E, 4096, 7168], N=4096, K=7168
    # SwiGLU outputs N_OUT=2048. Zero allocated to prevent uninitialized padding NaNs.
    # Allocate as float32 to prevent bf16 mantissa clip from failing exact bench match
    Intermediate = torch.zeros((num_padded, 2048), dtype=torch.float32, device=device)
    
    BLOCK_M_1 = 64
    BLOCK_N_1 = 64
    BLOCK_K_1 = 128
    GROUP_M_1 = 8
    
    grid1 = lambda META: (triton.cdiv(num_padded, META['BLOCK_M']) * triton.cdiv(2048, META['BLOCK_N']),)
    
    _fused_moe_gemm1_swiglu_kernel[grid1](
        A_ptr=hidden_states,
        A_scale_ptr=hidden_states_scale,
        B_ptr=gemm1_weights,
        C_ptr=Intermediate,
        B_scale_ptr=gemm1_weights_scale,
        token_ids_ptr=sorted_token_ids,
        expert_ids_ptr=block_expert_ids,
        num_padded=num_padded, T=T, H=7168, N=4096, K=7168,
        stride_at=hidden_states.stride(0), stride_ah=hidden_states.stride(1),
        stride_as0=hidden_states_scale.stride(0), stride_as1=hidden_states_scale.stride(1),
        stride_be=gemm1_weights.stride(0), stride_bn=gemm1_weights.stride(1), stride_bh=gemm1_weights.stride(2),
        stride_cm=Intermediate.stride(0), stride_cn=Intermediate.stride(1),
        stride_bse=gemm1_weights_scale.stride(0), stride_bsn=gemm1_weights_scale.stride(1), stride_bsh=gemm1_weights_scale.stride(2),
        BLOCK_M=BLOCK_M_1, BLOCK_N=BLOCK_N_1, BLOCK_K=BLOCK_K_1, GROUP_M=GROUP_M_1
    )

    # ── 4. Fused GEMM2 + Scatter Add ──
    # Intermediate is [num_padded, 2048]
    # W2 is [E, 7168, 2048]
    # We scatter add directly to output [T, 7168]
    
    BLOCK_M_2 = 64
    BLOCK_N_2 = 64
    BLOCK_K_2 = 64
    GROUP_M_2 = 8
    
    grid2 = lambda META: (triton.cdiv(num_padded, META['BLOCK_M']) * triton.cdiv(7168, META['BLOCK_N']),)

    _fused_moe_gemm2_scatter_kernel[grid2](
        A_ptr=Intermediate,
        B_ptr=gemm2_weights,
        C_ptr=output_fp32,
        B_scale_ptr=gemm2_weights_scale,
        token_weights_ptr=sorted_weights,
        token_ids_ptr=sorted_token_ids,
        expert_ids_ptr=block_expert_ids,
        T=T, num_padded=num_padded, N=7168, K=2048,
        stride_am=Intermediate.stride(0), stride_ak=Intermediate.stride(1),
        stride_be=gemm2_weights.stride(0), stride_bn=gemm2_weights.stride(1), stride_bk=gemm2_weights.stride(2),
        stride_ct=output_fp32.stride(0), stride_cn=output_fp32.stride(1),
        stride_bse=gemm2_weights_scale.stride(0), stride_bsn=gemm2_weights_scale.stride(1), stride_bsk=gemm2_weights_scale.stride(2),
        BLOCK_M=BLOCK_M_2, BLOCK_N=BLOCK_N_2, BLOCK_K=BLOCK_K_2, GROUP_M=GROUP_M_2
    )

    # Final cast to bfloat16
    output.copy_(output_fp32)
