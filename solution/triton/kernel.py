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
# Triton Routing Kernel — DeepSeek-V3 no-aux routing
# ═══════════════════════════════════════════════════════════════

@triton.jit
def triton_ds_routing_kernel(
    logits_ptr,      # [T, E_GLOBAL] f32
    bias_ptr,        # [E_GLOBAL] bf16
    topk_idx_ptr,    # [T, TOP_K] int64
    topk_wts_ptr,    # [T, TOP_K] f32
    scale_factor,    # float32
    T,               # Dynamic batch size
    E_GLOBAL: tl.constexpr,
    TOP_K: tl.constexpr, 
    TOPK_GROUP: tl.constexpr, 
    N_GROUP: tl.constexpr,
):
    """
    Computes sigmoid, group top-K masking, and global top-K routing.
    1 block per token. Block size = E_GLOBAL (256 threads).
    """
    pid = tl.program_id(0)
    if pid >= T:
        return

    # Thread index corresponds to expert index
    tid = tl.arange(0, 256)
    
    # Load logits and bias
    l_ptr = logits_ptr + pid * 256 + tid
    b_ptr = bias_ptr + tid
    logit = tl.load(l_ptr)
    bias = tl.load(b_ptr).to(tl.float32)

    s = 1.0 / (1.0 + tl.exp(-(logit)))
    sb = s + bias

    # Group configuration
    G_SIZE = 32  # 256 // 8
    
    # Shape manipulation: [N_GROUP, G_SIZE]
    group_sb = tl.reshape(sb, (8, 32))
    
    # Find max in each group
    max1 = tl.max(group_sb, axis=1)
    
    # Mask out max1 to find max2
    group_sb_no_max1 = tl.where(group_sb < max1[:, None], group_sb, -float('inf'))
    max2 = tl.max(group_sb_no_max1, axis=1)
    
    # Group scores
    g_scores = max1 + max2
    
    # Find Top-4 groups out of N_GROUP (8)
    g_mask = tl.zeros((8,), dtype=tl.int32)
    curr_g_scores = g_scores
    for _ in tl.static_range(4):
        best_g_idx = tl.argmax(curr_g_scores, axis=0)
        idx_vec = tl.arange(0, 8)
        g_mask = tl.where(idx_vec == best_g_idx, 1, g_mask)
        curr_g_scores = tl.where(idx_vec == best_g_idx, -float('inf'), curr_g_scores)
        
    # Expand group mask to element-level shape [E_GLOBAL]
    # g_mask is [8], we need [256]. We can broadcast and reshape
    g_mask_2d = g_mask[:, None] * tl.full((1, 32), 1, dtype=tl.int32)
    s_mask = tl.reshape(g_mask_2d, (256,))
    
    # Masked elements become -inf
    sb_masked = tl.where(s_mask, sb, -float('inf'))
    
    # Find Global Top-K (8) out of E_GLOBAL (256)
    curr_sb = sb_masked
    topk_wgts_sum = 0.0
    
    for k in tl.static_range(8):
        # Find max and argmax
        best_idx = tl.argmax(curr_sb, axis=0)
        
        # Save to output pointers
        out_idx_ptr = topk_idx_ptr + pid * 8 + k
        out_wts_ptr = topk_wts_ptr + pid * 8 + k
        
        is_best = tid == best_idx
        best_s = tl.sum(tl.where(is_best, s, 0.0), axis=0)
        
        tl.store(out_idx_ptr, best_idx.to(tl.int64))
        tl.store(out_wts_ptr, best_s)  # Temporarily store unnormalized weight
        topk_wgts_sum += best_s
        
        # Mask out for next iteration
        curr_sb = tl.where(is_best, -float('inf'), curr_sb)
        
    # Normalize weights
    k_tid = tl.arange(0, 16)
    k_mask = k_tid < 8
    out_wts_ptrs = topk_wts_ptr + pid * 8 + k_tid
    wts = tl.load(out_wts_ptrs, mask=k_mask)
    norm_wts = (wts / (topk_wgts_sum + 1e-20)) * scale_factor
    tl.store(out_wts_ptrs, norm_wts, mask=k_mask)


def ds_routing(logits, bias, scale_factor):
    """
    Launch wrapper for Triton routing kernel.
    """
    T = logits.shape[0]
    device = logits.device
    
    topk_idx = torch.empty((T, TOP_K), dtype=torch.int64, device=device)
    topk_weights = torch.empty((T, TOP_K), dtype=torch.float32, device=device)
    
    # 1 block per token. Number of threads is implicitly inferred by E_GLOBAL (256) -> need block size to cover it
    grid = (T, )
    
    triton_ds_routing_kernel[grid](
        logits, bias, topk_idx, topk_weights, float(scale_factor),
        T=T, E_GLOBAL=256, TOP_K=8, 
        TOPK_GROUP=4, N_GROUP=8,
        num_warps=8 # E_GLOBAL is 256, 8 warps = 256 threads
    )
    
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

    # Count tokens per expert
    counts = torch.zeros(E_LOCAL, dtype=torch.int32, device=device)
    counts.scatter_add_(0, loc_experts.long(), torch.ones_like(loc_experts, dtype=torch.int32))

    if loc_tokens.numel() < 1024:
        # CPU path: single .cpu().tolist() sync + fast Python loop
        # Faster for small T (covers T≤128, 16/19 workloads) — avoids ~25 GPU kernel launches
        counts_cpu = counts.cpu().tolist()
        # Pre-allocate padding buffers to avoid per-expert torch.full/torch.zeros
        max_padding = E_LOCAL * BLOCK_M  # worst case
        pad_token_buf = torch.full((max_padding,), T, device=device, dtype=torch.int64)
        pad_weight_buf = torch.zeros(max_padding, device=device)
        pad_offset = 0
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
                padded_tokens_list.append(pad_token_buf[pad_offset:pad_offset + pad_cnt])
                padded_weights_list.append(pad_weight_buf[pad_offset:pad_offset + pad_cnt])
                pad_offset += pad_cnt
            block_experts_list.extend([e] * num_blocks)
            offset += cnt
        sorted_token_ids = torch.cat(padded_tokens_list)
        sorted_weights = torch.cat(padded_weights_list)
        block_expert_ids = torch.tensor(block_experts_list, device=device, dtype=torch.int32)
        num_padded = sorted_token_ids.shape[0]
    else:
        # GPU path: vectorized cumsum + scatter (faster for large T)
        padded_counts = ((counts + BLOCK_M - 1) // BLOCK_M) * BLOCK_M
        offsets = torch.cumsum(padded_counts, dim=0)
        total = offsets[-1].item()  # ONE sync point
        starts = offsets - padded_counts

        # Compute within-expert offsets for scatter
        cum_actual = torch.cumsum(counts, dim=0)
        cum_shifted = torch.cat([torch.zeros(1, dtype=torch.int32, device=device), cum_actual[:-1]])
        arange = torch.arange(loc_tokens.numel(), device=device, dtype=torch.int32)
        within_offset = arange - cum_shifted[loc_experts.long()]
        dest_idx = starts[loc_experts.long()] + within_offset

        # Scatter tokens and weights into padded buffer
        sorted_token_ids = torch.full((total,), T, device=device, dtype=torch.int64)
        sorted_token_ids[dest_idx.long()] = loc_tokens
        sorted_weights = torch.zeros(total, device=device)
        sorted_weights[dest_idx.long()] = loc_weights

        # Block expert IDs via repeat_interleave
        blocks_per_expert = (padded_counts // BLOCK_M).long()
        block_expert_ids = torch.repeat_interleave(
            torch.arange(E_LOCAL, device=device, dtype=torch.int32),
            blocks_per_expert)
        num_padded = total

    return sorted_token_ids, block_expert_ids, sorted_weights, num_padded


# ═══════════════════════════════════════════════════════════════
# Triton Kernel 1: Fused GEMM1 + SwiGLU
# Computes: SwiGLU( A[sorted_idx] @ W13 )
# ═══════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        # FA4-inspired: GROUP_M=16 for better L2 cache reuse of weight tiles
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 16}, num_warps=4, num_stages=3),
    ],
    key=['num_padded', 'N', 'K'],
    restore_value=['C_ptr'],
)
@triton.jit
def _fused_moe_gemm1_swiglu_kernel(
    # Data pointers
    A_ptr,            # [T, H] fp8 (hidden_states)
    A_scale_ptr,      # [H//128, T] fp32 (hidden_states_scale)
    B_ptr,            # [E, N, H] fp8 (W13 weights, transposed physically or logically)
    C_ptr,            # [num_padded, N//2] fp32 (intermediate SwiGLU output)
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

    safe_token_idx = tl.where(token_idx < T, token_idx, 0)

    # Hoist loop-invariant pointer bases outside K-loop (56 iterations)
    a_base = A_ptr + safe_token_idx[:, None] * stride_at
    a_scale_base = A_scale_ptr + safe_token_idx * stride_as1
    b_base_1 = B_ptr + expert_id * stride_be + rn_1[None, :] * stride_bn
    b_base_3 = B_ptr + expert_id * stride_be + rn_3[None, :] * stride_bn
    b_scale_base_1 = B_scale_ptr + expert_id * stride_bse + (rn_1[None, :] // 128) * stride_bsn
    b_scale_base_3 = B_scale_ptr + expert_id * stride_bse + (rn_3[None, :] // 128) * stride_bsn

    # K loop
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)

        # Load A: fp8
        a = tl.load(a_base + rk[None, :] * stride_ah, mask=m_mask[:, None], other=0.0)

        # Load A scale: fp32 [BLOCK_M]
        a_scale = tl.load(a_scale_base + (k // 128) * stride_as0, mask=m_mask, other=0.0)

        # Load B for W1: fp8
        b_1 = tl.load(b_base_1 + rk[:, None] * stride_bh)

        # Load B scale for W1: fp32 [1, BLOCK_N]
        b_scale_1 = tl.load(b_scale_base_1 + (k // 128) * stride_bsh)

        # Load B for W3: fp8
        b_3 = tl.load(b_base_3 + rk[:, None] * stride_bh)

        # Load B scale for W3: fp32 [1, BLOCK_N]
        b_scale_3 = tl.load(b_scale_base_3 + (k // 128) * stride_bsh)

        # Native FP8 tensor cores — 2x throughput vs TF32 on Blackwell
        partial_1 = tl.dot(a, b_1, out_dtype=tl.float32)
        partial_3 = tl.dot(a, b_3, out_dtype=tl.float32)

        # Post-dot scale: BLOCK_K == QBLOCK so scales are constant within each dot
        # (a * a_scale) @ (b * b_scale) = (a_scale ⊗ b_scale) * (a @ b)
        scale_1 = a_scale[:, None] * b_scale_1
        scale_3 = a_scale[:, None] * b_scale_3
        acc_1 += partial_1 * scale_1
        acc_3 += partial_3 * scale_3

    # Epilogue: SwiGLU (Note: PyTorch baseline does silu(W3) * W1)
    sig_out = 1.0 / (1.0 + tl.exp(-acc_3))
    swiglu_out = (acc_3 * sig_out) * acc_1
    
    # Store to C as fp32 (bf16 causes too much precision loss — 6/19 PASSED)
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, swiglu_out, mask=m_mask[:, None])


# ═══════════════════════════════════════════════════════════════
# Triton Kernel 2: Fused GEMM2 + Routing Weight + Scatter Add
# Computes: Output[token_id] += (Intermediate[sorted_idx] @ W2) * routing_weight
# ═══════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        # FA4-inspired: GROUP_M=16 for better L2 cache reuse of weight tiles
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 16}, num_warps=4, num_stages=3),
    ],
    key=['num_padded', 'N', 'K'],
    restore_value=['C_ptr'],
)
@triton.jit
def _fused_moe_gemm2_scatter_kernel(
    # Data pointers
    A_ptr,            # [num_padded, K] fp32 (intermediate SwiGLU output)
    B_ptr,            # [E, N, K] fp8 (W2 weights trans)
    C_ptr,            # [T, N] fp32 (output accumulation buffer)
    B_scale_ptr,      # [E, N//128, K//128] fp32 (W2 block scales)
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

    safe_rm = tl.where(rm < num_padded, rm, 0)

    # Hoist loop-invariant pointer bases outside K-loop (16 iterations)
    a_base = A_ptr + safe_rm[:, None] * stride_am
    b_base = B_ptr + expert_id * stride_be + rn[None, :] * stride_bn
    b_scale_base = B_scale_ptr + expert_id * stride_bse + (rn[None, :] // 128) * stride_bsn

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)

        # Load A: fp32 from Intermediate buffer
        a = tl.load(a_base + rk[None, :] * stride_ak, mask=m_mask[:, None], other=0.0)

        # Load B: fp8 (N=7168 always divisible by BLOCK_N, no mask needed)
        b = tl.load(b_base + rk[:, None] * stride_bk)

        # Post-dot B-scale: BLOCK_K == QBLOCK so b_scale constant along K within tile
        # a @ (b * scale) = (a @ b) * scale — fewer muls + better TF32 precision
        b_scale = tl.load(b_scale_base + (k // 128) * stride_bsk)
        partial = tl.dot(a, b.to(tl.float32), out_dtype=tl.float32)
        acc += partial * b_scale
        
    # Scale by routing weights
    out = acc * token_weights[:, None]

    # Scatter Add directly to output tensor C[T, N]
    # Cap token_idx to 0 to prevent out-of-bounds pointer crashes even when masked
    safe_token_idx = tl.where(token_idx < T, token_idx, 0)
    c_ptrs = C_ptr + safe_token_idx[:, None] * stride_ct + rn[None, :] * stride_cn
    
    # Triton atomic_add into FP32 buffer
    tl.atomic_add(c_ptrs, out, mask=valid_mask[:, None] & n_mask[None, :], sem='relaxed')

# ═══════════════════════════════════════════════════════════════
# Pre-allocated buffer cache — reuse output_fp32/Intermediate
# ═══════════════════════════════════════════════════════════════

_buf_cache = {}


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

    # ── 3. Pre-allocated FP32 accumulation buffer ──
    bkey = T
    if bkey in _buf_cache:
        output_fp32 = _buf_cache[bkey]
        output_fp32.zero_()
    else:
        output_fp32 = torch.zeros((T, 7168), dtype=torch.float32, device=device)
        _buf_cache[bkey] = output_fp32

    if sorted_token_ids is None or num_padded == 0:
        output.copy_(output_fp32)
        return

    # ── 4. Fused GEMM1 + SwiGLU ──
    Intermediate = torch.empty((num_padded, 2048), dtype=torch.float32, device=device)

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
    )

    # ── 5. Fused GEMM2 + Scatter Add ──
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
    )

    # Final cast to bfloat16
    output.copy_(output_fp32)
