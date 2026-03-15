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
BLOCK_M_TINY = 16   # for medium-T regimes where padding dominates GEMM efficiency
BLOCK_M_SMALL = 32  # for moderate batches
BLOCK_M_LARGE = 64  # default tile for larger batches
BLOCK_M_XLARGE = 128 # for very large batches (T>=2048, better K-loop reuse)
BLOCK_M_CANDIDATES = (16, 32, 64, 128)
TINY_BATCH_TOPK_TOKENS = 1024  # if T * TOP_K <= this, prefer the smallest padding tile
SMALL_BATCH_TOPK_TOKENS = 4096  # if T * TOP_K <= this, use BLOCK_M_SMALL
LARGE_BATCH_TOPK_TOKENS = 16384 # if T * TOP_K > this, use BLOCK_M_XLARGE
HISTOGRAM_BLOCK_M_MAX_TOPK_TOKENS = 8192
SMALL_MEDIUM_T_MIN = 32
SMALL_MEDIUM_T_MAX = 64
UPPER_MEDIUM_T_MIN = 65
UPPER_MEDIUM_T_MAX = 128
BLOCK_K = 128  # K-block (must equal QBLOCK for scale alignment)
GROUP_SIZE_M = 8  # L2 cache reuse grouping
SORT_BLOCK_ITEMS = 256
PARALLEL_SORT_MIN_TILES = 128
T1_GENERIC_BLOCK_M = 16
T1_GENERIC_MAX_PADDED = TOP_K * T1_GENERIC_BLOCK_M


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
    tid = tl.arange(0, E_GLOBAL)
    
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
    idx_vec = tl.arange(0, N_GROUP)
    for _ in tl.static_range(4):
        best_g_idx = tl.argmax(curr_g_scores, axis=0)
        g_mask = tl.where(idx_vec == best_g_idx, 1, g_mask)
        curr_g_scores = tl.where(idx_vec == best_g_idx, -float('inf'), curr_g_scores)
        
    # Expand group mask to element-level shape [E_GLOBAL]
    # g_mask is [8], we need [256]. We can broadcast and reshape
    g_mask_2d = g_mask[:, None] * tl.full((1, 32), 1, dtype=tl.int32)
    s_mask = tl.reshape(g_mask_2d, (256,)) !=0
    
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
        
        tl.store(out_idx_ptr, best_idx.to(tl.int32))
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


def ds_routing(logits, bias, scale_factor, topk_idx=None, topk_weights=None):
    """
    Launch wrapper for Triton routing kernel.
    """
    T = logits.shape[0]
    device = logits.device
    
    if topk_idx is None:
        topk_idx = torch.empty((T, TOP_K), dtype=torch.int32, device=device)
    if topk_weights is None:
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


@triton.jit
def triton_ds_routing_t1_local_kernel(
    logits_ptr,
    bias_ptr,
    local_expert_ids_ptr,
    local_expert_wts_ptr,
    num_local_ptr,
    sorted_tokens_ptr,
    sorted_weights_ptr,
    scatter_map_ptr,
    block_offsets_ptr,
    total_blocks_ptr,
    scale_factor,
    local_start: tl.constexpr,
    E_LOCAL: tl.constexpr,
    E_GLOBAL: tl.constexpr,
    TOP_K: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    N_GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    tid = tl.arange(0, E_GLOBAL)
    logit = tl.load(logits_ptr + tid)
    bias = tl.load(bias_ptr + tid).to(tl.float32)

    s = 1.0 / (1.0 + tl.exp(-(logit)))
    sb = s + bias

    group_sb = tl.reshape(sb, (N_GROUP, E_GLOBAL // N_GROUP))
    max1 = tl.max(group_sb, axis=1)
    group_sb_no_max1 = tl.where(group_sb < max1[:, None], group_sb, -float('inf'))
    max2 = tl.max(group_sb_no_max1, axis=1)
    g_scores = max1 + max2

    g_mask = tl.zeros((N_GROUP,), dtype=tl.int32)
    curr_g_scores = g_scores
    idx_vec = tl.arange(0, N_GROUP)
    for _ in tl.static_range(TOPK_GROUP):
        best_g_idx = tl.argmax(curr_g_scores, axis=0)
        g_mask = tl.where(idx_vec == best_g_idx, 1, g_mask)
        curr_g_scores = tl.where(idx_vec == best_g_idx, -float('inf'), curr_g_scores)

    g_mask_2d = g_mask[:, None] * tl.full((1, E_GLOBAL // N_GROUP), 1, dtype=tl.int32)
    s_mask = tl.reshape(g_mask_2d, (E_GLOBAL,)) != 0
    curr_sb = tl.where(s_mask, sb, -float('inf'))

    local_count = tl.zeros((), dtype=tl.int32)
    topk_wgts_sum = tl.zeros((), dtype=tl.float32)
    slots = tl.arange(0, TOP_K)
    tl.store(local_expert_ids_ptr + slots, 0)
    tl.store(local_expert_wts_ptr + slots, 0.0)
    tl.store(scatter_map_ptr + slots, -1)

    prep_row = tl.arange(0, TOP_K * BLOCK_M)
    tl.store(sorted_tokens_ptr + prep_row, 1)
    tl.store(sorted_weights_ptr + prep_row, 0.0)

    for _ in tl.static_range(TOP_K):
        best_idx = tl.argmax(curr_sb, axis=0)
        is_best = tid == best_idx
        best_s = tl.sum(tl.where(is_best, s, 0.0), axis=0)
        topk_wgts_sum += best_s

        is_local = (best_idx >= local_start) & (best_idx < local_start + E_LOCAL)
        tl.store(local_expert_ids_ptr + local_count, (best_idx - local_start).to(tl.int32), mask=is_local)
        tl.store(local_expert_wts_ptr + local_count, best_s, mask=is_local)
        local_count += is_local.to(tl.int32)
        curr_sb = tl.where(is_best, -float('inf'), curr_sb)

    valid_slots = slots < local_count
    local_wts = tl.load(local_expert_wts_ptr + slots, mask=valid_slots, other=0.0)
    local_wts = (local_wts / (topk_wgts_sum + 1e-20)) * scale_factor
    tl.store(local_expert_wts_ptr + slots, local_wts, mask=valid_slots)
    tl.store(num_local_ptr, local_count)

    running = 0
    for e in tl.static_range(E_LOCAL):
        tl.store(block_offsets_ptr + e, running)
        present = 0
        present_slot = 0
        for slot in tl.static_range(TOP_K):
            is_valid = slot < local_count
            expert_id = tl.load(local_expert_ids_ptr + slot, mask=is_valid, other=0)
            is_match = is_valid & (expert_id == e)
            present = tl.where(is_match, 1, present)
            present_slot = tl.where(is_match, slot, present_slot)
        if present:
            dest = running * BLOCK_M
            block_row = tl.arange(0, BLOCK_M)
            tl.store(sorted_tokens_ptr + dest + block_row, 1)
            tl.store(sorted_tokens_ptr + dest, 0)
            token_w = tl.load(local_expert_wts_ptr + present_slot)
            tl.store(sorted_weights_ptr + dest, token_w)
            tl.store(scatter_map_ptr + present_slot, dest)
            running += 1
    tl.store(block_offsets_ptr + E_LOCAL, running)
    tl.store(total_blocks_ptr, running)


def ds_routing_t1_local(
    logits,
    bias,
    scale_factor,
    local_start,
    local_expert_ids,
    local_expert_wts,
    num_local,
    sorted_token_ids,
    sorted_weights,
    scatter_map,
    block_offsets,
    total_blocks,
):
    triton_ds_routing_t1_local_kernel[(1,)](
        logits,
        bias,
        local_expert_ids,
        local_expert_wts,
        num_local,
        sorted_token_ids,
        sorted_weights,
        scatter_map,
        block_offsets,
        total_blocks,
        float(scale_factor),
        local_start=local_start,
        E_LOCAL=E_LOCAL,
        E_GLOBAL=E_GLOBAL,
        TOP_K=TOP_K,
        TOPK_GROUP=TOPK_GROUP,
        N_GROUP=N_GROUP,
        BLOCK_M=T1_GENERIC_BLOCK_M,
        num_warps=8,
    )


# ═══════════════════════════════════════════════════════════════
# Token Sorting (Triton)
# ═══════════════════════════════════════════════════════════════

@triton.jit
def triton_sort_and_scatter_kernel(
    topk_idx_ptr,       # [T, TOP_K]
    topk_wts_ptr,       # [T, TOP_K]
    sorted_tokens_ptr,  # [MAX_PADDED]
    sorted_weights_ptr, # [MAX_PADDED]
    scatter_map_ptr,    # [T * TOP_K] int32 -- output: where each (token, expert) ended up
    block_offsets_ptr,  # [E_LOCAL + 1] -> 33
    total_blocks_ptr,   # [1]
    counts_ptr,         # [E_LOCAL] workspace
    local_start: tl.constexpr,
    T,                  # scalar
    TOP_K: tl.constexpr,
    E_LOCAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    MAX_PADDED,         # scalar
):
    tid = tl.arange(0, 256)

    e_idx = tl.arange(0, 32)
    e_mask = e_idx < E_LOCAL
    tl.store(counts_ptr + e_idx, 0, mask=e_mask)
    tl.debug_barrier()

    N = T * TOP_K

    offset = 0
    while offset < N:
        idx = offset + tid
        mask = idx < N
        exp = tl.load(topk_idx_ptr + idx, mask=mask, other=-1)
        is_local = (exp >= local_start) & (exp < local_start + E_LOCAL)
        local_exp = exp - local_start
        ptr = counts_ptr + tl.where(is_local, local_exp, 0)
        tl.atomic_add(ptr, 1, mask=is_local, sem='relaxed')
        offset += 256
    tl.debug_barrier()

    cnts = tl.load(counts_ptr + e_idx, mask=e_mask, other=0)
    blocks = (cnts + (BLOCK_M - 1)) // BLOCK_M
    padded_cnts = blocks * BLOCK_M
    inc_sum = tl.cumsum(padded_cnts, axis=0)
    offsets = inc_sum - padded_cnts
    total_blocks = tl.sum(blocks, axis=0)

    tl.store(block_offsets_ptr + e_idx, offsets // BLOCK_M, mask=e_mask)
    tl.store(block_offsets_ptr + E_LOCAL, total_blocks)
    tl.store(total_blocks_ptr, total_blocks)
    tl.store(counts_ptr + e_idx, offsets, mask=e_mask)
    tl.debug_barrier()

    num_padded = total_blocks * BLOCK_M
    offset = 0
    while offset < num_padded:
        idx = offset + tid
        mask = idx < num_padded
        tl.store(sorted_tokens_ptr + idx, T, mask=mask)
        tl.store(sorted_weights_ptr + idx, 0.0, mask=mask)
        offset += 256
    tl.debug_barrier()

    offset = 0
    while offset < N:
        idx = offset + tid
        mask = idx < N
        exp = tl.load(topk_idx_ptr + idx, mask=mask, other=-1)
        wgt = tl.load(topk_wts_ptr + idx, mask=mask, other=0.0)
        is_local = (exp >= local_start) & (exp < local_start + E_LOCAL)
        local_exp = exp - local_start
        ptr = counts_ptr + tl.where(is_local, local_exp, 0)
        dest = tl.atomic_add(ptr, 1, mask=is_local, sem='relaxed')
        tok_id = idx // TOP_K

        tl.store(sorted_tokens_ptr + tl.where(is_local, dest, 0), tok_id, mask=is_local)
        tl.store(sorted_weights_ptr + tl.where(is_local, dest, 0), wgt, mask=is_local)
        tl.store(scatter_map_ptr + idx, dest, mask=is_local)
        tl.store(scatter_map_ptr + idx, -1, mask=mask & (~is_local))
        offset += 256


@triton.jit
def triton_sort_histogram_kernel(
    topk_idx_ptr,
    partial_counts_ptr,
    local_start: tl.constexpr,
    N,
    E_LOCAL: tl.constexpr,
    BLOCK_ITEMS: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_ITEMS + tl.arange(0, BLOCK_ITEMS)
    mask = idx < N
    exp = tl.load(topk_idx_ptr + idx, mask=mask, other=-1)
    is_local = mask & (exp >= local_start) & (exp < local_start + E_LOCAL)
    local_exp = exp - local_start

    for e in tl.static_range(E_LOCAL):
        count = tl.sum(((local_exp == e) & is_local).to(tl.int32), axis=0)
        tl.store(partial_counts_ptr + pid * E_LOCAL + e, count)


@triton.jit
def triton_sort_layout_kernel(
    partial_counts_ptr,
    tile_offsets_ptr,
    block_offsets_ptr,
    total_blocks_ptr,
    num_tiles,
    E_LOCAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    e_idx = tl.arange(0, E_LOCAL)
    total_counts = tl.zeros((E_LOCAL,), dtype=tl.int32)

    tile = 0
    while tile < num_tiles:
        total_counts += tl.load(partial_counts_ptr + tile * E_LOCAL + e_idx)
        tile += 1

    blocks = (total_counts + (BLOCK_M - 1)) // BLOCK_M
    padded_counts = blocks * BLOCK_M
    inc_sum = tl.cumsum(padded_counts, axis=0)
    global_offsets = inc_sum - padded_counts
    total_blocks = tl.sum(blocks, axis=0)

    tl.store(block_offsets_ptr + e_idx, global_offsets // BLOCK_M)
    tl.store(block_offsets_ptr + E_LOCAL, total_blocks)
    tl.store(total_blocks_ptr, total_blocks)

    running = tl.zeros((E_LOCAL,), dtype=tl.int32)
    tile = 0
    while tile < num_tiles:
        counts = tl.load(partial_counts_ptr + tile * E_LOCAL + e_idx)
        tl.store(tile_offsets_ptr + tile * E_LOCAL + e_idx, global_offsets + running)
        running += counts
        tile += 1


@triton.jit
def triton_init_sorted_buffers_kernel(
    sorted_tokens_ptr,
    sorted_weights_ptr,
    total_blocks_ptr,
    T,
    BLOCK_M: tl.constexpr,
    MAX_PADDED,
    BLOCK_ITEMS: tl.constexpr,
):
    pid = tl.program_id(0)
    total_blocks = tl.load(total_blocks_ptr)
    num_padded = total_blocks * BLOCK_M
    idx = pid * BLOCK_ITEMS + tl.arange(0, BLOCK_ITEMS)
    mask = (idx < num_padded) & (idx < MAX_PADDED)
    tl.store(sorted_tokens_ptr + idx, T, mask=mask)
    tl.store(sorted_weights_ptr + idx, 0.0, mask=mask)


@triton.jit
def triton_sort_scatter_kernel(
    topk_idx_ptr,
    topk_wts_ptr,
    sorted_tokens_ptr,
    sorted_weights_ptr,
    scatter_map_ptr,
    tile_offsets_ptr,
    local_start: tl.constexpr,
    N,
    TOP_K: tl.constexpr,
    E_LOCAL: tl.constexpr,
    BLOCK_ITEMS: tl.constexpr,
):
    pid = tl.program_id(0)
    e_idx = tl.arange(0, E_LOCAL)
    tile_base = tl.load(tile_offsets_ptr + pid * E_LOCAL + e_idx)
    running = tl.zeros((E_LOCAL,), dtype=tl.int32)
    start = pid * BLOCK_ITEMS

    for i in tl.static_range(BLOCK_ITEMS):
        idx = start + i
        mask = idx < N
        exp = tl.load(topk_idx_ptr + idx, mask=mask, other=-1)
        wgt = tl.load(topk_wts_ptr + idx, mask=mask, other=0.0)
        is_local = mask & (exp >= local_start) & (exp < local_start + E_LOCAL)
        local_exp = exp - local_start
        is_match = e_idx == local_exp
        dest = tl.sum(tl.where(is_match, tile_base + running, 0), axis=0)
        tok_id = idx // TOP_K

        tl.store(scatter_map_ptr + idx, -1, mask=mask)
        tl.store(sorted_tokens_ptr + dest, tok_id, mask=is_local)
        tl.store(sorted_weights_ptr + dest, wgt, mask=is_local)
        tl.store(scatter_map_ptr + idx, dest, mask=is_local)
        running += is_match.to(tl.int32) * is_local.to(tl.int32)


def parallel_sort_and_scatter(
    topk_idx,
    topk_wts,
    sorted_token_ids,
    sorted_weights,
    scatter_map,
    block_offsets,
    total_blocks,
    partial_counts,
    tile_offsets,
    local_start,
    T,
    block_m,
    max_padded,
    counts_workspace=None,
):
    num_items = T * TOP_K
    num_tiles = triton.cdiv(num_items, SORT_BLOCK_ITEMS)
    if num_tiles < PARALLEL_SORT_MIN_TILES:
        triton_sort_and_scatter_kernel[(1,)](
            topk_idx,
            topk_wts,
            sorted_token_ids,
            sorted_weights,
            scatter_map,
            block_offsets,
            total_blocks,
            counts_workspace,
            local_start,
            T,
            TOP_K,
            E_LOCAL,
            block_m,
            max_padded,
            num_warps=8,
        )
        return

    triton_sort_histogram_kernel[(num_tiles,)](
        topk_idx,
        partial_counts,
        local_start=local_start,
        N=num_items,
        E_LOCAL=E_LOCAL,
        BLOCK_ITEMS=SORT_BLOCK_ITEMS,
        num_warps=4,
    )
    triton_sort_layout_kernel[(1,)](
        partial_counts,
        tile_offsets,
        block_offsets,
        total_blocks,
        num_tiles,
        E_LOCAL=E_LOCAL,
        BLOCK_M=block_m,
        num_warps=1,
    )

    triton_init_sorted_buffers_kernel[(triton.cdiv(max_padded, SORT_BLOCK_ITEMS),)](
        sorted_token_ids,
        sorted_weights,
        total_blocks,
        T,
        BLOCK_M=block_m,
        MAX_PADDED=max_padded,
        BLOCK_ITEMS=SORT_BLOCK_ITEMS,
        num_warps=4,
    )
    triton_sort_scatter_kernel[(num_tiles,)](
        topk_idx,
        topk_wts,
        sorted_token_ids,
        sorted_weights,
        scatter_map,
        tile_offsets,
        local_start=local_start,
        N=num_items,
        TOP_K=TOP_K,
        E_LOCAL=E_LOCAL,
        BLOCK_ITEMS=SORT_BLOCK_ITEMS,
        num_warps=4,
    )


# ═══════════════════════════════════════════════════════════════
# Triton Kernel 1: Fused GEMM1 + SwiGLU
# Computes: SwiGLU( A[sorted_idx] @ W13 )
# ═══════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        # Default configs
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        # FA4-inspired: GROUP_M=16 for better L2 cache reuse of weight tiles
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 16}, num_warps=4, num_stages=3),
        # New extensive sweeps for T=512 (which produces large num_padded)
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=5),
        # Varied Group Sizes
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 32}, num_warps=8, num_stages=3),
    ],
    key=['MAX_PID_M', 'N', 'K', 'BLOCK_M'],
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
    block_offsets_ptr,# [E_LOCAL + 1] int32 (starting M-block for each expert)
    total_blocks_ptr, # [1] int32
    
    # Dimensions
    MAX_PID_M: tl.constexpr, T: tl.constexpr, H: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    
    # Strides
    stride_at, stride_ah,
    stride_as0, stride_as1,
    stride_be, stride_bn, stride_bh,
    stride_cm, stride_cn,
    stride_bse, stride_bsn, stride_bsh,
    E_LOCAL: tl.constexpr,

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
    num_pid_n = tl.cdiv(N_OUT, BLOCK_N)

    total_blocks = tl.load(total_blocks_ptr)
    if total_blocks <= 0:
        return
    if pid >= total_blocks * num_pid_n:
        return
    num_pid_m = total_blocks
    
    # Grouped Swizzle
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
        
    pid_n = (pid % num_pid_in_group) // group_size_m

    e_idx = tl.arange(0, E_LOCAL)
    b_start = tl.load(block_offsets_ptr + e_idx)
    b_end = tl.load(block_offsets_ptr + e_idx + 1)
    valid = (b_start <= pid_m) & (pid_m < b_end)
    expert_id = tl.argmax(valid.to(tl.int32), axis=0)
    
    # Offsets for M (tokens) and N (output channels)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # SwiGLU needs both rn (W1) and rn + N_OUT (W3)
    rn_1 = rn
    rn_3 = rn + N_OUT
    
    # Load token indices (buffer initialized to T, so no out-of-bounds error)
    token_idx = tl.load(token_ids_ptr + rm)
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
    tl.store(c_ptrs, swiglu_out)


# ═══════════════════════════════════════════════════════════════
# Triton Kernel 2: Fused GEMM2 + Routing Weight + Scatter Add
# Computes: Output[token_id] += (Intermediate[sorted_idx] @ W2) * routing_weight
# ═══════════════════════════════════════════════════════════════

@triton.autotune(
    configs=[
        # Default configs
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        # FA4-inspired: GROUP_M=16 for better L2 cache reuse of weight tiles
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 16}, num_warps=4, num_stages=3),
        # New extensive sweeps for T=512
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=6),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=6),
        # Varied Group Sizes
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 1}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 32}, num_warps=8, num_stages=4),
    ],
    key=['MAX_PID_M', 'N', 'K', 'BLOCK_M'],
    restore_value=['C_ptr'],
)
@triton.jit
def _fused_moe_gemm2_kernel(
    # Data pointers
    A_ptr,            # [num_padded, K] fp32 (intermediate SwiGLU output)
    B_ptr,            # [E, N, K] fp8 (W2 weights trans)
    C_ptr,            # [num_padded, N] fp32 (expert_out buffer)
    B_scale_ptr,      # [E, N//128, K//128] fp32 (W2 block scales)
    token_weights_ptr,# [num_padded] fp32 (routing weights for each slot)
    block_offsets_ptr,# [E_LOCAL + 1] int32
    total_blocks_ptr, # [1] int32
    
    # Dimensions
    MAX_PID_M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    
    # Strides
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_bse, stride_bsn, stride_bsk,
    E_LOCAL: tl.constexpr,
    
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    total_blocks = tl.load(total_blocks_ptr)
    if total_blocks <= 0:
        return
    if pid >= total_blocks * num_pid_n:
        return
    num_pid_m = total_blocks
    
    # Grouped Swizzle
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
        
    pid_n = (pid % num_pid_in_group) // group_size_m

    e_idx = tl.arange(0, E_LOCAL)
    b_start = tl.load(block_offsets_ptr + e_idx)
    b_end = tl.load(block_offsets_ptr + e_idx + 1)
    valid = (b_start <= pid_m) & (pid_m < b_end)
    expert_id = tl.argmax(valid.to(tl.int32), axis=0)
    
    # Offsets
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Load routing weights
    token_weights = tl.load(token_weights_ptr + rm)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop invariant bases
    a_base = A_ptr + rm[:, None] * stride_am
    b_base = B_ptr + expert_id * stride_be + rn[None, :] * stride_bn
    b_scale_base = B_scale_ptr + expert_id * stride_bse + (rn[None, :] // 128) * stride_bsn

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)

        # Load A: fp32 from Intermediate buffer
        a = tl.load(a_base + rk[None, :] * stride_ak)

        # Load B: fp8
        b = tl.load(b_base + rk[:, None] * stride_bk)

        # Post-dot B-scale
        b_scale = tl.load(b_scale_base + (k // 128) * stride_bsk)
        partial = tl.dot(a, b.to(tl.float32), out_dtype=tl.float32)
        acc += partial * b_scale
        
    # Scale by routing weights and store (NO atomic!)
    out = acc * token_weights[:, None]
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 1}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 2}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 1}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 2}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 4}, num_warps=4, num_stages=2),
    ],
    key=['MAX_PID_M', 'N', 'K', 'BLOCK_M'],
    restore_value=['C_ptr'],
)
@triton.jit
def _small_medium_fused_moe_gemm1_swiglu_kernel(
    A_ptr,
    A_scale_ptr,
    B_ptr,
    C_ptr,
    B_scale_ptr,
    token_ids_ptr,
    block_offsets_ptr,
    total_blocks_ptr,
    MAX_PID_M: tl.constexpr, T: tl.constexpr, H: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_at, stride_ah,
    stride_as0, stride_as1,
    stride_be, stride_bn, stride_bh,
    stride_cm, stride_cn,
    stride_bse, stride_bsn, stride_bsh,
    E_LOCAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    N_OUT = N // 2
    num_pid_n = tl.cdiv(N_OUT, BLOCK_N)

    total_blocks = tl.load(total_blocks_ptr)
    if total_blocks <= 0:
        return
    if pid >= total_blocks * num_pid_n:
        return
    num_pid_m = total_blocks

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Small-medium batches are fixed-cost sensitive, so avoid a full 32-way scan here.
    lo = pid_m - pid_m
    hi = lo + E_LOCAL
    for _ in tl.static_range(0, 6):
        mid = (lo + hi) // 2
        upper = tl.load(block_offsets_ptr + mid + 1)
        go_left = pid_m < upper
        hi = tl.where(go_left, mid, hi)
        lo = tl.where(go_left, lo, mid + 1)
    expert_id = lo

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rn_1 = rn
    rn_3 = rn + N_OUT
    token_idx = tl.load(token_ids_ptr + rm)
    m_mask = token_idx < T
    safe_token_idx = tl.where(m_mask, token_idx, 0)

    acc_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_base = A_ptr + safe_token_idx[:, None] * stride_at
    a_scale_base = A_scale_ptr + safe_token_idx * stride_as1
    b_base_1 = B_ptr + expert_id * stride_be + rn_1[None, :] * stride_bn
    b_base_3 = B_ptr + expert_id * stride_be + rn_3[None, :] * stride_bn
    b_scale_base_1 = B_scale_ptr + expert_id * stride_bse + (rn_1[None, :] // 128) * stride_bsn
    b_scale_base_3 = B_scale_ptr + expert_id * stride_bse + (rn_3[None, :] // 128) * stride_bsn

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_base + rk[None, :] * stride_ah, mask=m_mask[:, None], other=0.0)
        a_scale = tl.load(a_scale_base + (k // 128) * stride_as0, mask=m_mask, other=0.0)
        b_1 = tl.load(b_base_1 + rk[:, None] * stride_bh)
        b_3 = tl.load(b_base_3 + rk[:, None] * stride_bh)
        b_scale_1 = tl.load(b_scale_base_1 + (k // 128) * stride_bsh)
        b_scale_3 = tl.load(b_scale_base_3 + (k // 128) * stride_bsh)
        partial_1 = tl.dot(a, b_1, out_dtype=tl.float32)
        partial_3 = tl.dot(a, b_3, out_dtype=tl.float32)
        acc_1 += partial_1 * (a_scale[:, None] * b_scale_1)
        acc_3 += partial_3 * (a_scale[:, None] * b_scale_3)

    swiglu_out = (acc_3 * (1.0 / (1.0 + tl.exp(-acc_3)))) * acc_1
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, swiglu_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 1}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 2}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 1}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 2}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 1}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 2}, num_warps=4, num_stages=3),
    ],
    key=['MAX_PID_M', 'N', 'K', 'BLOCK_M'],
    restore_value=['C_ptr'],
)
@triton.jit
def _small_medium_fused_moe_gemm2_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    B_scale_ptr,
    token_weights_ptr,
    block_offsets_ptr,
    total_blocks_ptr,
    MAX_PID_M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_bse, stride_bsn, stride_bsk,
    E_LOCAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    total_blocks = tl.load(total_blocks_ptr)
    if total_blocks <= 0:
        return
    if pid >= total_blocks * num_pid_n:
        return
    num_pid_m = total_blocks

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    e_idx = tl.arange(0, E_LOCAL)
    b_start = tl.load(block_offsets_ptr + e_idx)
    b_end = tl.load(block_offsets_ptr + e_idx + 1)
    valid = (b_start <= pid_m) & (pid_m < b_end)
    expert_id = tl.argmax(valid.to(tl.int32), axis=0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    token_weights = tl.load(token_weights_ptr + rm)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_base = A_ptr + rm[:, None] * stride_am
    b_base = B_ptr + expert_id * stride_be + rn[None, :] * stride_bn
    b_scale_base = B_scale_ptr + expert_id * stride_bse + (rn[None, :] // 128) * stride_bsn

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_base + rk[None, :] * stride_ak)
        b = tl.load(b_base + rk[:, None] * stride_bk)
        b_scale = tl.load(b_scale_base + (k // 128) * stride_bsk)
        acc += tl.dot(a, b.to(tl.float32), out_dtype=tl.float32) * b_scale

    out = acc * token_weights[:, None]
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 1}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 2}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 1}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 2}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 4}, num_warps=8, num_stages=2),
    ],
    key=['MAX_PID_M', 'N', 'K', 'BLOCK_M'],
    restore_value=['C_ptr'],
)
@triton.jit
def _medium_fused_moe_gemm1_swiglu_kernel(
    A_ptr,
    A_scale_ptr,
    B_ptr,
    C_ptr,
    B_scale_ptr,
    token_ids_ptr,
    block_offsets_ptr,
    total_blocks_ptr,
    MAX_PID_M: tl.constexpr, T: tl.constexpr, H: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_at, stride_ah,
    stride_as0, stride_as1,
    stride_be, stride_bn, stride_bh,
    stride_cm, stride_cn,
    stride_bse, stride_bsn, stride_bsh,
    E_LOCAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    N_OUT = N // 2
    num_pid_n = tl.cdiv(N_OUT, BLOCK_N)

    total_blocks = tl.load(total_blocks_ptr)
    if total_blocks <= 0:
        return
    if pid >= total_blocks * num_pid_n:
        return
    num_pid_m = total_blocks

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    e_idx = tl.arange(0, E_LOCAL)
    b_start = tl.load(block_offsets_ptr + e_idx)
    b_end = tl.load(block_offsets_ptr + e_idx + 1)
    valid = (b_start <= pid_m) & (pid_m < b_end)
    expert_id = tl.argmax(valid.to(tl.int32), axis=0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rn_1 = rn
    rn_3 = rn + N_OUT
    token_idx = tl.load(token_ids_ptr + rm)
    m_mask = token_idx < T
    safe_token_idx = tl.where(m_mask, token_idx, 0)

    acc_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_base = A_ptr + safe_token_idx[:, None] * stride_at
    a_scale_base = A_scale_ptr + safe_token_idx * stride_as1
    b_base_1 = B_ptr + expert_id * stride_be + rn_1[None, :] * stride_bn
    b_base_3 = B_ptr + expert_id * stride_be + rn_3[None, :] * stride_bn
    b_scale_base_1 = B_scale_ptr + expert_id * stride_bse + (rn_1[None, :] // 128) * stride_bsn
    b_scale_base_3 = B_scale_ptr + expert_id * stride_bse + (rn_3[None, :] // 128) * stride_bsn

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_base + rk[None, :] * stride_ah, mask=m_mask[:, None], other=0.0)
        a_scale = tl.load(a_scale_base + (k // 128) * stride_as0, mask=m_mask, other=0.0)
        b_1 = tl.load(b_base_1 + rk[:, None] * stride_bh)
        b_3 = tl.load(b_base_3 + rk[:, None] * stride_bh)
        b_scale_1 = tl.load(b_scale_base_1 + (k // 128) * stride_bsh)
        b_scale_3 = tl.load(b_scale_base_3 + (k // 128) * stride_bsh)
        partial_1 = tl.dot(a, b_1, out_dtype=tl.float32)
        partial_3 = tl.dot(a, b_3, out_dtype=tl.float32)
        acc_1 += partial_1 * (a_scale[:, None] * b_scale_1)
        acc_3 += partial_3 * (a_scale[:, None] * b_scale_3)

    swiglu_out = (acc_3 * (1.0 / (1.0 + tl.exp(-acc_3)))) * acc_1
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, swiglu_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 1}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 2}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 1}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 4}, num_warps=8, num_stages=3),
    ],
    key=['MAX_PID_M', 'N', 'K', 'BLOCK_M'],
    restore_value=['C_ptr'],
)
@triton.jit
def _medium_fused_moe_gemm2_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    B_scale_ptr,
    token_weights_ptr,
    block_offsets_ptr,
    total_blocks_ptr,
    MAX_PID_M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_bse, stride_bsn, stride_bsk,
    E_LOCAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    total_blocks = tl.load(total_blocks_ptr)
    if total_blocks <= 0:
        return
    if pid >= total_blocks * num_pid_n:
        return
    num_pid_m = total_blocks

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    e_idx = tl.arange(0, E_LOCAL)
    b_start = tl.load(block_offsets_ptr + e_idx)
    b_end = tl.load(block_offsets_ptr + e_idx + 1)
    valid = (b_start <= pid_m) & (pid_m < b_end)
    expert_id = tl.argmax(valid.to(tl.int32), axis=0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    token_weights = tl.load(token_weights_ptr + rm)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_base = A_ptr + rm[:, None] * stride_am
    b_base = B_ptr + expert_id * stride_be + rn[None, :] * stride_bn
    b_scale_base = B_scale_ptr + expert_id * stride_bse + (rn[None, :] // 128) * stride_bsn

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_base + rk[None, :] * stride_ak)
        b = tl.load(b_base + rk[:, None] * stride_bk)
        b_scale = tl.load(b_scale_base + (k // 128) * stride_bsk)
        acc += tl.dot(a, b.to(tl.float32), out_dtype=tl.float32) * b_scale

    out = acc * token_weights[:, None]
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=4),
    ],
    key=['N', 'K'],
)
@triton.jit
def _t1_fused_gemm2_reduce_kernel(
    A_ptr,
    B_ptr,
    B_scale_ptr,
    output_ptr,
    token_weights_ptr,
    scatter_map_ptr,
    local_expert_ids_ptr,
    num_local_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    TOP_K: tl.constexpr,
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_bse, stride_bsn, stride_bsk,
    stride_on,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rn = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    num_local = tl.load(num_local_ptr)
    acc_out = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for slot in tl.static_range(TOP_K):
        if slot < num_local:
            safe_pos = tl.load(scatter_map_ptr + slot)
            safe_expert_id = tl.load(local_expert_ids_ptr + slot)
            token_weight = tl.load(token_weights_ptr + safe_pos)
            slot_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            a_base = A_ptr + safe_pos * stride_am
            b_base = B_ptr + safe_expert_id * stride_be + rn[None, :] * stride_bn
            b_scale_base = B_scale_ptr + safe_expert_id * stride_bse + (rn // 128) * stride_bsn

            for k in range(0, K, BLOCK_K):
                rk = k + tl.arange(0, BLOCK_K)
                a = tl.load(a_base + rk * stride_ak)
                b = tl.load(b_base + rk[:, None] * stride_bk)
                b_scale = tl.load(b_scale_base + (k // 128) * stride_bsk)
                partial = tl.sum(a[:, None] * b.to(tl.float32), axis=0)
                slot_acc += partial * b_scale

            acc_out += slot_acc * token_weight

    o_ptrs = output_ptr + rn * stride_on
    tl.store(o_ptrs, acc_out.to(tl.bfloat16))


# ═══════════════════════════════════════════════════════════════
# Triton Kernel 3: Token-Centric Reduce (zero-free, atomic-free, bf16 fused)
# For each output token, sums its TOP_K expert contributions and writes bf16.
# ═══════════════════════════════════════════════════════════════

@triton.jit
def _token_reduce_kernel(
    expert_out_ptr,   # [num_padded, N] fp32
    output_ptr,       # [T, N] bf16 (final output)
    scatter_map_ptr,  # [T * TOP_K] int32
    T_val: tl.constexpr,
    N: tl.constexpr,
    TOP_K: tl.constexpr,
    stride_em, stride_en,
    stride_ot, stride_on,
    BLOCK_N: tl.constexpr,
):
    """
    Grid: T_val * cdiv(N, BLOCK_N)
    Each program handles one token's BLOCK_N columns.
    Iterates over TOP_K expert slots, sums contributions, stores bf16.
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    
    pid_t = pid // num_pid_n
    pid_n = pid % num_pid_n
    
    if pid_t >= T_val:
        return
    
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = rn < N
    
    # Accumulate over TOP_K expert contributions
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    base_idx = pid_t * TOP_K
    for k in tl.static_range(TOP_K):
        pos = tl.load(scatter_map_ptr + base_idx + k)
        if pos >= 0:
            e_ptrs = expert_out_ptr + pos * stride_em + rn * stride_en
            vals = tl.load(e_ptrs, mask=n_mask, other=0.0)
            acc += vals
    
    # Store directly as bf16
    o_ptrs = output_ptr + pid_t * stride_ot + rn * stride_on
    tl.store(o_ptrs, acc.to(tl.bfloat16), mask=n_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_M': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 32}, num_warps=8, num_stages=3),
    ],
    key=['MAX_PID_M', 'N', 'K', 'BLOCK_M'],
    restore_value=['C_ptr'],
)
@triton.jit
def _t1_fused_gemm1_swiglu_kernel(
    A_ptr,
    A_scale_ptr,
    B_ptr,
    C_ptr,
    B_scale_ptr,
    token_ids_ptr,
    block_offsets_ptr,
    total_blocks_ptr,
    MAX_PID_M: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_at, stride_ah,
    stride_as0, stride_as1,
    stride_be, stride_bn, stride_bh,
    stride_cm, stride_cn,
    stride_bse, stride_bsn, stride_bsh,
    E_LOCAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    N_OUT = N // 2
    num_pid_n = tl.cdiv(N_OUT, BLOCK_N)
    total_blocks = tl.load(total_blocks_ptr)
    if total_blocks <= 0:
        return
    if pid >= total_blocks * num_pid_n:
        return
    num_pid_m = total_blocks
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    e_idx = tl.arange(0, E_LOCAL)
    b_start = tl.load(block_offsets_ptr + e_idx)
    b_end = tl.load(block_offsets_ptr + e_idx + 1)
    valid = (b_start <= pid_m) & (pid_m < b_end)
    expert_id = tl.argmax(valid.to(tl.int32), axis=0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rn_1 = rn
    rn_3 = rn + N_OUT
    token_idx = tl.load(token_ids_ptr + rm)
    m_mask = token_idx < T

    acc_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    safe_token_idx = tl.where(token_idx < T, token_idx, 0)
    a_base = A_ptr + safe_token_idx[:, None] * stride_at
    a_scale_base = A_scale_ptr + safe_token_idx * stride_as1
    b_base_1 = B_ptr + expert_id * stride_be + rn_1[None, :] * stride_bn
    b_base_3 = B_ptr + expert_id * stride_be + rn_3[None, :] * stride_bn
    b_scale_base_1 = B_scale_ptr + expert_id * stride_bse + (rn_1[None, :] // 128) * stride_bsn
    b_scale_base_3 = B_scale_ptr + expert_id * stride_bse + (rn_3[None, :] // 128) * stride_bsn

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_base + rk[None, :] * stride_ah, mask=m_mask[:, None], other=0.0)
        a_scale = tl.load(a_scale_base + (k // 128) * stride_as0, mask=m_mask, other=0.0)
        b_1 = tl.load(b_base_1 + rk[:, None] * stride_bh)
        b_3 = tl.load(b_base_3 + rk[:, None] * stride_bh)
        b_scale_1 = tl.load(b_scale_base_1 + (k // 128) * stride_bsh)
        b_scale_3 = tl.load(b_scale_base_3 + (k // 128) * stride_bsh)
        partial_1 = tl.dot(a, b_1, out_dtype=tl.float32)
        partial_3 = tl.dot(a, b_3, out_dtype=tl.float32)
        acc_1 += partial_1 * (a_scale[:, None] * b_scale_1)
        acc_3 += partial_3 * (a_scale[:, None] * b_scale_3)

    swiglu_out = (acc_3 * (1.0 / (1.0 + tl.exp(-acc_3)))) * acc_1
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, swiglu_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256, 'GROUP_M': 1}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'GROUP_M': 1}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'GROUP_M': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'GROUP_M': 4}, num_warps=8, num_stages=3),
    ],
    key=['MAX_PID_M', 'N', 'K', 'BLOCK_M'],
)
@triton.jit
def _t1_generic_gemm1_swiglu_kernel(
    A_ptr,
    A_scale_ptr,
    B_ptr,
    C_ptr,
    B_scale_ptr,
    token_ids_ptr,
    block_offsets_ptr,
    total_blocks_ptr,
    MAX_PID_M: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_at, stride_ah,
    stride_as0, stride_as1,
    stride_be, stride_bn, stride_bh,
    stride_cm, stride_cn,
    stride_bse, stride_bsn, stride_bsh,
    E_LOCAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    N_OUT = N // 2
    num_pid_n = tl.cdiv(N_OUT, BLOCK_N)
    total_blocks = tl.load(total_blocks_ptr)
    if total_blocks <= 0:
        return
    if pid >= total_blocks * num_pid_n:
        return
    num_pid_m = total_blocks
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    e_idx = tl.arange(0, E_LOCAL)
    b_start = tl.load(block_offsets_ptr + e_idx)
    b_end = tl.load(block_offsets_ptr + e_idx + 1)
    valid = (b_start <= pid_m) & (pid_m < b_end)
    expert_id = tl.argmax(valid.to(tl.int32), axis=0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rn_1 = rn
    rn_3 = rn + N_OUT
    token_idx = tl.load(token_ids_ptr + rm)
    m_mask = token_idx < T
    safe_token_idx = tl.where(m_mask, token_idx, 0)

    acc_1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_base = A_ptr + safe_token_idx[:, None] * stride_at
    a_scale_base = A_scale_ptr + safe_token_idx * stride_as1
    b_base_1 = B_ptr + expert_id * stride_be + rn_1[None, :] * stride_bn
    b_base_3 = B_ptr + expert_id * stride_be + rn_3[None, :] * stride_bn
    b_scale_base_1 = B_scale_ptr + expert_id * stride_bse + (rn_1[None, :] // 128) * stride_bsn
    b_scale_base_3 = B_scale_ptr + expert_id * stride_bse + (rn_3[None, :] // 128) * stride_bsn

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_base + rk[None, :] * stride_ah, mask=m_mask[:, None], other=0.0)
        a_scale = tl.load(a_scale_base + (k // 128) * stride_as0, mask=m_mask, other=0.0)
        b_1 = tl.load(b_base_1 + rk[:, None] * stride_bh)
        b_3 = tl.load(b_base_3 + rk[:, None] * stride_bh)
        b_scale_1 = tl.load(b_scale_base_1 + (k // 128) * stride_bsh)
        b_scale_3 = tl.load(b_scale_base_3 + (k // 128) * stride_bsh)
        partial_1 = tl.dot(a, b_1, out_dtype=tl.float32)
        partial_3 = tl.dot(a, b_3, out_dtype=tl.float32)
        acc_1 += partial_1 * (a_scale[:, None] * b_scale_1)
        acc_3 += partial_3 * (a_scale[:, None] * b_scale_3)

    swiglu_out = (acc_3 * (1.0 / (1.0 + tl.exp(-acc_3)))) * acc_1
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, swiglu_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'GROUP_M': 1}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'GROUP_M': 2}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'GROUP_M': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'GROUP_M': 1}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'GROUP_M': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'GROUP_M': 4}, num_warps=8, num_stages=3),
    ],
    key=['MAX_PID_M', 'N', 'K', 'BLOCK_M'],
)
@triton.jit
def _t1_generic_gemm2_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    B_scale_ptr,
    token_weights_ptr,
    block_offsets_ptr,
    total_blocks_ptr,
    MAX_PID_M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am, stride_ak,
    stride_be, stride_bn, stride_bk,
    stride_cm, stride_cn,
    stride_bse, stride_bsn, stride_bsk,
    E_LOCAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    total_blocks = tl.load(total_blocks_ptr)
    if total_blocks <= 0:
        return
    if pid >= total_blocks * num_pid_n:
        return
    num_pid_m = total_blocks
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    e_idx = tl.arange(0, E_LOCAL)
    b_start = tl.load(block_offsets_ptr + e_idx)
    b_end = tl.load(block_offsets_ptr + e_idx + 1)
    valid = (b_start <= pid_m) & (pid_m < b_end)
    expert_id = tl.argmax(valid.to(tl.int32), axis=0)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    token_weights = tl.load(token_weights_ptr + rm)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_base = A_ptr + rm[:, None] * stride_am
    b_base = B_ptr + expert_id * stride_be + rn[None, :] * stride_bn
    b_scale_base = B_scale_ptr + expert_id * stride_bse + (rn[None, :] // 128) * stride_bsn

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(a_base + rk[None, :] * stride_ak)
        b = tl.load(b_base + rk[:, None] * stride_bk)
        b_scale = tl.load(b_scale_base + (k // 128) * stride_bsk)
        acc += tl.dot(a, b.to(tl.float32), out_dtype=tl.float32) * b_scale

    out = acc * token_weights[:, None]
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, out)

# ═══════════════════════════════════════════════════════════════
# Pre-allocated buffer cache — reuse output_fp32/Intermediate
# ═══════════════════════════════════════════════════════════════

_buf_cache = {}
_routing_cache = {}
_sort_cache = {}
_t1_cache = {}


def _select_block_m(num_topk_tokens: int) -> int:
    if num_topk_tokens <= TINY_BATCH_TOPK_TOKENS:
        return BLOCK_M_TINY
    if num_topk_tokens <= SMALL_BATCH_TOPK_TOKENS:
        return BLOCK_M_SMALL
    if num_topk_tokens > LARGE_BATCH_TOPK_TOKENS:
        return BLOCK_M_XLARGE
    return BLOCK_M_LARGE


def _select_block_m_from_histogram(topk_idx: torch.Tensor, local_start: int) -> tuple[int, int]:
    num_topk_tokens = int(topk_idx.numel())
    fallback_block_m = _select_block_m(num_topk_tokens)
    if num_topk_tokens > HISTOGRAM_BLOCK_M_MAX_TOPK_TOKENS:
        return fallback_block_m, num_topk_tokens + E_LOCAL * fallback_block_m

    local_mask = (topk_idx >= local_start) & (topk_idx < local_start + E_LOCAL)
    local_ids = (topk_idx[local_mask] - local_start).to(torch.int64)
    if local_ids.numel() == 0:
        return fallback_block_m, 0

    counts = torch.bincount(local_ids, minlength=E_LOCAL)
    best_block_m = fallback_block_m
    best_padded = None
    for candidate in BLOCK_M_CANDIDATES:
        padded = int((((counts + (candidate - 1)) // candidate) * candidate).sum().item())
        if best_padded is None or padded < best_padded or (padded == best_padded and candidate < best_block_m):
            best_block_m = candidate
            best_padded = padded
    return best_block_m, int(best_padded)


def _use_small_medium_gemm_bucket(t: int) -> bool:
    return SMALL_MEDIUM_T_MIN <= t <= SMALL_MEDIUM_T_MAX


def _use_upper_medium_gemm_bucket(t: int) -> bool:
    return UPPER_MEDIUM_T_MIN <= t <= UPPER_MEDIUM_T_MAX


_host_scalar_cache = {}


def _cached_host_scalar(x, cast_fn):
    if torch.is_tensor(x):
        key = (x.data_ptr(), str(x.device), str(x.dtype), cast_fn.__name__)
        if key in _host_scalar_cache:
            return _host_scalar_cache[key]
        v = cast_fn(x.item())
        _host_scalar_cache[key] = v
        return v
    return cast_fn(x)


def _kernel_t1(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_start,
    scale_factor,
    output,
):
    device = hidden_states.device
    if device not in _t1_cache:
        local_expert_ids = torch.empty((TOP_K,), dtype=torch.int32, device=device)
        local_expert_wts = torch.empty((TOP_K,), dtype=torch.float32, device=device)
        num_local = torch.empty((1,), dtype=torch.int32, device=device)
        sorted_token_ids = torch.empty((T1_GENERIC_MAX_PADDED,), dtype=torch.int64, device=device)
        sorted_weights = torch.empty((T1_GENERIC_MAX_PADDED,), dtype=torch.float32, device=device)
        scatter_map = torch.empty((TOP_K,), dtype=torch.int32, device=device)
        block_offsets = torch.empty((E_LOCAL + 1,), dtype=torch.int32, device=device)
        total_blocks = torch.empty((1,), dtype=torch.int32, device=device)
        intermediate = torch.empty((T1_GENERIC_MAX_PADDED, I_SIZE), dtype=torch.float32, device=device)
        expert_out = torch.empty((T1_GENERIC_MAX_PADDED, H), dtype=torch.float32, device=device)
        _t1_cache[device] = (
            local_expert_ids, local_expert_wts, num_local,
            sorted_token_ids, sorted_weights, scatter_map,
            block_offsets, total_blocks, intermediate, expert_out,
        )

    (
        local_expert_ids, local_expert_wts, num_local,
        sorted_token_ids, sorted_weights, scatter_map,
        block_offsets, total_blocks, intermediate, expert_out,
    ) = _t1_cache[device]

    ds_routing_t1_local(
        routing_logits,
        routing_bias,
        scale_factor,
        local_start,
        local_expert_ids,
        local_expert_wts,
        num_local,
        sorted_token_ids,
        sorted_weights,
        scatter_map,
        block_offsets,
        total_blocks,
    )

    max_pid_m = T1_GENERIC_MAX_PADDED // T1_GENERIC_BLOCK_M
    grid1 = lambda META: (max_pid_m * triton.cdiv(I_SIZE, META['BLOCK_N']),)
    _t1_fused_gemm1_swiglu_kernel[grid1](
        A_ptr=hidden_states,
        A_scale_ptr=hidden_states_scale,
        B_ptr=gemm1_weights,
        C_ptr=intermediate,
        B_scale_ptr=gemm1_weights_scale,
        token_ids_ptr=sorted_token_ids,
        block_offsets_ptr=block_offsets,
        total_blocks_ptr=total_blocks,
        MAX_PID_M=max_pid_m,
        T=1, H=H, N=4096, K=H,
        stride_at=hidden_states.stride(0), stride_ah=hidden_states.stride(1),
        stride_as0=hidden_states_scale.stride(0), stride_as1=hidden_states_scale.stride(1),
        stride_be=gemm1_weights.stride(0), stride_bn=gemm1_weights.stride(1), stride_bh=gemm1_weights.stride(2),
        stride_cm=intermediate.stride(0), stride_cn=intermediate.stride(1),
        stride_bse=gemm1_weights_scale.stride(0), stride_bsn=gemm1_weights_scale.stride(1), stride_bsh=gemm1_weights_scale.stride(2),
        E_LOCAL=E_LOCAL,
        BLOCK_M=T1_GENERIC_BLOCK_M,
    )

    grid2 = lambda META: (triton.cdiv(H, META['BLOCK_N']),)
    _t1_fused_gemm2_reduce_kernel[grid2](
        A_ptr=intermediate,
        B_ptr=gemm2_weights,
        B_scale_ptr=gemm2_weights_scale,
        output_ptr=output,
        token_weights_ptr=sorted_weights,
        scatter_map_ptr=scatter_map,
        local_expert_ids_ptr=local_expert_ids,
        num_local_ptr=num_local,
        N=H,
        K=I_SIZE,
        TOP_K=TOP_K,
        stride_am=intermediate.stride(0),
        stride_ak=intermediate.stride(1),
        stride_be=gemm2_weights.stride(0),
        stride_bn=gemm2_weights.stride(1),
        stride_bk=gemm2_weights.stride(2),
        stride_bse=gemm2_weights_scale.stride(0),
        stride_bsn=gemm2_weights_scale.stride(1),
        stride_bsk=gemm2_weights_scale.stride(2),
        stride_on=output.stride(1),
        BLOCK_K=BLOCK_K,
    )
    return output

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
    output=None,              # [T, 7168] bfloat16 (optional destination-passing)
):
    T = routing_logits.shape[0]
    device = hidden_states.device
    local_start = _cached_host_scalar(local_expert_offset, int)
    scale_factor = _cached_host_scalar(routed_scaling_factor, float)

    if output is None:
        output = torch.empty((T, 7168), dtype=torch.bfloat16, device=device)

    if T == 1:
        return _kernel_t1(
            routing_logits,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            local_start,
            scale_factor,
            output,
        )

    rkey = T
    if rkey in _routing_cache:
        topk_idx_ws, topk_wts_ws = _routing_cache[rkey]
    else:
        topk_idx_ws = torch.empty((T, TOP_K), dtype=torch.int32, device=device)
        topk_wts_ws = torch.empty((T, TOP_K), dtype=torch.float32, device=device)
        _routing_cache[rkey] = (topk_idx_ws, topk_wts_ws)

    ds_routing(
        routing_logits,
        routing_bias,
        scale_factor,
        topk_idx=topk_idx_ws,
        topk_weights=topk_wts_ws,
    )

    block_m = _select_block_m(T * TOP_K)
    MAX_PADDED = T * TOP_K + E_LOCAL * block_m
    MAX_PID_M = MAX_PADDED // block_m
    bkey = (T, block_m)

    if bkey in _buf_cache:
        Intermediate, expert_out = _buf_cache[bkey]
    else:
        Intermediate = torch.empty((MAX_PADDED, 2048), dtype=torch.float32, device=device)
        expert_out = torch.empty((MAX_PADDED, 7168), dtype=torch.float32, device=device)
        _buf_cache[bkey] = (Intermediate, expert_out)

    if bkey in _sort_cache:
        sorted_token_ids, sorted_weights, scatter_map, block_offsets, total_blocks, counts_workspace, partial_counts, tile_offsets = _sort_cache[bkey]
    else:
        num_tiles = triton.cdiv(T * TOP_K, SORT_BLOCK_ITEMS)
        sorted_token_ids = torch.empty((MAX_PADDED,), dtype=torch.int64, device=device)
        sorted_weights = torch.empty((MAX_PADDED,), dtype=torch.float32, device=device)
        scatter_map = torch.empty((T * TOP_K,), dtype=torch.int32, device=device)
        block_offsets = torch.empty((E_LOCAL + 1,), dtype=torch.int32, device=device)
        total_blocks = torch.empty((1,), dtype=torch.int32, device=device)
        counts_workspace = torch.empty((E_LOCAL,), dtype=torch.int32, device=device)
        partial_counts = torch.empty((num_tiles, E_LOCAL), dtype=torch.int32, device=device)
        tile_offsets = torch.empty((num_tiles, E_LOCAL), dtype=torch.int32, device=device)
        _sort_cache[bkey] = (
            sorted_token_ids, sorted_weights, scatter_map,
            block_offsets, total_blocks, counts_workspace, partial_counts, tile_offsets,
        )
    parallel_sort_and_scatter(
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
        block_m,
        MAX_PADDED,
        counts_workspace,
    )

    use_small_medium_gemm = _use_small_medium_gemm_bucket(T)
    use_upper_medium_gemm = _use_upper_medium_gemm_bucket(T)

    # -- 4. Fused GEMM1 + SwiGLU --
    grid1 = lambda META: (MAX_PID_M * triton.cdiv(2048, META['BLOCK_N']),)

    if use_small_medium_gemm:
        gemm1_kernel = _small_medium_fused_moe_gemm1_swiglu_kernel
    elif use_upper_medium_gemm:
        gemm1_kernel = _medium_fused_moe_gemm1_swiglu_kernel
    else:
        gemm1_kernel = _fused_moe_gemm1_swiglu_kernel
    gemm1_kernel[grid1](
        A_ptr=hidden_states,
        A_scale_ptr=hidden_states_scale,
        B_ptr=gemm1_weights,
        C_ptr=Intermediate,
        B_scale_ptr=gemm1_weights_scale,
        token_ids_ptr=sorted_token_ids,
        block_offsets_ptr=block_offsets,
        total_blocks_ptr=total_blocks,
        MAX_PID_M=MAX_PID_M, T=T, H=7168, N=4096, K=7168,
        stride_at=hidden_states.stride(0), stride_ah=hidden_states.stride(1),
        stride_as0=hidden_states_scale.stride(0), stride_as1=hidden_states_scale.stride(1),
        stride_be=gemm1_weights.stride(0), stride_bn=gemm1_weights.stride(1), stride_bh=gemm1_weights.stride(2),
        stride_cm=Intermediate.stride(0), stride_cn=Intermediate.stride(1),
        stride_bse=gemm1_weights_scale.stride(0), stride_bsn=gemm1_weights_scale.stride(1), stride_bsh=gemm1_weights_scale.stride(2),
        E_LOCAL=E_LOCAL,
        BLOCK_M=block_m,
    )

    # -- 5. GEMM2 (non-atomic, writes to expert_out) --
    grid2 = lambda META: (MAX_PID_M * triton.cdiv(7168, META['BLOCK_N']),)

    if use_small_medium_gemm:
        gemm2_kernel = _small_medium_fused_moe_gemm2_kernel
    elif use_upper_medium_gemm:
        gemm2_kernel = _medium_fused_moe_gemm2_kernel
    else:
        gemm2_kernel = _fused_moe_gemm2_kernel
    gemm2_kernel[grid2](
        A_ptr=Intermediate,
        B_ptr=gemm2_weights,
        C_ptr=expert_out,
        B_scale_ptr=gemm2_weights_scale,
        token_weights_ptr=sorted_weights,
        block_offsets_ptr=block_offsets,
        total_blocks_ptr=total_blocks,
        MAX_PID_M=MAX_PID_M, N=7168, K=2048,
        stride_am=Intermediate.stride(0), stride_ak=Intermediate.stride(1),
        stride_be=gemm2_weights.stride(0), stride_bn=gemm2_weights.stride(1), stride_bk=gemm2_weights.stride(2),
        stride_cm=expert_out.stride(0), stride_cn=expert_out.stride(1),
        stride_bse=gemm2_weights_scale.stride(0), stride_bsn=gemm2_weights_scale.stride(1), stride_bsk=gemm2_weights_scale.stride(2),
        E_LOCAL=E_LOCAL,
        BLOCK_M=block_m,
    )

    # -- 6. Token-Centric Reduce (zero-free, atomic-free, bf16 fused) --
    RS_BLOCK_N = 256
    num_n_blocks = triton.cdiv(7168, RS_BLOCK_N)
    grid3 = (T * num_n_blocks,)

    _token_reduce_kernel[grid3](
        expert_out_ptr=expert_out,
        output_ptr=output,
        scatter_map_ptr=scatter_map,
        T_val=T,
        N=7168,
        TOP_K=TOP_K,
        stride_em=expert_out.stride(0), stride_en=expert_out.stride(1),
        stride_ot=output.stride(0), stride_on=output.stride(1),
        BLOCK_N=RS_BLOCK_N,
        num_warps=4,
    )

    return output
