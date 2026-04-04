import torch
import triton
import triton.language as tl
import os
import sys

# =====================================================================
# DEBUG PHASE SYSTEM — Binary-search GPU segfault isolation
# Phase 1: Basic setup and parsing
# Phase 2: Workspace allocation
# Phase 3: TMA descriptor creation
# Phase 4: CuTe kernel object initialization
# Phase 5: + JIT compilation
# Phase 6: Full kernel launch
# =====================================================================
_CUTE_DEBUG_PHASE = float(os.environ.get("CUTE_DEBUG_PHASE", "6"))

# =====================================================================
# LAZY INIT: defer ALL cutlass imports and CUDA operations to first call
# NOTHING happens at module import time except setting flags.
# =====================================================================
_USE_CUTE = False
_CUTE_KERNEL = None
_CUTE_STREAM = None
_CUTE_MAX_CLUSTERS = None
_CUTE_NUM_SMS = None
_CUTE_COMPILED_CACHE = {}

try:
    import cutlass
    import cutlass.cute as cute
    import cutlass.torch as cutlass_torch
    import cuda.bindings.driver as cuda_driver
    import cutlass.utils as utils
    
    try:
        from cute_kernel import Sm100GroupedSwiGLUBlockscaledKernel
    except ImportError:
        from .cute_kernel import Sm100GroupedSwiGLUBlockscaledKernel

    if torch.cuda.is_available():
        sf_vec = 32
        tiler = (128, 128)
        cluster = (1, 1)
        _CUTE_KERNEL = Sm100GroupedSwiGLUBlockscaledKernel(sf_vec, tiler, cluster)
        
        torch_stream = torch.cuda.current_stream()
        _CUTE_STREAM = cuda_driver.CUstream(torch_stream.cuda_stream)
        
        _CUTE_MAX_CLUSTERS = int(utils.HardwareInfo().get_max_active_clusters(1))
        _CUTE_NUM_SMS = _CUTE_MAX_CLUSTERS * cluster[0] * cluster[1]
        _CUTE_COMPILED_CACHE = {}
        
        _USE_CUTE = True
    else:
        raise RuntimeError("torch.cuda.is_available() returned False during module load")
except Exception as e:
    import traceback
    err_msg = f"CuTe module load FAILED: {e}\n{traceback.format_exc()}"
    print(err_msg, flush=True)
    raise RuntimeError(err_msg)

_GATHER_A_BUF = None
_GATHER_A_SCALE_BUF = None
_GATHER_W13_SCALE_CACHE = {}

# ---- Cached dispatch buffers (avoid per-call torch.empty) ----
_DISPATCH_BUF_CACHE = {}  # key: num_g -> {ptrs, strides, shapes, tensormaps}

def _get_dispatch_bufs(num_g, H, KBLKS_32, N_DIM):
    """Get or create cached dispatch buffers. Strides are pre-filled (constant per model)."""
    global _DISPATCH_BUF_CACHE
    if num_g in _DISPATCH_BUF_CACHE:
        bufs = _DISPATCH_BUF_CACHE[num_g]
        # shapes columns 1-3 are constant, only column 0 (M) changes per call
        return bufs
    
    ptrs = torch.empty((num_g, 7), dtype=torch.int64, device='cuda')
    strides = torch.empty((num_g, 7, 2), dtype=torch.int32, device='cuda')
    shapes = torch.empty((num_g, 4), dtype=torch.int32, device='cuda')
    tensormaps = torch.empty((_CUTE_NUM_SMS, 7, 16), dtype=torch.int64, device='cuda')
    
    # Pre-fill constant strides (same for every call — H, KBLKS_32, N_DIM never change)
    for g in range(num_g):
        strides[g, 0, 0] = H;        strides[g, 0, 1] = 1  # A
        strides[g, 1, 0] = H;        strides[g, 1, 1] = 1  # B_w1
        strides[g, 2, 0] = H;        strides[g, 2, 1] = 1  # B_w3
        strides[g, 3, 0] = KBLKS_32; strides[g, 3, 1] = 1  # SFA
        strides[g, 4, 0] = KBLKS_32; strides[g, 4, 1] = 1  # SFB_w1
        strides[g, 5, 0] = KBLKS_32; strides[g, 5, 1] = 1  # SFB_w3
        strides[g, 6, 0] = N_DIM;    strides[g, 6, 1] = 1  # C
    # Pre-fill constant shape columns
    shapes[:, 1] = N_DIM
    shapes[:, 2] = H
    shapes[:, 3] = 1
    
    bufs = {'ptrs': ptrs, 'strides': strides, 'shapes': shapes, 'tensormaps': tensormaps}
    _DISPATCH_BUF_CACHE[num_g] = bufs
    return bufs


# ---- Cached CuTe metadata wrappers (avoid per-call convert_cute_tensor) ----
_META_WRAPPER_CACHE = {}  # key: (id(shapes), id(ptrs), id(strides), id(tensormaps))

def _get_cute_meta_wrappers(shapes, ptrs, strides, tensormaps):
    """Get or create cached CuTe tensor wrappers for metadata tensors."""
    global _META_WRAPPER_CACHE
    cache_key = (id(shapes), id(ptrs), id(strides), id(tensormaps))
    if cache_key in _META_WRAPPER_CACHE:
        return _META_WRAPPER_CACHE[cache_key]
    
    p_shape = cutlass_torch.convert_cute_tensor(shapes, cutlass_torch.cute_tensor_like(shapes, cutlass.Int32, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Int32, is_dynamic_layout=False)
    p_ptrs = cutlass_torch.convert_cute_tensor(ptrs, cutlass_torch.cute_tensor_like(ptrs, cutlass.Int64, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Int64, is_dynamic_layout=False)
    p_strides = cutlass_torch.convert_cute_tensor(strides, cutlass_torch.cute_tensor_like(strides, cutlass.Int32, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Int32, is_dynamic_layout=False)
    p_tmaps = cutlass_torch.convert_cute_tensor(tensormaps, cutlass_torch.cute_tensor_like(tensormaps, cutlass.Int64, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Int64, is_dynamic_layout=False)
    
    result = (p_shape, p_ptrs, p_strides, p_tmaps)
    _META_WRAPPER_CACHE[cache_key] = result
    return result

@triton.jit
def tma_gather_a_explicit(
    src_a_ptr, src_sa_ptr,
    dst_a_ptr, dst_sa_ptr,
    token_ids_ptr,
    T_PAD, T_TOTAL, H: tl.constexpr, KBLKS_32: tl.constexpr, BLOCK_M: tl.constexpr
):
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = rm < T_PAD
    idx = tl.load(token_ids_ptr + rm, mask=mask, other=0)
    
    valid_idx_mask = mask & (idx < T_TOTAL) & (idx >= 0)
    safe_idx = tl.where(valid_idx_mask, idx, 0)
    
    # Gather A matrix
    for h in range(0, H, 128):
        rh = h + tl.arange(0, 128)
        src_ptrs = src_a_ptr + safe_idx[:, None] * H + rh[None, :]
        dst_ptrs = dst_a_ptr + rm[:, None] * H + rh[None, :]
        val = tl.load(src_ptrs, mask=valid_idx_mask[:, None], other=0.0)
        tl.store(dst_ptrs, val, mask=mask[:, None])
        
    # Gather A_scale: convert FP32 -> e8m0
    for k in range(0, KBLKS_32, 32):
        rk_32 = k + tl.arange(0, 32)
        rk_128 = rk_32 // 4
        
        sa_mask = valid_idx_mask[:, None] & (rk_32[None, :] < KBLKS_32)
        dst_mask = mask[:, None] & (rk_32[None, :] < KBLKS_32)
        
        src_ptrs = src_sa_ptr + rk_128[None, :] * T_TOTAL + safe_idx[:, None]
        dst_ptrs = dst_sa_ptr + rm[:, None] * KBLKS_32 + rk_32[None, :]
        
        fp32_val = tl.load(src_ptrs, mask=sa_mask, other=1.0)
        
        fp_safe = tl.where(fp32_val > 1e-15, fp32_val, 1e-15)
        log2_val = tl.math.log2(fp_safe)
        exp_int = (log2_val + tl.where(log2_val < 0.0, -0.5, 0.5)).to(tl.int32)
        e8_byte = (exp_int + 127).to(tl.uint8)
        e8_byte = tl.where(sa_mask, e8_byte, 127)
        
        tl.store(dst_ptrs, e8_byte, mask=dst_mask)

def run_cute_path(
    hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale,
    sorted_token_ids, block_offsets, total_blocks,
    intermediate, block_m=128
):
    """
    Launch CuTe-based FP8 Grouped SwiGLU kernel.
    """
    if not _USE_CUTE:
        return intermediate
        
    if block_m < 128:
        # CuTe's 128-row MMA tiler is a hard requirement from tcgen05. Sub-128 M causes ILLEGAL_ADDRESS.
        return None
        
    global _GATHER_A_BUF, _GATHER_A_SCALE_BUF, _GATHER_W13_SCALE_E8
    
    # ---- EARLY EXIT: check num_g BEFORE any GPU work ----
    total_blks_host = int(total_blocks.item())
    T_PAD = total_blks_host * block_m
    T_TOTAL, H = hidden_states.shape
    N_DIM = gemm1_weights.shape[1] // 2
    
    if T_PAD == 0:
        return intermediate
    
    # Check expert count early — avoid wasted gather/alloc for fallback workloads
    e_starts = block_offsets.cpu().numpy()
    num_experts = len(e_starts) - 1
    num_g = sum(1 for e in range(num_experts) if (e_starts[e+1] - e_starts[e]) > 0)
    
    if num_g > 16:
        # 32-expert workloads crash CuTe (structural ILLEGAL_ADDRESS).
        # Return None BEFORE any GPU work to avoid penalizing Triton fallback.
        return None
    
    KBLKS_32 = H // 32
    
    # ---- Gather A into contiguous buffer ----
    if _GATHER_A_BUF is None or _GATHER_A_BUF.shape[0] < T_PAD:
        _GATHER_A_BUF = torch.empty((T_PAD, H), dtype=torch.float8_e4m3fn, device=hidden_states.device)
        _GATHER_A_SCALE_BUF = torch.empty((T_PAD, KBLKS_32), dtype=torch.int8, device=hidden_states.device)
    
    tma_gather_a_explicit[(triton.cdiv(T_PAD, 64),)](
        hidden_states, hidden_states_scale,
        _GATHER_A_BUF, _GATHER_A_SCALE_BUF,
        sorted_token_ids, T_PAD, T_TOTAL, H, KBLKS_32, BLOCK_M=64
    )
    
    sfa = _GATHER_A_SCALE_BUF[:T_PAD].view(torch.float8_e8m0fnu).contiguous()
    cache_key_w = id(gemm1_weights_scale)
    if cache_key_w not in _GATHER_W13_SCALE_CACHE:
        safe_w_scale = torch.clamp(gemm1_weights_scale, min=1e-15)
        exponent = torch.round(torch.log2(safe_w_scale)).to(torch.int32)
        e8_bytes = (exponent + 127).to(torch.uint8)
        
        expanded_e8 = e8_bytes.repeat_interleave(4, dim=1).repeat_interleave(4, dim=2)
        _GATHER_W13_SCALE_E8 = expanded_e8.view(torch.float8_e8m0fnu).contiguous()
        
        _GATHER_W13_SCALE_CACHE[cache_key_w] = (
            _GATHER_W13_SCALE_E8[:, :N_DIM//32, :].contiguous(),
            _GATHER_W13_SCALE_E8[:, N_DIM//32:, :].contiguous()
        )
    
    sfb_w1, sfb_w3 = _GATHER_W13_SCALE_CACHE[cache_key_w]

    # ---- Build per-expert pointer table (optimized: cached buffers + pointer arithmetic) ----
    # Reuse pre-allocated buffers instead of allocating every call
    _dispatch_bufs = _get_dispatch_bufs(num_g, H, KBLKS_32, N_DIM)
    ptrs = _dispatch_bufs['ptrs']
    strides = _dispatch_bufs['strides']  # pre-filled with constant values
    shapes = _dispatch_bufs['shapes']
    tensormaps = _dispatch_bufs['tensormaps']

    # Base pointers for arithmetic (replaces per-expert slicing + data_ptr())
    gather_a_base = _GATHER_A_BUF.data_ptr()
    gather_a_stride_row = H * _GATHER_A_BUF.element_size()
    sfa_base = sfa.data_ptr()
    sfa_stride_row = KBLKS_32 * sfa.element_size()
    w_base = gemm1_weights.data_ptr()
    w_expert_stride = gemm1_weights.stride(0) * gemm1_weights.element_size()
    w_n_stride = gemm1_weights.stride(1) * gemm1_weights.element_size()
    sfb_w1_base = sfb_w1.data_ptr()
    sfb_w1_expert_stride = sfb_w1.stride(0) * sfb_w1.element_size()
    sfb_w3_base = sfb_w3.data_ptr()
    sfb_w3_expert_stride = sfb_w3.stride(0) * sfb_w3.element_size()
    inter_base = intermediate.data_ptr()
    inter_stride_row = N_DIM * intermediate.element_size()  # N_DIM * sizeof(float32)

    local_g = 0
    first_e_idx = -1
    first_offset_m = 0
    first_m = 0
    for e in range(num_experts):
        n_blks = int(e_starts[e+1] - e_starts[e])
        if n_blks <= 0:
            continue
        m = n_blks * block_m
        offset_m = int(e_starts[e]) * block_m

        # shapes: only M changes per expert
        shapes[local_g, 0] = m

        # ptrs: use base + offset arithmetic (no slicing, no data_ptr() per expert)
        ptrs[local_g, 0] = gather_a_base + offset_m * gather_a_stride_row
        ptrs[local_g, 1] = w_base + e * w_expert_stride
        ptrs[local_g, 2] = w_base + e * w_expert_stride + N_DIM * w_n_stride
        ptrs[local_g, 3] = sfa_base + offset_m * sfa_stride_row
        ptrs[local_g, 4] = sfb_w1_base + e * sfb_w1_expert_stride
        ptrs[local_g, 5] = sfb_w3_base + e * sfb_w3_expert_stride
        ptrs[local_g, 6] = inter_base + offset_m * inter_stride_row

        if local_g == 0:
            first_e_idx = e
            first_offset_m = offset_m
            first_m = m
        local_g += 1

    # ---- Convert metadata to CuTe tensors (cached wrappers) ----
    p_shape, p_ptrs, p_strides, p_tmaps = _get_cute_meta_wrappers(shapes, ptrs, strides, tensormaps)

    _CUTE_KERNEL.tensormaps = p_tmaps
    
    # Compute total tiles for PTS scheduler
    total_tile_clusters = 0
    for e in range(num_experts):
        n_blks = int(e_starts[e+1] - e_starts[e])
        if n_blks > 0:
            m = n_blks * block_m
            total_tile_clusters += ((m + 127) // 128) * ((N_DIM + 127) // 128)
    total_tile_clusters = int(total_tile_clusters)

    cache_key = (num_g, total_tile_clusters)
    if cache_key not in _CUTE_COMPILED_CACHE:
        # Build init tensors ONLY for JIT compilation (first occurrence of this config)
        m0 = first_m
        off0 = first_offset_m
        e0 = first_e_idx
        a_3d = _GATHER_A_BUF[off0:off0+m0].unsqueeze(-1)
        init_a = cutlass_torch.convert_cute_tensor(a_3d, cutlass_torch.cute_tensor_like(a_3d, cutlass.Float8E4M3FN, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Float8E4M3FN, is_dynamic_layout=False)
        b_w1_3d = gemm1_weights[e0, :N_DIM, :].unsqueeze(-1)
        init_b_w1 = cutlass_torch.convert_cute_tensor(b_w1_3d, cutlass_torch.cute_tensor_like(b_w1_3d, cutlass.Float8E4M3FN, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Float8E4M3FN, is_dynamic_layout=False)
        b_w3_3d = gemm1_weights[e0, N_DIM:, :].unsqueeze(-1)
        init_b_w3 = cutlass_torch.convert_cute_tensor(b_w3_3d, cutlass_torch.cute_tensor_like(b_w3_3d, cutlass.Float8E4M3FN, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Float8E4M3FN, is_dynamic_layout=False)
        c_3d = intermediate[off0:off0+m0].unsqueeze(-1)
        init_c = cutlass_torch.convert_cute_tensor(c_3d, cutlass_torch.cute_tensor_like(c_3d, cutlass.Float32, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Float32, is_dynamic_layout=False)
        init_sfa = cutlass_torch.convert_cute_tensor(sfa[off0:off0+m0, :], cutlass_torch.cute_tensor_like(sfa[off0:off0+m0, :], cutlass.Float8E8M0FNU, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Float8E8M0FNU, is_dynamic_layout=False)
        init_sfb_w1 = cutlass_torch.convert_cute_tensor(sfb_w1[e0], cutlass_torch.cute_tensor_like(sfb_w1[e0], cutlass.Float8E8M0FNU, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Float8E8M0FNU, is_dynamic_layout=False)
        init_sfb_w3 = cutlass_torch.convert_cute_tensor(sfb_w3[e0], cutlass_torch.cute_tensor_like(sfb_w3[e0], cutlass.Float8E8M0FNU, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Float8E8M0FNU, is_dynamic_layout=False)

        try:
            print(f"CuTe JIT compiling for cache_key={cache_key}", flush=True)
            _CUTE_COMPILED_CACHE[cache_key] = cute.compile(_CUTE_KERNEL, init_a, init_b_w1, init_b_w3, init_sfa, init_sfb_w1, init_sfb_w3, init_c, num_g, p_shape, p_ptrs, p_strides, total_tile_clusters, p_tmaps, _CUTE_MAX_CLUSTERS, _CUTE_STREAM)
            # Cache init tensors for launches
            _CUTE_COMPILED_CACHE[('inits', cache_key)] = (init_a, init_b_w1, init_b_w3, init_sfa, init_sfb_w1, init_sfb_w3, init_c)
            print(f"CuTe JIT compile succeeded for {cache_key}", flush=True)
        except Exception as e:
            print(f"JIT compile failed for {cache_key}: {e}", flush=True)
            raise RuntimeError(f"CuTe JIT Failed: {e}")

    _CUTE_COMPILED = _CUTE_COMPILED_CACHE[cache_key]
    init_a, init_b_w1, init_b_w3, init_sfa, init_sfb_w1, init_sfb_w3, init_c = _CUTE_COMPILED_CACHE[('inits', cache_key)]
    _CUTE_COMPILED(init_a, init_b_w1, init_b_w3, init_sfa, init_sfb_w1, init_sfb_w3, init_c, num_g, p_shape, p_ptrs, p_strides, total_tile_clusters, p_tmaps, _CUTE_MAX_CLUSTERS, _CUTE_STREAM)
    
    return intermediate
