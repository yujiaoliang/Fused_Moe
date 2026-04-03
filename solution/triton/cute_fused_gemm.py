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
    
    # ---- PHASE 1: Just enter and return (NO CUDA ops at all) ----
    if _CUTE_DEBUG_PHASE <= 1:
        raise RuntimeError(f"CUTE_PHASE_1_ENTRY_ONLY")
    
    total_blks_host = int(total_blocks.item())
    T_PAD = total_blks_host * block_m
    T_TOTAL, H = hidden_states.shape
    N_DIM = gemm1_weights.shape[1] // 2
    
    if T_PAD == 0:
        return intermediate
        
    KBLKS_32 = H // 32
    
    # Allocate gather buffers — need enough room for 128-aligned expert padding
    # T_PAD is the Triton-side padded total; CuTe may need more due to 128-alignment
    T_PAD_CUTE = T_PAD  # will be recomputed after expert_ms if block_m < 128
    
    if _GATHER_A_BUF is None or _GATHER_A_BUF.shape[0] < T_PAD:
        _GATHER_A_BUF = torch.empty((T_PAD, H), dtype=torch.float8_e4m3fn, device=hidden_states.device)
        _GATHER_A_SCALE_BUF = torch.empty((T_PAD, KBLKS_32), dtype=torch.int8, device=hidden_states.device)
    
    # (Synchronize removed to allow async execution)
    
    tma_gather_a_explicit[(triton.cdiv(T_PAD, 64),)](
        hidden_states, hidden_states_scale,
        _GATHER_A_BUF, _GATHER_A_SCALE_BUF,
        sorted_token_ids, T_PAD, T_TOTAL, H, KBLKS_32, BLOCK_M=64
    )
    # (Synchronize removed)
    
    # ---- PHASE 2: Gather done ----
    if _CUTE_DEBUG_PHASE <= 2:
        raise RuntimeError(f"CUTE_PHASE_2_GATHER_OK: T_PAD={T_PAD}, T_TOTAL={T_TOTAL}")
    
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
    
    # ---- PHASE 3: Scale conversion done ----
    if _CUTE_DEBUG_PHASE <= 3:
        raise RuntimeError(f"CUTE_PHASE_3_SCALES_OK: sfa={sfa.shape}, sfb_w1={sfb_w1.shape}")

    # Convert active blocks into ptr distributions
    expert_ms = []
    e_starts = block_offsets.cpu().numpy()
    num_experts = len(e_starts) - 1
    for e in range(num_experts):
        n_blks = e_starts[e+1] - e_starts[e]
        expert_ms.append(n_blks * block_m)
        
    num_g = sum(1 for me in expert_ms if me > 0)
    expert_list = [me for me in expert_ms if me > 0]
    e_idx_list = [i for i, me in enumerate(expert_ms) if me > 0]
    
    if num_g == 0:
        return intermediate

    if num_g > 16:
        # 32-expert workloads crash CuTe even with batch splitting (ILLEGAL_ADDRESS in 2nd batch).
        # Root cause is structural in PTS — not fixable with host-side workarounds.
        return None
        
    ptrs = torch.empty((num_g, 7), dtype=torch.int64, device='cuda')
    strides = torch.empty((num_g, 7, 2), dtype=torch.int32, device='cuda')
    shapes = torch.empty((num_g, 4), dtype=torch.int32, device='cuda')
    tensormaps = torch.empty((_CUTE_NUM_SMS, 7, 16), dtype=torch.int64, device='cuda')
    
    for local_g, e_idx in enumerate(e_idx_list):
        m = expert_list[local_g]
        shapes[local_g] = torch.tensor([m, N_DIM, H, 1], dtype=torch.int32)
        
        offset_m = e_starts[e_idx] * block_m
        ptrs[local_g, 0] = _GATHER_A_BUF[offset_m:offset_m+m].data_ptr()
        ptrs[local_g, 1] = gemm1_weights[e_idx, :N_DIM, :].data_ptr()
        ptrs[local_g, 2] = gemm1_weights[e_idx, N_DIM:, :].data_ptr()
        ptrs[local_g, 3] = sfa[offset_m:offset_m+m, :].data_ptr()
        ptrs[local_g, 4] = sfb_w1[e_idx].data_ptr()
        ptrs[local_g, 5] = sfb_w3[e_idx].data_ptr()
        ptrs[local_g, 6] = intermediate[offset_m:offset_m+m].data_ptr()
        
        strides[local_g, 0] = torch.tensor([H, 1], dtype=torch.int32)
        strides[local_g, 1] = torch.tensor([H, 1], dtype=torch.int32)
        strides[local_g, 2] = torch.tensor([H, 1], dtype=torch.int32)
        strides[local_g, 3] = torch.tensor([KBLKS_32, 1], dtype=torch.int32)
        strides[local_g, 4] = torch.tensor([KBLKS_32, 1], dtype=torch.int32)
        strides[local_g, 5] = torch.tensor([KBLKS_32, 1], dtype=torch.int32)
        strides[local_g, 6] = torch.tensor([N_DIM, 1], dtype=torch.int32)
        
        if local_g == 0:
            a_3d = _GATHER_A_BUF[offset_m:offset_m+m].unsqueeze(-1)
            init_a = cutlass_torch.convert_cute_tensor(a_3d, cutlass_torch.cute_tensor_like(a_3d, cutlass.Float8E4M3FN, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Float8E4M3FN, is_dynamic_layout=False)
            b_w1_3d = gemm1_weights[e_idx, :N_DIM, :].unsqueeze(-1)
            init_b_w1 = cutlass_torch.convert_cute_tensor(b_w1_3d, cutlass_torch.cute_tensor_like(b_w1_3d, cutlass.Float8E4M3FN, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Float8E4M3FN, is_dynamic_layout=False)
            b_w3_3d = gemm1_weights[e_idx, N_DIM:, :].unsqueeze(-1)
            init_b_w3 = cutlass_torch.convert_cute_tensor(b_w3_3d, cutlass_torch.cute_tensor_like(b_w3_3d, cutlass.Float8E4M3FN, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Float8E4M3FN, is_dynamic_layout=False)
            c_3d = intermediate[offset_m:offset_m+m].unsqueeze(-1)
            init_c = cutlass_torch.convert_cute_tensor(c_3d, cutlass_torch.cute_tensor_like(c_3d, cutlass.Float32, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Float32, is_dynamic_layout=False)
            init_sfa = cutlass_torch.convert_cute_tensor(sfa[offset_m:offset_m+m, :], cutlass_torch.cute_tensor_like(sfa[offset_m:offset_m+m, :], cutlass.Float8E8M0FNU, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Float8E8M0FNU, is_dynamic_layout=False)
            init_sfb_w1 = cutlass_torch.convert_cute_tensor(sfb_w1[e_idx], cutlass_torch.cute_tensor_like(sfb_w1[e_idx], cutlass.Float8E8M0FNU, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Float8E8M0FNU, is_dynamic_layout=False)
            init_sfb_w3 = cutlass_torch.convert_cute_tensor(sfb_w3[e_idx], cutlass_torch.cute_tensor_like(sfb_w3[e_idx], cutlass.Float8E8M0FNU, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Float8E8M0FNU, is_dynamic_layout=False)

    p_shape = cutlass_torch.convert_cute_tensor(shapes, cutlass_torch.cute_tensor_like(shapes, cutlass.Int32, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Int32, is_dynamic_layout=False)
    p_ptrs = cutlass_torch.convert_cute_tensor(ptrs, cutlass_torch.cute_tensor_like(ptrs, cutlass.Int64, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Int64, is_dynamic_layout=False)
    p_strides = cutlass_torch.convert_cute_tensor(strides, cutlass_torch.cute_tensor_like(strides, cutlass.Int32, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Int32, is_dynamic_layout=False)
    p_tmaps = cutlass_torch.convert_cute_tensor(tensormaps, cutlass_torch.cute_tensor_like(tensormaps, cutlass.Int64, is_dynamic_layout=False, assumed_align=16)[0], cutlass.Int64, is_dynamic_layout=False)

    _CUTE_KERNEL.tensormaps = p_tmaps
    total_tile_clusters = int(sum(((m + 127) // 128) * ((N_DIM + 127) // 128) for m in expert_list))
    
    cache_key = (num_g, total_tile_clusters)
    if cache_key not in _CUTE_COMPILED_CACHE:
        try:
            print(f"CuTe JIT compiling for cache_key={cache_key}", flush=True)
            _CUTE_COMPILED_CACHE[cache_key] = cute.compile(_CUTE_KERNEL, init_a, init_b_w1, init_b_w3, init_sfa, init_sfb_w1, init_sfb_w3, init_c, num_g, p_shape, p_ptrs, p_strides, total_tile_clusters, p_tmaps, _CUTE_MAX_CLUSTERS, _CUTE_STREAM)
            print(f"CuTe JIT compile succeeded for {cache_key}", flush=True)
        except Exception as e:
            print(f"JIT compile failed for {cache_key}: {e}", flush=True)
            raise RuntimeError(f"CuTe JIT Failed: {e}")

    _CUTE_COMPILED = _CUTE_COMPILED_CACHE[cache_key]
    _CUTE_COMPILED(init_a, init_b_w1, init_b_w3, init_sfa, init_sfb_w1, init_sfb_w3, init_c, num_g, p_shape, p_ptrs, p_strides, total_tile_clusters, p_tmaps, _CUTE_MAX_CLUSTERS, _CUTE_STREAM)
    
    return intermediate
