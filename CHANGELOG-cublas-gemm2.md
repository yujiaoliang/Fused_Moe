# CHANGELOG: cuBLAS GEMM2 Dispatch + FP16 Intermediate Optimization

## Branch: `feat/cublas-gemm2-dispatch`
**Date**: 2026-04-14
**Status**: 19/19 PASSED, A/B tested (neutral on Modal, cuBLAS untested on eval env)

---

## Summary

This branch adds a JIT-compiled cuBLAS GEMM2 dispatch path for large-T workloads (T >= 2048) and refines the FP16 intermediate buffer logic. The cuBLAS path activates only in environments with a full CUDA toolkit (e.g., the eval Docker `flashinfer-ci-cu132`), with graceful fallback to Triton on Modal or other pip-based environments.

---

## Changes

### New Files

#### `solution/triton/cutlass_kernels.py`
- JIT-compiled C++ CUDA module via `torch.utils.cpp_extension.load_inline()`
- **Architecture**: Dequant FP8 B -> FP16 -> cuBLAS GemmEx (fp16 x fp16 -> fp32) -> Scale output
- Three CUDA kernels:
  1. `dequant_fp8_to_fp16_kernel` — Per-128x128 block-scale dequantization
  2. `cublasGemmEx` — SM100 tensor core GEMM (row-major trick for cuBLAS column-major API)
  3. `scale_output_kernel` — Per-row token_weights multiplication with optional unscale (8.0 for fp16 intermediate)
- Auto-detection of CUDA_HOME from pip packages or system install
- Module-level compilation cache (`_compile_attempted` flag) — only attempts once
- Reusable FP16 workspace buffer for B dequantization

#### `scripts/test_cutlass_modal.py`
- Modal test script for validating cuBLAS JIT compilation on B200
- Includes CUDA environment probing, fp8 dequant correctness test, and full GEMM2 benchmark
- Documents the `nv/target` stub workaround for pip CUDA packages

### Modified Files

#### `solution/triton/kernel.py`
1. **cuBLAS GEMM2 dispatch** (lines 2331-2365):
   - Import `cutlass_kernels` module at top level with graceful fallback
   - Dispatch condition: `T >= 2048 AND exact_dispatch AND ensure_compiled()`
   - Per-expert cuBLAS GEMM2 loop reading `block_offsets` (already CPU-synced via exact dispatch)
   - Falls back to Triton GEMM2 if cuBLAS compilation fails

2. **`FP16_INTER_T_MIN = 512`** (was effectively `SMALL_MEDIUM_T_MIN = 32`):
   - FP16 intermediate buffer only for T >= 512 where bandwidth savings dominate
   - FP32 intermediate for T < 512 — simpler code path, no 0.125/8.0 scaling needed

3. **`USE_FP16_INTER` conditional in all GEMM kernels**:
   - `_fused_moe_gemm1_swiglu_kernel`: Store as `(swiglu * 0.125).to(fp16)` or fp32
   - `_fused_moe_gemm2_kernel`: `b.to(a.dtype)` naturally adapts (fp16 or fp32)
   - `_fused_moe_gemm2_t901_kernel`: Explicit branch for dot dtype and output scaling
   - `_small_medium_fused_moe_gemm*_kernel`: Same USE_FP16_INTER conditionals
   - `_medium_fused_moe_gemm*_kernel`: Same USE_FP16_INTER conditionals

4. **Autotune config pruning**:
   - Removed redundant configs from `_small_medium_fused_moe_gemm2_kernel` (23 -> 10 configs)
   - Removed redundant configs from `_small_medium_fused_moe_gemm1_swiglu_kernel` (similar reduction)
   - Faster first-run compilation with no measurable performance impact

#### `scripts/test_modal.py` and `scripts/ab_test_modal.py`
- Added `ninja` to `pip_install` for cuBLAS JIT compilation support

---

## A/B Test Results (Modal B200, same instance)

```
Workload        A speed    B speed    Delta    Verdict
------------ ---------- ---------- -------- ----------
1a4c6ba1         32.93x     33.02x    +0.3%     = SAME
2e69caee         40.65x     41.59x    +2.3%     BETTER
4822167c         41.40x     41.79x    +0.9%     = SAME
58a34f27         14.41x     14.46x    +0.3%     = SAME
5e8dc11c         12.87x     12.87x    -0.0%     = SAME
5eadab1e         44.32x     44.19x    -0.3%     = SAME
6230e838         41.43x     42.14x    +1.7%     = SAME
74d7ff04         41.44x     41.31x    -0.3%     = SAME
76010cb4         42.26x     42.20x    -0.1%     = SAME
81955b1e         41.38x     41.94x    +1.3%     = SAME
8cba5890         45.43x     45.94x    +1.1%     = SAME
8f1ff9f1         38.97x     39.49x    +1.3%     = SAME
a7c2bcfd         46.13x     46.44x    +0.7%     = SAME
b8f4f012         43.86x     43.35x    -1.2%     = SAME
e05c6c03         49.58x     50.05x    +0.9%     = SAME
e626d3e6         42.09x     41.48x    -1.4%     = SAME
eedc63b2         43.31x     43.61x    +0.7%     = SAME
f7d6ac7c         43.58x     43.73x    +0.3%     = SAME
fc378037         41.66x     41.75x    +0.2%     = SAME

Mean speedup: A=39.35x  B=39.54x  Delta=+0.5%
Summary: 1 improved, 0 regressed, 18 neutral (+/-2% threshold)
```

**Verdict**: Neutral on Modal (cuBLAS not activated). cuBLAS path will activate in eval env.

---

## Key Learnings

1. **Pip CUDA toolkit is incomplete for JIT**: Missing `nv/target` header and `thrust/complex.h`. Creating stubs reveals more missing dependencies (whack-a-mole). Eval env has full CUDA at `/usr/local/cuda`.

2. **torch._scaled_mm requires both inputs FP8**: Cannot do fp16 x fp8 mixed. Not viable for GEMM2 where A-side must stay fp32/fp16.

3. **CUTLASS SM100 tcgen05.mma**: The `.kind` qualifier applies to the entire operation — cannot mix TF32 A x FP8 B at hardware level.

4. **FP16 intermediate threshold**: T >= 512 is the sweet spot. Below this, fp32 avoids the 0.125/8.0 scaling overhead with no bandwidth penalty (small workloads are compute-bound on tensor cores, not memory-bound).

---

## Architecture Diagram

```
kernel.py (entry point)
  |-- Routing         -> PyTorch (sigmoid, group-top-4, global-top-8, normalize)
  |-- Token Sort      -> Triton (hybrid CPU/GPU sorting)
  |-- GEMM1 + SwiGLU  -> Triton (FP8xFP8 tensor cores, post-dot scale)
  |-- GEMM2           -> Triton (all T)  |  cuBLAS (T>=2048, eval env only)
  |                       ^-- cutlass_kernels.py (JIT-compiled C++)
  '-- Token Reduce    -> Triton (scatter-add to output)
```
