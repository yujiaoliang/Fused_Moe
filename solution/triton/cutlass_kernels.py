"""
cuBLAS C++ GEMM2 kernel for large-T MoE workloads.

JIT-compiled via torch.utils.cpp_extension.load_inline().
Falls back gracefully to Triton if compilation fails.

Target: NVIDIA B200 (Blackwell, SM100) — fp16 A × fp8 B with block-scale dequant.

Architecture:
  1. Dequant B from fp8 → fp16 using per-128×128 block scales (custom CUDA kernel)
  2. cuBLAS GemmEx: fp16 A × fp16 B_dequant → fp32 C (uses SM100 tensor cores)
  3. Scale C by per-row token_weights (custom CUDA kernel)

Why cuBLAS instead of CUTLASS blockwise:
  - CUTLASS SM100 blockwise (example 81) requires FP8×FP8 for both A and B
  - Our GEMM2 A-side (Intermediate) must stay fp32/fp16 for precision
  - cuBLAS on B200 uses optimized SM100 tile schedulers automatically
"""

import os
import torch

# ── Module-level state ──
_cuda_module = None
_compile_attempted = False
cutlass_available = False

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <ATen/cuda/CUDAContext.h>

// ────────────────────────────────────────────────
// Kernel: Dequant fp8_e4m3 → fp16 with block scales
// B: [N, K] fp8_e4m3fn (row-major, N=7168, K=2048)
// B_scale: [N//128, K//128] fp32
// B_out: [N, K] fp16  (same layout, element-wise dequant)
// ────────────────────────────────────────────────
__global__ void dequant_fp8_to_fp16_kernel(
    const __nv_fp8_e4m3* __restrict__ B,
    const float* __restrict__ B_scale,
    __half* __restrict__ B_out,
    int N, int K,
    int scale_n, int scale_k  // N//128, K//128
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * K;
    if (idx >= total) return;

    int n = idx / K;
    int k = idx % K;

    // Load fp8 value and convert to float
    float val = static_cast<float>(B[idx]);

    // Look up block scale: B_scale[n//128, k//128]
    int sn = n >> 7;  // n / 128
    int sk = k >> 7;  // k / 128
    float scale = B_scale[sn * scale_k + sk];

    // Dequant and store as fp16
    B_out[idx] = __float2half(val * scale);
}

// ────────────────────────────────────────────────
// Kernel: Scale output rows by token_weights
// C: [M, N] fp32 (cuBLAS output, row-major)
// token_weights: [M] fp32
// unscale: float (e.g. 8.0 for fp16 intermediate)
// ────────────────────────────────────────────────
__global__ void scale_output_kernel(
    float* __restrict__ C,
    const float* __restrict__ token_weights,
    int M, int N,
    float unscale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    int m = idx / N;
    C[idx] *= token_weights[m] * unscale;
}

// ────────────────────────────────────────────────
// Main entry: GEMM2 for a single expert
//
// Computes: C[M,N] = A[M,K] @ B[N,K]^T * token_weights * unscale
//
// A: [M, K] fp16 (Intermediate, contiguous row-major)
// B: [N, K] fp8 (gemm2_weights for one expert, contiguous row-major)
// B_scale: [N//128, K//128] fp32 (block scales)
// token_weights: [M] fp32
// C: [M, N] fp32 (output, contiguous row-major)
// B_dequant_buf: [N, K] fp16 (workspace, reused)
// unscale: float (1.0 for fp32 inter, 8.0 for fp16 inter)
// ────────────────────────────────────────────────
void cublas_gemm2(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor B_scale,
    torch::Tensor token_weights,
    torch::Tensor C,
    torch::Tensor B_dequant_buf,
    float unscale
) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);    // B is [N, K]
    int scale_n = B_scale.size(0);  // N//128
    int scale_k = B_scale.size(1);  // K//128

    auto stream = at::cuda::getCurrentCUDAStream();

    // Step 1: Dequant B[N,K] from fp8 → fp16
    {
        int total = N * K;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        dequant_fp8_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const __nv_fp8_e4m3*>(B.data_ptr()),
            B_scale.data_ptr<float>(),
            reinterpret_cast<__half*>(B_dequant_buf.data_ptr()),
            N, K, scale_n, scale_k
        );
    }

    // Step 2: cuBLAS GemmEx — C[M,N] = A[M,K] @ B_dequant[N,K]^T
    //
    // Row-major trick for cuBLAS (column-major):
    //   A row[M,K] → A_col^T [K,M], ld=K
    //   B row[N,K] → B_col^T [K,N], ld=K
    //   C row[M,N] → C_col^T [N,M], ld=N
    //
    //   Want: C^T = B * A^T (col-major)
    //   cuBLAS: C_col[N,M] = op(first)[N,K] * op(second)[K,M]
    //     first = B_col (which is B_row^T [K,N] with ld=K),
    //             CUBLAS_OP_T → B_row [N,K] → shape [N,K]
    //     second = A_col (which is A_row^T [K,M] with ld=K),
    //              CUBLAS_OP_N → [K,M]
    //   m=N, n=M, k=K
    {
        cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(handle, stream);

        float alpha = 1.0f;
        float beta = 0.0f;

        auto status = cublasGemmEx(
            handle,
            CUBLAS_OP_T,  // first (B_dequant): transpose [K,N]→[N,K]
            CUBLAS_OP_N,  // second (A): no-op, [K,M] col-major
            N,            // m = rows of result col-major = N
            M,            // n = cols of result col-major = M
            K,            // k
            &alpha,
            B_dequant_buf.data_ptr(), CUDA_R_16F, K,  // [K,N] col-major, ld=K
            A.data_ptr(), CUDA_R_16F, K,               // [K,M] col-major, ld=K
            &beta,
            C.data_ptr(), CUDA_R_32F, N,               // [N,M] col-major, ld=N
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );

        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasGemmEx failed with status " + std::to_string(status));
        }
    }

    // Step 3: Scale output by token_weights
    {
        int total = M * N;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        scale_output_kernel<<<blocks, threads, 0, stream>>>(
            C.data_ptr<float>(),
            token_weights.data_ptr<float>(),
            M, N, unscale
        );
    }
}

// ────────────────────────────────────────────────
// Trivial test kernel for compile-time validation
// ────────────────────────────────────────────────
torch::Tensor test_add(torch::Tensor a, torch::Tensor b) {
    return a + b;
}
"""

_CPP_SOURCE = r"""
void cublas_gemm2(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor B_scale,
    torch::Tensor token_weights,
    torch::Tensor C,
    torch::Tensor B_dequant_buf,
    float unscale
);

torch::Tensor test_add(torch::Tensor a, torch::Tensor b);
"""


def _setup_cuda_env():
    """Auto-detect CUDA_HOME and collect all nvidia include paths for pip installs."""
    import shutil
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return  # Already set (system CUDA install)

    # 1. Check if nvcc is in PATH (system install)
    if shutil.which("nvcc"):
        nvcc = shutil.which("nvcc")
        os.environ["CUDA_HOME"] = os.path.dirname(os.path.dirname(nvcc))
        return

    # 2. Search pip-installed nvidia packages for nvcc
    import site
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        nvidia_base = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_base):
            continue
        for d in sorted(os.listdir(nvidia_base)):
            nvcc_path = os.path.join(nvidia_base, d, "bin", "nvcc")
            if os.path.isfile(nvcc_path):
                cuda_home = os.path.join(nvidia_base, d)
                os.environ["CUDA_HOME"] = cuda_home
                os.environ["PATH"] = os.path.join(cuda_home, "bin") + ":" + os.environ.get("PATH", "")
                # Collect ALL nvidia/*/include dirs (headers are split across packages:
                # cuda_crt has nv/target, cu13 has cuda_fp8.h, etc.)
                extra_includes = []
                for pkg in sorted(os.listdir(nvidia_base)):
                    inc = os.path.join(nvidia_base, pkg, "include")
                    if os.path.isdir(inc):
                        extra_includes.append(inc)
                if extra_includes:
                    os.environ["_CUDA_EXTRA_INCLUDES"] = os.pathsep.join(extra_includes)
                return

    # 3. Fallback: /usr/local/cuda
    if os.path.isdir("/usr/local/cuda"):
        os.environ["CUDA_HOME"] = "/usr/local/cuda"


def _try_compile():
    """Attempt to JIT-compile the CUDA extension. Called once."""
    global _cuda_module, _compile_attempted, cutlass_available
    _compile_attempted = True

    try:
        # Ensure ninja is available (required by load_inline)
        try:
            import ninja
        except ImportError:
            import subprocess, sys
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "ninja", "-q"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        # Auto-detect CUDA_HOME from pip packages
        _setup_cuda_env()

        from torch.utils.cpp_extension import load_inline

        # Build extra flags: add include paths for ALL pip-installed nvidia packages
        extra_cuda_cflags = [
            "-O3",
            "--use_fast_math",
            "-gencode=arch=compute_100a,code=sm_100a",
            "--expt-relaxed-constexpr",
        ]
        extra_includes = os.environ.get("_CUDA_EXTRA_INCLUDES", "")
        for inc_dir in extra_includes.split(os.pathsep):
            if inc_dir:
                extra_cuda_cflags.append(f"-I{inc_dir}")

        _cuda_module = load_inline(
            name="moe_cublas_gemm",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["cublas_gemm2", "test_add"],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_ldflags=["-lcublas"],
            verbose=False,
            with_cuda=True,
        )
        cutlass_available = True
    except Exception as e:
        print(f"[cutlass_kernels] CUDA compilation failed, falling back to Triton: {e}")
        _cuda_module = None
        cutlass_available = False


def ensure_compiled():
    """Ensure the CUDA module is compiled. Returns True if available."""
    global _compile_attempted
    if not _compile_attempted:
        _try_compile()
    return cutlass_available


# ── Workspace caches ──
_dequant_buf_cache = {}


def get_dequant_buf(N, K, device):
    """Get or allocate a reusable fp16 buffer for B dequant [N, K]."""
    key = (N, K, device)
    if key not in _dequant_buf_cache:
        _dequant_buf_cache[key] = torch.empty((N, K), dtype=torch.float16, device=device)
    return _dequant_buf_cache[key]


def cublas_gemm2_expert(
    A,              # [M_rows, K] fp16 or fp32 (Intermediate slice for this expert)
    B,              # [N, K] fp8_e4m3fn (single expert W2, native layout)
    B_scale,        # [N//128, K//128] fp32
    token_weights,  # [M_rows] fp32
    C,              # [M_rows, N] fp32 (output slice)
    use_fp16_inter, # bool
):
    """
    Run cuBLAS-based GEMM2 for a single expert.
    B is in its native [N, K] layout (no transpose needed).
    """
    if not cutlass_available or _cuda_module is None:
        raise RuntimeError("CUDA module not compiled")

    K = A.shape[1]
    N = B.shape[0]
    device = A.device

    # Ensure A is fp16 for cuBLAS HGEMM
    if A.dtype != torch.float16:
        A_fp16 = A.half()
    else:
        A_fp16 = A

    # Get workspace buffer for dequanted B
    B_dequant_buf = get_dequant_buf(N, K, device)

    unscale = 8.0 if use_fp16_inter else 1.0

    _cuda_module.cublas_gemm2(A_fp16, B, B_scale, token_weights, C, B_dequant_buf, unscale)
    return C
