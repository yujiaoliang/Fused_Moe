"""
Modal test: verify cuBLAS GEMM2 JIT compilation + correctness on B200.

Usage:
  python -m modal run scripts/test_cutlass_modal.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("test-cutlass-compilation")

# Use debian_slim base but create nv/target stub to bypass missing header.
# The actual nv/target header provides NV_IF_TARGET macro for host/device code.
# The pip CUDA packages don't ship it, so we create a minimal stub.
_NV_TARGET_STUB = r'''
#ifndef __NV_TARGET_H__
#define __NV_TARGET_H__
#ifdef __CUDA_ARCH__
#define NV_IF_TARGET(target, ...) __VA_ARGS__
#define NV_IF_ELSE_TARGET(target, t, f) t
#else
#define NV_IF_TARGET(target, ...)
#define NV_IF_ELSE_TARGET(target, t, f) f
#endif
#define NV_IS_DEVICE (__CUDA_ARCH__)
#define NV_IS_HOST (!__CUDA_ARCH__)
#define NV_PROVIDES_SM_100 (__CUDA_ARCH__ >= 1000)
#define NV_PROVIDES_SM_90 (__CUDA_ARCH__ >= 900)
#define NV_PROVIDES_SM_80 (__CUDA_ARCH__ >= 800)
#define NV_PROVIDES_SM_70 (__CUDA_ARCH__ >= 700)
#endif
'''

import base64
_nv_target_b64 = base64.b64encode(_NV_TARGET_STUB.encode()).decode()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "triton", "numpy", "ninja", "nvidia-cuda-nvcc", "nvidia-thrust")
    .run_commands(
        "mkdir -p /usr/local/lib/python3.12/site-packages/nvidia/cu13/include/nv",
        f"echo '{_nv_target_b64}' | base64 -d > /usr/local/lib/python3.12/site-packages/nvidia/cu13/include/nv/target",
    )
)


def _collect_all_include_dirs():
    """Find ALL include directories from nvidia pip packages + nvcc internal paths."""
    import os
    import site
    import subprocess

    include_dirs = []
    seen = set()

    # 1. Pip nvidia packages
    for sp in site.getsitepackages():
        nvidia_base = os.path.join(sp, "nvidia")
        if not os.path.isdir(nvidia_base):
            continue
        for pkg_name in sorted(os.listdir(nvidia_base)):
            inc = os.path.join(nvidia_base, pkg_name, "include")
            if os.path.isdir(inc) and inc not in seen:
                seen.add(inc)
                include_dirs.append(inc)

    # 2. Search for nv/target using find (fallback for non-standard locations)
    try:
        result = subprocess.run(
            ["find", "/usr/local/lib", "-name", "target", "-path", "*/nv/*", "-type", "f"],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.strip().split("\n"):
            if line and "/nv/target" in line:
                # e.g. /usr/local/lib/.../include/nv/target -> add .../include
                inc_dir = line.rsplit("/nv/target", 1)[0]
                if inc_dir not in seen:
                    seen.add(inc_dir)
                    include_dirs.append(inc_dir)
    except Exception:
        pass

    # 3. nvcc --include-path (builtin include dirs)
    try:
        import shutil
        nvcc_bin = shutil.which("nvcc")
        if nvcc_bin:
            # nvcc has built-in include paths; check its parent tree
            nvcc_root = os.path.dirname(os.path.dirname(nvcc_bin))
            for subdir in ["include", os.path.join("targets", "x86_64-linux", "include")]:
                inc = os.path.join(nvcc_root, subdir)
                if os.path.isdir(inc) and inc not in seen:
                    seen.add(inc)
                    include_dirs.append(inc)
    except Exception:
        pass

    return include_dirs


def _setup_cuda_env_full():
    """Auto-detect CUDA_HOME and collect ALL nvidia include paths."""
    import os
    import shutil
    import site

    info = {"cuda_home": None, "nvcc": None, "include_dirs": [], "nv_target_found": False}

    # 1. Find nvcc
    nvcc = shutil.which("nvcc")
    if nvcc:
        info["nvcc"] = nvcc
        info["cuda_home"] = os.path.dirname(os.path.dirname(nvcc))
    else:
        for sp in site.getsitepackages():
            nvidia_base = os.path.join(sp, "nvidia")
            if not os.path.isdir(nvidia_base):
                continue
            for d in sorted(os.listdir(nvidia_base)):
                nvcc_path = os.path.join(nvidia_base, d, "bin", "nvcc")
                if os.path.isfile(nvcc_path):
                    info["nvcc"] = nvcc_path
                    info["cuda_home"] = os.path.join(nvidia_base, d)
                    os.environ["CUDA_HOME"] = info["cuda_home"]
                    os.environ["PATH"] = os.path.join(info["cuda_home"], "bin") + ":" + os.environ.get("PATH", "")
                    break
            if info["nvcc"]:
                break

    if not info["cuda_home"] and os.path.isdir("/usr/local/cuda"):
        info["cuda_home"] = "/usr/local/cuda"

    if info["cuda_home"]:
        os.environ["CUDA_HOME"] = info["cuda_home"]

    # 2. Collect ALL include dirs
    info["include_dirs"] = _collect_all_include_dirs()

    # 3. Check for nv/target
    for inc_dir in info["include_dirs"]:
        nv_target = os.path.join(inc_dir, "nv", "target")
        if os.path.isfile(nv_target):
            info["nv_target_found"] = True
            info["nv_target_path"] = inc_dir
            break

    return info


@app.function(image=image, gpu="B200:1", timeout=1800)
def test_cutlass_compilation() -> str:
    """Test CUDA JIT compilation and cuBLAS GEMM2 on B200."""
    import os
    import time
    import subprocess
    import torch

    lines = []
    def log(msg):
        lines.append(str(msg))
        print(msg)

    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA version: {torch.version.cuda}")

    # ── Find nv/target anywhere on the system ──
    log("\n=== Searching for nv/target ===")
    try:
        r = subprocess.run(
            ["find", "/usr/local/lib", "-name", "target", "-path", "*/nv/*", "-type", "f"],
            capture_output=True, text=True, timeout=10
        )
        if r.stdout.strip():
            log(f"  Found: {r.stdout.strip()}")
        else:
            log("  NOT FOUND in /usr/local/lib")
    except Exception as e:
        log(f"  Search error: {e}")

    # Also search in wider paths
    try:
        r2 = subprocess.run(
            ["find", "/usr", "-name", "target", "-path", "*/nv/*", "-type", "f"],
            capture_output=True, text=True, timeout=10
        )
        if r2.stdout.strip():
            for line in r2.stdout.strip().split("\n"):
                log(f"  Found: {line}")
        else:
            log("  NOT FOUND in /usr")
    except Exception as e:
        log(f"  Search error: {e}")

    # Check nvidia-cuda-crt package location
    log("\n=== nvidia-cuda-crt package ===")
    try:
        r3 = subprocess.run(
            ["pip", "show", "-f", "nvidia-cuda-crt"],
            capture_output=True, text=True, timeout=10
        )
        log(r3.stdout[:2000] if r3.stdout else "  (no output)")
    except Exception as e:
        log(f"  Error: {e}")

    # ── Probe CUDA environment ──
    log("\n=== CUDA Environment Probe ===")
    info = _setup_cuda_env_full()
    log(f"  CUDA_HOME: {info['cuda_home']}")
    log(f"  nvcc: {info['nvcc']}")
    log(f"  Include dirs ({len(info['include_dirs'])}):")
    for d in info["include_dirs"]:
        key_headers = []
        for h in ["cuda.h", "cuda_fp8.h", "cuda_fp16.h", "cublas_v2.h"]:
            if os.path.isfile(os.path.join(d, h)):
                key_headers.append(h)
        nv_dir = os.path.join(d, "nv")
        has_nv = os.path.isdir(nv_dir)
        log(f"    {d}")
        if key_headers:
            log(f"      headers: {', '.join(key_headers)}")
        if has_nv:
            nv_files = sorted(os.listdir(nv_dir))
            log(f"      nv/: {nv_files}")
    log(f"  nv/target found: {info['nv_target_found']}")
    if info.get("nv_target_path"):
        log(f"  nv/target in: {info['nv_target_path']}")

    # Build include flags
    extra_include_flags = [f"-I{d}" for d in info["include_dirs"]]
    log(f"  Extra -I flags: {len(extra_include_flags)}")

    if not info["nv_target_found"]:
        log("\n  WARNING: nv/target NOT FOUND — compilation will likely fail")
        log("  Checking if nvcc has it built-in...")
        if info["nvcc"]:
            try:
                r4 = subprocess.run(
                    [info["nvcc"], "--include-path"],
                    capture_output=True, text=True, timeout=5
                )
                log(f"  nvcc --include-path: {r4.stdout.strip()}")
                log(f"  nvcc stderr: {r4.stderr.strip()[:200]}")
            except Exception as e:
                log(f"  nvcc probe error: {e}")

    # ── Test 1: Simple CUDA + fp8 kernel ──
    log("\n=== Test 1: load_inline CUDA + fp8 ===")
    try:
        from torch.utils.cpp_extension import load_inline

        cuda_src = r"""
        #include <torch/extension.h>
        #include <cuda_fp16.h>
        #include <cuda_fp8.h>
        #include <cuda_runtime.h>

        __global__ void dequant_kernel(const __nv_fp8_e4m3* B, __half* out, int n) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < n) {
                out[i] = __float2half(static_cast<float>(B[i]) * 2.0f);
            }
        }

        torch::Tensor test_dequant(torch::Tensor B) {
            int n = B.numel();
            auto out = torch::empty({n}, torch::TensorOptions().dtype(torch::kFloat16).device(B.device()));
            dequant_kernel<<<(n+255)/256, 256>>>(
                reinterpret_cast<const __nv_fp8_e4m3*>(B.data_ptr()),
                reinterpret_cast<__half*>(out.data_ptr()),
                n
            );
            return out;
        }
        """
        cpp_src = "torch::Tensor test_dequant(torch::Tensor B);"

        cflags = ["-O3", "--use_fast_math", "-gencode=arch=compute_100a,code=sm_100a"] + extra_include_flags

        t1_start = time.time()
        mod = load_inline(
            name="test_fp8_dq",
            cpp_sources=cpp_src,
            cuda_sources=cuda_src,
            functions=["test_dequant"],
            extra_cuda_cflags=cflags,
            verbose=True,
            with_cuda=True,
        )
        t1 = time.time() - t1_start
        B_test = torch.tensor([1.0, 0.5, -0.25, 2.0], device="cuda").to(torch.float8_e4m3fn)
        out = mod.test_dequant(B_test)
        log(f"  PASSED ({t1:.1f}s): dequant={out.float().tolist()}")
    except Exception as e:
        log(f"  FAILED: {e}")
        import traceback
        log(traceback.format_exc())

    # ── Test 2: Full cuBLAS GEMM2 ──
    log("\n=== Test 2: cuBLAS GEMM2 ===")
    try:
        from torch.utils.cpp_extension import load_inline

        cuda_src_full = r"""
        #include <torch/extension.h>
        #include <cuda_runtime.h>
        #include <cuda_fp16.h>
        #include <cuda_fp8.h>
        #include <cublas_v2.h>
        #include <ATen/cuda/CUDAContext.h>

        __global__ void dequant_fp8_to_fp16_kernel(
            const __nv_fp8_e4m3* __restrict__ B,
            const float* __restrict__ B_scale,
            __half* __restrict__ B_out,
            int N, int K, int scale_n, int scale_k
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= N * K) return;
            int n = idx / K;
            int k = idx % K;
            float val = static_cast<float>(B[idx]);
            float scale = B_scale[(n >> 7) * scale_k + (k >> 7)];
            B_out[idx] = __float2half(val * scale);
        }

        __global__ void scale_output_kernel(
            float* __restrict__ C, const float* __restrict__ tw,
            int M, int N, float unscale
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= M * N) return;
            C[idx] *= tw[idx / N] * unscale;
        }

        void cublas_gemm2(
            torch::Tensor A, torch::Tensor B, torch::Tensor B_scale,
            torch::Tensor token_weights, torch::Tensor C,
            torch::Tensor B_dequant_buf, float unscale
        ) {
            int M = A.size(0), K = A.size(1), N = B.size(0);
            int scale_n = B_scale.size(0), scale_k = B_scale.size(1);
            auto stream = at::cuda::getCurrentCUDAStream();

            {
                int total = N * K, threads = 256;
                dequant_fp8_to_fp16_kernel<<<(total+255)/256, threads, 0, stream>>>(
                    reinterpret_cast<const __nv_fp8_e4m3*>(B.data_ptr()),
                    B_scale.data_ptr<float>(),
                    reinterpret_cast<__half*>(B_dequant_buf.data_ptr()),
                    N, K, scale_n, scale_k
                );
            }

            {
                cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
                cublasSetStream(handle, stream);
                float alpha = 1.0f, beta = 0.0f;
                cublasGemmEx(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    B_dequant_buf.data_ptr(), CUDA_R_16F, K,
                    A.data_ptr(), CUDA_R_16F, K,
                    &beta,
                    C.data_ptr(), CUDA_R_32F, N,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
                );
            }

            {
                int total = M * N;
                scale_output_kernel<<<(total+255)/256, 256, 0, stream>>>(
                    C.data_ptr<float>(), token_weights.data_ptr<float>(),
                    M, N, unscale
                );
            }
        }
        """

        cpp_src_full = """
        void cublas_gemm2(
            torch::Tensor A, torch::Tensor B, torch::Tensor B_scale,
            torch::Tensor token_weights, torch::Tensor C,
            torch::Tensor B_dequant_buf, float unscale
        );
        """

        cflags2 = [
            "-O3", "--use_fast_math",
            "-gencode=arch=compute_100a,code=sm_100a",
            "--expt-relaxed-constexpr",
        ] + extra_include_flags

        t2_start = time.time()
        mod2 = load_inline(
            name="moe_cublas_test",
            cpp_sources=cpp_src_full,
            cuda_sources=cuda_src_full,
            functions=["cublas_gemm2"],
            extra_cuda_cflags=cflags2,
            extra_ldflags=["-lcublas"],
            verbose=False,
            with_cuda=True,
        )
        t2 = time.time() - t2_start
        log(f"  Compilation: PASSED ({t2:.1f}s)")

        # Correctness test
        M, K, N = 256, 2048, 7168
        device = "cuda"
        A_fp16 = torch.randn(M, K, dtype=torch.float16, device=device) * 0.1
        B_fp32 = torch.randn(N, K, dtype=torch.float32, device=device) * 0.1
        B_fp8 = B_fp32.to(torch.float8_e4m3fn)
        B_scale = torch.ones(N // 128, K // 128, dtype=torch.float32, device=device)
        tw = torch.ones(M, dtype=torch.float32, device=device)
        C = torch.zeros(M, N, dtype=torch.float32, device=device)
        B_buf = torch.empty(N, K, dtype=torch.float16, device=device)

        mod2.cublas_gemm2(A_fp16, B_fp8, B_scale, tw, C, B_buf, 1.0)

        C_ref = A_fp16.float() @ B_fp8.float().t()
        abs_err = (C - C_ref).abs()
        log(f"  Max abs error (scale=1): {abs_err.max().item():.4e}")
        log(f"  Mean abs error: {abs_err.mean().item():.4e}")

        # With random block scales
        B_scale_r = torch.rand(N // 128, K // 128, dtype=torch.float32, device=device) * 2.0
        C2 = torch.zeros(M, N, dtype=torch.float32, device=device)
        mod2.cublas_gemm2(A_fp16, B_fp8, B_scale_r, tw, C2, B_buf, 1.0)

        B_dq = torch.zeros(N, K, dtype=torch.float32, device=device)
        for sn in range(N // 128):
            for sk in range(K // 128):
                s = B_scale_r[sn, sk].item()
                B_dq[sn*128:(sn+1)*128, sk*128:(sk+1)*128] = B_fp8[sn*128:(sn+1)*128, sk*128:(sk+1)*128].float() * s
        C2_ref = A_fp16.float() @ B_dq.t()
        abs_err2 = (C2 - C2_ref).abs()
        log(f"  Max abs error (block-scale): {abs_err2.max().item():.4e}")
        log(f"  Mean abs error: {abs_err2.mean().item():.4e}")

        # Benchmark
        log("\n=== Benchmark ===")
        for M_b in [64, 256, 1024, 4096]:
            A_b = torch.randn(M_b, K, dtype=torch.float16, device=device) * 0.1
            tw_b = torch.ones(M_b, dtype=torch.float32, device=device)
            C_b = torch.zeros(M_b, N, dtype=torch.float32, device=device)

            for _ in range(5):
                mod2.cublas_gemm2(A_b, B_fp8, B_scale, tw_b, C_b, B_buf, 1.0)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(20):
                mod2.cublas_gemm2(A_b, B_fp8, B_scale, tw_b, C_b, B_buf, 1.0)
            end.record()
            torch.cuda.synchronize()
            cublas_ms = start.elapsed_time(end) / 20.0

            A_f = A_b.float()
            B_f16 = B_fp8.half()
            for _ in range(5):
                torch.mm(A_f, B_f16.float().t())
            torch.cuda.synchronize()
            start.record()
            for _ in range(20):
                torch.mm(A_f, B_f16.float().t())
            end.record()
            torch.cuda.synchronize()
            ref_ms = start.elapsed_time(end) / 20.0

            log(f"  M={M_b:>5d}: cuBLAS={cublas_ms:.3f}ms  fp32_mm={ref_ms:.3f}ms  ratio={ref_ms/cublas_ms:.2f}x")

    except Exception as e:
        log(f"  FAILED: {e}")
        import traceback
        log(traceback.format_exc())

    log(f"\n=== Done ===")
    return "\n".join(lines)


@app.local_entrypoint()
def main():
    print("Running cuBLAS GEMM2 test on Modal B200...")
    result = test_cutlass_compilation.remote()
    print("\n=== FULL RESULTS ===")
    print(result)
