"""
Modal script to check TMA descriptor availability and tl.make_block_ptr
compilation on eval Triton 3.7.0 (flashinfer-ci-cu132 Docker).
"""
import modal

app = modal.App('test-tma')
image = modal.Image.from_registry('flashinfer/flashinfer-ci-cu132:latest', add_python='3.12')

KERNEL_SRC = r'''
import triton
import triton.language as tl
import torch

# Test 1: Block pointer kernel (fp8 weight load via tl.make_block_ptr)
@triton.jit
def _test_block_ptr_kernel(
    B_ptr, Out_ptr,
    N: tl.constexpr, K: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    b_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(N, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    b = tl.load(b_block_ptr)
    # Store first row as fp32 output
    out_ptrs = Out_ptr + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(out_ptrs, b[0, :].to(tl.float32))

# Test 2: Block pointer GEMM (fp16 A × fp8 B with block ptr on B side)
@triton.jit
def _test_block_ptr_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # A: pointer arithmetic (like our Intermediate buffer)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # B: block pointer for TMA-eligible loads
    b_block_ptr = tl.make_block_ptr(
        base=B_ptr,
        shape=(K, N),
        strides=(N, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_block_ptr)
        acc += tl.dot(a, b.to(a.dtype), out_dtype=tl.float32)
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    c_ptrs = C_ptr + rm[:, None] * N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(c_ptrs + pid_n * BLOCK_N, acc)
'''


@app.function(image=image, gpu='B200')
def test():
    import sys, importlib

    # ── Check 1: TMA device descriptor API ──
    print("=" * 60)
    print("CHECK 1: TMA device-side descriptor API")
    print("=" * 60)
    try:
        from triton.language.extra.cuda import experimental_device_tensormap_create2d
        print("  experimental_device_tensormap_create2d: AVAILABLE")
    except (ImportError, AttributeError) as e:
        print(f"  experimental_device_tensormap_create2d: NOT AVAILABLE ({e})")

    try:
        from triton.language.extra import cuda as tl_cuda
        print(f"  triton.language.extra.cuda attrs: {[a for a in dir(tl_cuda) if not a.startswith('_')]}")
    except ImportError as e:
        print(f"  triton.language.extra.cuda: NOT IMPORTABLE ({e})")

    # ── Check 2: Triton version ──
    import triton
    import triton.language as tl
    print(f"\nTriton version: {triton.__version__}")
    print(f"tl.make_block_ptr: {'available' if hasattr(tl, 'make_block_ptr') else 'NOT FOUND'}")
    print(f"tl.advance: {'available' if hasattr(tl, 'advance') else 'NOT FOUND'}")

    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA cap: {torch.cuda.get_device_capability(0)}")

    # ── Check 3: Block pointer kernel compilation ──
    print("\n" + "=" * 60)
    print("CHECK 3: tl.make_block_ptr compilation (fp8 load)")
    print("=" * 60)
    with open('/root/_tma_kernels.py', 'w') as f:
        f.write(KERNEL_SRC)
    sys.path.insert(0, '/root')
    mod = importlib.import_module('_tma_kernels')

    N, K = 256, 128
    BLOCK_N, BLOCK_K = 128, 128
    B = torch.randn(K, N, device='cuda', dtype=torch.float16).to(torch.float8_e4m3fn)
    Out = torch.zeros(N, device='cuda', dtype=torch.float32)

    try:
        grid = (N // BLOCK_N,)
        mod._test_block_ptr_kernel[grid](B, Out, N=N, K=K, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        torch.cuda.synchronize()
        print(f"  Block ptr fp8 load: OK (out max={Out.abs().max().item():.4f})")
    except Exception as e:
        import traceback
        print(f"  Block ptr fp8 load: FAILED ({type(e).__name__})")
        traceback.print_exc()

    # ── Check 4: Block pointer GEMM (mixed pointer arithmetic + block ptr) ──
    print("\n" + "=" * 60)
    print("CHECK 4: Mixed A=ptr_arith, B=block_ptr GEMM compilation")
    print("=" * 60)
    M2, N2, K2 = 128, 256, 256
    BM, BN, BK = 64, 128, 128
    A2 = torch.randn(M2, K2, device='cuda', dtype=torch.float16)
    B2 = torch.randn(K2, N2, device='cuda', dtype=torch.float16)
    C2 = torch.zeros(M2, N2, device='cuda', dtype=torch.float32)

    try:
        grid2 = (M2 // BM, N2 // BN)
        mod._test_block_ptr_gemm_kernel[grid2](A2, B2, C2, M2, N2, K2, BM, BN, BK)
        torch.cuda.synchronize()
        ref = A2.float() @ B2.float()
        max_diff = (C2 - ref).abs().max().item()
        print(f"  Mixed ptr GEMM: OK (max_diff={max_diff:.6f})")
    except Exception as e:
        import traceback
        print(f"  Mixed ptr GEMM: FAILED ({type(e).__name__})")
        traceback.print_exc()

    print("\nDone.")


if __name__ == '__main__':
    with modal.enable_output():
        with app.run():
            test.remote()
