"""
Test tl.make_tensor_descriptor with triton.set_allocator fix.
The tensor descriptor API requires scratch memory allocation for TMA descriptors.
"""
import modal

app = modal.App('test-td-v3')
image = modal.Image.from_registry('flashinfer/flashinfer-ci-cu132:latest', add_python='3.12')

KERNEL_SRC = r'''
import triton
import triton.language as tl
import torch

# ── Mixed-ptr GEMM: A=ptr_arith, B=tensor_desc ──
@triton.jit
def _gemm_td(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    desc_b = tl.make_tensor_descriptor(
        B_ptr,
        shape=[K, N],
        strides=[N, 1],
        block_shape=[BLOCK_K, BLOCK_N],
    )

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load_tensor_descriptor(desc_b, [k, pid_n * BLOCK_N])
        acc += tl.dot(a, b.to(a.dtype), out_dtype=tl.float32)

    c_ptrs = C_ptr + rm[:, None] * N + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    tl.store(c_ptrs, acc)


# ── Mixed-ptr GEMM: A=ptr_arith, B=tensor_desc, runtime base offset ──
@triton.jit
def _gemm_td_expert(
    A_ptr, B_ptr, C_ptr,
    expert_offset,  # runtime: expert_id * stride_be
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_bk, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    desc_b = tl.make_tensor_descriptor(
        B_ptr + expert_offset,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N],
    )

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load_tensor_descriptor(desc_b, [k, pid_n * BLOCK_N])
        acc += tl.dot(a, b.to(a.dtype), out_dtype=tl.float32)

    c_ptrs = C_ptr + rm[:, None] * N + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    tl.store(c_ptrs, acc)


# ── Baseline: ptr_arith ──
@triton.jit
def _gemm_ptr(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(B_ptr + rk[:, None] * N + rn[None, :])
        acc += tl.dot(a, b.to(a.dtype), out_dtype=tl.float32)

    c_ptrs = C_ptr + rm[:, None] * N + rn[None, :]
    tl.store(c_ptrs, acc)


# ── block_ptr baseline ──
@triton.jit
def _gemm_bp(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    b_block_ptr = tl.make_block_ptr(
        base=B_ptr, shape=(K, N), strides=(N, 1),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N), order=(1, 0),
    )
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        a = tl.load(A_ptr + rm[:, None] * K + rk[None, :])
        b = tl.load(b_block_ptr)
        acc += tl.dot(a, b.to(a.dtype), out_dtype=tl.float32)
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
    c_ptrs = C_ptr + rm[:, None] * N + (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    tl.store(c_ptrs, acc)
'''


@app.function(image=image, gpu='B200')
def test():
    import sys, importlib, time
    import triton
    import triton.language as tl
    import torch

    print(f"Triton: {triton.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Set allocator for tensor descriptor scratch memory ──
    print("\nSetting triton allocator...")
    try:
        triton.set_allocator(triton.runtime.torch_allocator.TorchAllocator())
        print("  Used TorchAllocator")
    except (AttributeError, ImportError):
        try:
            def torch_alloc(size, align, stream):
                # Align size up
                aligned_size = (size + align - 1) // align * align
                t = torch.empty(aligned_size, dtype=torch.uint8, device='cuda')
                return t.data_ptr()
            triton.set_allocator(torch_alloc)
            print("  Used custom torch allocator")
        except Exception as e:
            print(f"  FAILED to set allocator: {e}")

    with open('/root/_td3_kernels.py', 'w') as f:
        f.write(KERNEL_SRC)
    sys.path.insert(0, '/root')
    mod = importlib.import_module('_td3_kernels')

    # ── Test 1: correctness ──
    print("\n" + "=" * 60)
    print("TEST 1: tensor_desc GEMM correctness")
    print("=" * 60)
    M, N, K = 128, 256, 256
    BM, BN, BK = 64, 128, 128
    A = torch.randn(M, K, device='cuda', dtype=torch.float16)
    B = torch.randn(K, N, device='cuda', dtype=torch.float16)
    C_td = torch.zeros(M, N, device='cuda', dtype=torch.float32)
    ref = A.float() @ B.float()

    try:
        mod._gemm_td[(M // BM, N // BN)](A, B, C_td, M, N, K, BM, BN, BK)
        torch.cuda.synchronize()
        diff = (C_td - ref).abs().max().item()
        print(f"  tensor_desc GEMM: OK! max_diff={diff:.6f}")
        td_ok = True
    except Exception as e:
        import traceback
        print(f"  tensor_desc GEMM: FAILED ({type(e).__name__})")
        traceback.print_exc()
        td_ok = False

    # ── Test 2: runtime base offset (expert dispatch pattern) ──
    print("\n" + "=" * 60)
    print("TEST 2: tensor_desc with runtime base + runtime strides")
    print("=" * 60)
    E_count = 32
    B_expert = torch.randn(E_count, K, N, device='cuda', dtype=torch.float16)
    C_exp = torch.zeros(M, N, device='cuda', dtype=torch.float32)
    eid = 7
    offset = eid * B_expert.stride(0)
    try:
        mod._gemm_td_expert[(M // BM, N // BN)](
            A, B_expert, C_exp, offset, M, N, K,
            B_expert.stride(1), B_expert.stride(2),
            BM, BN, BK,
        )
        torch.cuda.synchronize()
        ref_exp = A.float() @ B_expert[eid].float()
        diff = (C_exp - ref_exp).abs().max().item()
        print(f"  Expert dispatch: OK! max_diff={diff:.6f}")
        td_expert_ok = True
    except Exception as e:
        import traceback
        print(f"  Expert dispatch: FAILED ({type(e).__name__})")
        traceback.print_exc()
        td_expert_ok = False

    # ── Benchmark: GEMM2-like shape ──
    print("\n" + "=" * 60)
    print("BENCHMARK: 4096x7168x2048 fp16 (GEMM2-like shape)")
    print("=" * 60)
    M_b, N_b, K_b = 4096, 7168, 2048
    BM_b, BN_b, BK_b = 128, 128, 128
    A_b = torch.randn(M_b, K_b, device='cuda', dtype=torch.float16)
    B_b = torch.randn(K_b, N_b, device='cuda', dtype=torch.float16)
    C_b = torch.zeros(M_b, N_b, device='cuda', dtype=torch.float32)
    grid = (M_b // BM_b, N_b // BN_b)

    N_WARMUP = 5
    N_ITER = 30
    results = {}

    for name, fn, ok in [
        ('ptr_arith', mod._gemm_ptr, True),
        ('block_ptr', mod._gemm_bp, True),
        ('tensor_desc', mod._gemm_td, td_ok),
    ]:
        if not ok:
            print(f"  {name:15s}: SKIPPED")
            continue
        try:
            for _ in range(N_WARMUP):
                fn[grid](A_b, B_b, C_b, M_b, N_b, K_b, BM_b, BN_b, BK_b)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(N_ITER):
                fn[grid](A_b, B_b, C_b, M_b, N_b, K_b, BM_b, BN_b, BK_b)
            torch.cuda.synchronize()
            t = (time.perf_counter() - t0) / N_ITER * 1000
            results[name] = t
            print(f"  {name:15s}: {t:.3f} ms")
        except Exception as e:
            print(f"  {name:15s}: FAILED ({type(e).__name__}: {e})")

    if 'ptr_arith' in results:
        base = results['ptr_arith']
        for name, t in results.items():
            if name != 'ptr_arith':
                print(f"  {name} vs ptr_arith: {base/t:.3f}x")

    print("\nDone.")


if __name__ == '__main__':
    with modal.enable_output():
        with app.run():
            test.remote()
