"""Test torch._scaled_mm API on B200 to find the right block-scale strategy."""
import modal

app = modal.App("test-scaled-mm")
volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "numpy")
)

@app.function(
    gpu="B200:1",
    image=image,
    timeout=300,
)
def test_scaled_mm():
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.cuda.get_device_capability()}")
    
    # Test basic _scaled_mm API
    print("\n=== Testing torch._scaled_mm ===")
    
    # Check if _scaled_mm exists
    if not hasattr(torch, '_scaled_mm'):
        print("ERROR: torch._scaled_mm not available!")
        return
    
    # Check signature
    try:
        print(f"_scaled_mm help: {torch._scaled_mm.__doc__[:200] if torch._scaled_mm.__doc__ else 'no doc'}")
    except:
        print("  (no docstring available)")
    
    device = 'cuda'
    
    # Test 1: Basic fp8 matmul with per-tensor scale
    print("\n--- Test 1: Per-tensor scale ---")
    M, N, K = 64, 4096, 7168
    
    # Create fp8 tensors  
    A_fp32 = torch.randn(M, K, device=device)
    B_fp32 = torch.randn(N, K, device=device)  # [N, K], will be transposed
    
    A_fp8 = A_fp32.to(torch.float8_e4m3fn)
    B_fp8 = B_fp32.to(torch.float8_e4m3fn)
    
    scale_a = torch.ones(1, device=device, dtype=torch.float32)
    scale_b = torch.ones(1, device=device, dtype=torch.float32)
    
    try:
        result = torch._scaled_mm(A_fp8, B_fp8.t(), scale_a=scale_a, scale_b=scale_b)
        print(f"  Result shape: {result.shape}, dtype: {result.dtype}")
        print(f"  SUCCESS!")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Test 2: Per-row scaling
    print("\n--- Test 2: Per-row scale ---")
    scale_a_row = torch.ones(M, 1, device=device, dtype=torch.float32)
    scale_b_row = torch.ones(1, N, device=device, dtype=torch.float32)
    
    try:
        result = torch._scaled_mm(A_fp8, B_fp8.t(), scale_a=scale_a_row, scale_b=scale_b_row)
        print(f"  Result shape: {result.shape}, dtype: {result.dtype}")
        print(f"  SUCCESS!")
    except Exception as e:
        print(f"  ERROR: {e}")
    
    # Test 3: Block scale (our use case)
    print("\n--- Test 3: Block scale (128x128) ---")
    # Our scales are [M/128, K/128] for A and [N/128, K/128] for B
    # Can we pass block-scale tensors?
    try:
        scale_a_block = torch.ones(M // 128 if M >= 128 else 1, K // 128, device=device, dtype=torch.float32)
        scale_b_block = torch.ones(N // 128, K // 128, device=device, dtype=torch.float32)
        result = torch._scaled_mm(A_fp8, B_fp8.t(), scale_a=scale_a_block, scale_b=scale_b_block)
        print(f"  Result shape: {result.shape}, dtype: {result.dtype}")
        print(f"  SUCCESS with block scale!")
    except Exception as e:
        print(f"  Block scale ERROR: {e}")
    
    # Test 4: Check output dtype options
    print("\n--- Test 4: Output dtype control ---")
    for out_dtype in [torch.float32, torch.bfloat16, torch.float16]:
        try:
            result = torch._scaled_mm(
                A_fp8, B_fp8.t(),
                scale_a=torch.ones(1, device=device, dtype=torch.float32),
                scale_b=torch.ones(1, device=device, dtype=torch.float32),
                out_dtype=out_dtype,
            )
            print(f"  out_dtype={out_dtype}: shape={result.shape}, dtype={result.dtype} ✓")
        except Exception as e:
            print(f"  out_dtype={out_dtype}: ERROR: {e}")
    
    # Test 5: Performance comparison
    print("\n--- Test 5: Performance comparison ---")
    import time
    
    # Warm up
    for _ in range(5):
        _ = torch.mm(A_fp32, B_fp32.t())
        _ = torch._scaled_mm(A_fp8, B_fp8.t(), scale_a=scale_a, scale_b=scale_b)
    
    torch.cuda.synchronize()
    
    # Time fp32 matmul
    start = time.perf_counter()
    for _ in range(100):
        _ = torch.mm(A_fp32, B_fp32.t())
    torch.cuda.synchronize()
    fp32_time = (time.perf_counter() - start) / 100
    
    # Time fp8 _scaled_mm
    start = time.perf_counter()
    for _ in range(100):
        _ = torch._scaled_mm(A_fp8, B_fp8.t(), scale_a=scale_a, scale_b=scale_b)
    torch.cuda.synchronize()
    fp8_time = (time.perf_counter() - start) / 100
    
    print(f"  fp32 matmul:    {fp32_time*1000:.3f}ms")
    print(f"  fp8 _scaled_mm: {fp8_time*1000:.3f}ms")
    print(f"  Speedup:        {fp32_time/fp8_time:.2f}x")
    
    # Test 6: Precision with our actual weight shapes
    print("\n--- Test 6: Precision with MoE shapes ---")
    T_vals = [7, 64, 512]
    for T_test in T_vals:
        A_test = torch.randn(T_test, 7168, device=device)
        W_test = torch.randn(4096, 7168, device=device)
        
        ref = torch.mm(A_test, W_test.t())
        
        A_fp8_t = A_test.to(torch.float8_e4m3fn)
        W_fp8_t = W_test.to(torch.float8_e4m3fn)
        
        result_fp8 = torch._scaled_mm(
            A_fp8_t, W_fp8_t.t(),
            scale_a=torch.ones(1, device=device),
            scale_b=torch.ones(1, device=device),
            out_dtype=torch.float32,
        )
        
        abs_err = (result_fp8 - ref).abs().max().item()
        rel_err = ((result_fp8 - ref).abs() / (ref.abs() + 1e-10)).max().item()
        print(f"  T={T_test}: abs_err={abs_err:.2e}, rel_err={rel_err:.2e}")

@app.local_entrypoint()
def main():
    test_scaled_mm.remote()
