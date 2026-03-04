"""
Isolate Phase 3 GEMM1+SwiGLU vs PyTorch to find where the divergence starts.
We call the full fused kernel but only compare the final output.
Focus: small T to make PyTorch baseline fast.
"""
import torch
import torch.nn.functional as F
import solution.triton.kernel as kernel_mod
import triton

def run():
    T = 16  # Small for fast PyTorch loop
    device = torch.device('cuda')
    
    # FP8 inputs
    hidden_states = torch.randint(-10, 10, (T, 7168), dtype=torch.int8, device=device).to(torch.float8_e4m3fn)
    hidden_states_scale = torch.rand((7168 // 128, T), dtype=torch.float32, device=device) * 0.1
    
    routing_logits = torch.randn((T, 256), dtype=torch.float32, device=device)
    routing_bias = torch.zeros((256,), dtype=torch.float32, device=device)
    
    gemm1_weights = torch.randint(-10, 10, (32, 4096, 7168), dtype=torch.int8, device=device).to(torch.float8_e4m3fn)
    gemm1_weights_scale = torch.rand((32, 4096 // 128, 7168 // 128), dtype=torch.float32, device=device) * 0.1
    
    gemm2_weights = torch.randint(-10, 10, (32, 7168, 2048), dtype=torch.int8, device=device).to(torch.float8_e4m3fn)
    gemm2_weights_scale = torch.rand((32, 7168 // 128, 2048 // 128), dtype=torch.float32, device=device) * 0.1
    
    # Routing
    topk_idx, topk_weights = kernel_mod.ds_routing(routing_logits, routing_bias, 1.0)
    sorted_token_ids, block_expert_ids, sorted_weights, num_padded = \
        kernel_mod.moe_sort_tokens(topk_idx, topk_weights, 0, 16, T, device)
    
    print(f"T={T}, num_padded={num_padded}")
    print(f"sorted_token_ids shape: {sorted_token_ids.shape}, dtype: {sorted_token_ids.dtype}")
    print(f"block_expert_ids shape: {block_expert_ids.shape}, dtype: {block_expert_ids.dtype}")
    print(f"sorted_weights shape: {sorted_weights.shape}, dtype: {sorted_weights.dtype}")
    print(f"topk_idx[:4]: {topk_idx[:4]}")
    
    # ── Run Phase 3 fully fused kernel ──
    output_fp32 = torch.zeros((T, 7168), dtype=torch.float32, device=device)
    
    grid_fused = lambda META: (
        triton.cdiv(num_padded, META['BLOCK_M']) * triton.cdiv(7168, META['BLOCK_H']),
    )
    kernel_mod._fully_fused_moe_kernel[grid_fused](
        A_ptr=hidden_states,
        A_scale_ptr=hidden_states_scale,
        B1_ptr=gemm1_weights,
        B1_scale_ptr=gemm1_weights_scale,
        B2_ptr=gemm2_weights,
        B2_scale_ptr=gemm2_weights_scale,
        C_ptr=output_fp32,
        token_weights_ptr=sorted_weights,
        token_ids_ptr=sorted_token_ids,
        expert_ids_ptr=block_expert_ids,
        T=T, num_padded=num_padded,
        H=7168, N1=4096, N1_HALF=2048,
        stride_at=hidden_states.stride(0), stride_ah=hidden_states.stride(1),
        stride_as0=hidden_states_scale.stride(0), stride_as1=hidden_states_scale.stride(1),
        stride_b1e=gemm1_weights.stride(0), stride_b1n=gemm1_weights.stride(1), stride_b1h=gemm1_weights.stride(2),
        stride_b1se=gemm1_weights_scale.stride(0), stride_b1sn=gemm1_weights_scale.stride(1), stride_b1sh=gemm1_weights_scale.stride(2),
        stride_b2e=gemm2_weights.stride(0), stride_b2n=gemm2_weights.stride(2), stride_b2h=gemm2_weights.stride(1),
        stride_b2se=gemm2_weights_scale.stride(0), stride_b2sh=gemm2_weights_scale.stride(1), stride_b2sn=gemm2_weights_scale.stride(2),
        stride_ct=output_fp32.stride(0), stride_ch=output_fp32.stride(1),
        BLOCK_M=16, BLOCK_N1=64, BLOCK_H=64
    )
    triton_out = output_fp32.clone()
    
    # ── Run PyTorch baseline ──
    out_pt = torch.zeros((T, 7168), dtype=torch.float32, device=device)
    
    # Dequantize all inputs to fp32
    A = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.t()  # [T, 56]
    A_scale_expanded = A_scale.unsqueeze(-1).expand(T, 56, 128).reshape(T, 7168)
    A_dequant = A * A_scale_expanded
    
    W13 = gemm1_weights.to(torch.float32)
    W13_scale = gemm1_weights_scale  # [32, 32, 56]
    W13_scale_expanded = W13_scale.unsqueeze(2).unsqueeze(-1).expand(32, 32, 128, 56, 128).reshape(32, 4096, 7168)
    W13_dequant = W13 * W13_scale_expanded
    
    W2 = gemm2_weights.to(torch.float32)
    W2_scale = gemm2_weights_scale  # [32, 56, 16]
    W2_scale_expanded = W2_scale.unsqueeze(2).unsqueeze(-1).expand(32, 56, 128, 16, 128).reshape(32, 7168, 2048)
    W2_dequant = W2 * W2_scale_expanded
    
    for t in range(T):
        for i in range(8):  # topk=8
            expert_id = topk_idx[t, i].item()
            weight = topk_weights[t, i].item()
            if expert_id >= 32:
                continue
            
            h = A_dequant[t:t+1]  # [1, 7168]
            w13 = W13_dequant[expert_id]  # [4096, 7168]
            w1, w3 = w13[:2048], w13[2048:]  # Each [2048, 7168]
            
            y1 = h @ w1.t()  # [1, 2048]
            y3 = h @ w3.t()  # [1, 2048]
            swiglu = F.silu(y3) * y1  # [1, 2048]
            
            w2 = W2_dequant[expert_id]  # [7168, 2048]
            res = swiglu @ w2.t()  # [1, 7168]
            out_pt[t] += (res * weight)[0]
    
    # Compare
    diff = (triton_out - out_pt).abs()
    max_d = diff.max().item()
    mean_d = diff.mean().item()
    print(f"\nMax absolute diff: {max_d:.4f}")
    print(f"Mean absolute diff: {mean_d:.4f}")
    
    if max_d > 1.0:
        idx = (diff == max_d).nonzero()
        t_idx, h_idx = idx[0].tolist()
        print(f"\nWorst divergence at [{t_idx}, {h_idx}]:")
        print(f"  Triton: {triton_out[t_idx, h_idx].item():.4f}")
        print(f"  PyTorch: {out_pt[t_idx, h_idx].item():.4f}")
        
        # Show distribution of errors
        big_errors = (diff > 100).sum().item()
        total = diff.numel()
        print(f"\n  Elements with error > 100: {big_errors}/{total} ({100*big_errors/total:.2f}%)")
        
        # Check if errors are concentrated in specific rows or columns
        row_max = diff.max(dim=1).values
        col_max = diff.max(dim=0).values
        worst_rows = (row_max > 100).nonzero().squeeze()
        worst_cols = (col_max > 100).nonzero().squeeze()
        print(f"  Rows with error > 100: {worst_rows.numel()} / {T}")
        print(f"  Cols with error > 100: {worst_cols.numel()} / 7168")
        if worst_cols.numel() > 0 and worst_cols.numel() <= 20:
            print(f"  Worst column indices: {worst_cols.tolist()}")
    else:
        print("PASSED! Errors within tolerance.")

if __name__ == '__main__':
    run()
