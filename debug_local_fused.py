import torch
import torch.nn.functional as F
from solution.triton.kernel import kernel

def run_local_fused():
    T = 4096
    device = torch.device('cuda')

    # Create dummy datatypes matching fp8 benchmarks exactly natively inside the B200 harness
    hidden_states = torch.randint(-10, 10, (T, 7168), dtype=torch.int8, device=device).to(torch.float8_e4m3fn)
    hidden_states_scale = torch.rand((7168 // 128, T), dtype=torch.float32, device=device) * 0.1

    routed_scaling_factor = torch.tensor(1.0, dtype=torch.float32, device=device)
    routing_logits = torch.randn((T, 256), dtype=torch.float32, device=device)
    routing_bias = torch.zeros((256,), dtype=torch.float32, device=device)

    # 32 local experts for Phase 3 monolithic fused Triton loop natively
    gemm1_weights = torch.randint(-10, 10, (32, 4096, 7168), dtype=torch.int8, device=device).to(torch.float8_e4m3fn)
    gemm1_weights_scale = torch.rand((32, 4096 // 128, 7168 // 128), dtype=torch.float32, device=device) * 0.1

    gemm2_weights = torch.randint(-10, 10, (32, 7168, 2048), dtype=torch.int8, device=device).to(torch.float8_e4m3fn)
    gemm2_weights_scale = torch.rand((32, 7168 // 128, 2048 // 128), dtype=torch.float32, device=device) * 0.1
    local_expert_offset = torch.tensor([0], dtype=torch.int32, device=device)[0]
    out_16 = torch.empty((T, 7168), dtype=torch.bfloat16, device=device)
    print("Running kernel_16...")
    # Inject an evaluation wrapper for 16
    import solution.triton.kernel as kernel_mod
    
    # Run the exact routing to get the sorted layout
    topk_idx, topk_weights = kernel_mod.ds_routing(routing_logits, routing_bias, 1.0)
    sorted_token_ids, block_expert_ids, sorted_weights, num_padded = kernel_mod.moe_sort_tokens(topk_idx, topk_weights, 0, 16, T, device)
    
    output_fp32 = torch.zeros((T, 7168), dtype=torch.float32, device=device)
    import triton
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
    out_16.copy_(output_fp32.to(torch.bfloat16))

    # We cannot hot swap BLOCK_M_3 natively since it's hardcoded. We need to evaluate the PyTorch baseline directly!
    print("Running PyTorch baseline natively to find where Phase 3 explodes...")
    
    out_pt = torch.zeros((T, 7168), dtype=torch.float32, device=device)
    W13 = gemm1_weights.to(torch.float32)
    W2 = gemm2_weights.to(torch.float32)
    A = hidden_states.to(torch.float32)
    
    # ── Re-apply Phase 2 PyTorch evaluation mappings natively ──
    A_scale = hidden_states_scale.t() # [T, 56]
    A_scale_expanded = A_scale.unsqueeze(-1).expand(T, 56, 128).reshape(T, 7168)
    A = A * A_scale_expanded
    
    W13_scale_expanded = gemm1_weights_scale.unsqueeze(2).unsqueeze(-1).expand(32, 32, 128, 56, 128).reshape(32, 4096, 7168)
    W13 = W13 * W13_scale_expanded
    
    W2_scale_expanded = gemm2_weights_scale.unsqueeze(2).unsqueeze(-1).expand(32, 56, 128, 16, 128).reshape(32, 7168, 2048)
    W2 = W2 * W2_scale_expanded
    
    for t in range(T):
        for i in range(8):
            expert_id = topk_idx[t, i].item()
            weight = topk_weights[t, i].item()
            if expert_id >= 32: continue
                
            h = A[t:t+1]
            w13_expert = W13[expert_id]
            w1, w3 = w13_expert.chunk(2, dim=0)
            
            y1 = h @ w1.t()
            y3 = h @ w3.t()
            swiglu = F.silu(y3) * y1
            
            w2_expert = W2[expert_id]
            res = swiglu @ w2_expert.t()
            out_pt[t] += (res * weight)[0]
    
    diff = (out_16.to(torch.float32) - out_pt).abs()
    max_d = diff.max().item()
    print(f"Max absolute diff: {max_d:.6f}")
    if max_d > 1.0:
        idx = (diff == max_d).nonzero()
        print(f"Found diverging threshold natively! Indices: {idx[0].tolist()}")
        print(f"  Triton says:  {out_16[idx[0][0], idx[0][1]].item()}")
        print(f"  PyTorch says: {out_pt[idx[0][0], idx[0][1]].item()}")

if __name__ == '__main__':
    run_local_fused()
