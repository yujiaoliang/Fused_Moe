"""
Compare Phase 2 Triton kernels vs the same PyTorch baseline.
If Phase 2 also shows huge errors vs PyTorch, the PyTorch baseline is wrong.
"""
import torch
import torch.nn.functional as F
import solution.triton.kernel as kernel_mod
import triton

def run():
    T = 16
    device = torch.device('cuda')
    
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
    
    # ═══════════════════════════════════════════════
    # Phase 2: Run the two-kernel approach
    # ═══════════════════════════════════════════════
    BLOCK_M = 64
    sorted_token_ids, block_expert_ids, sorted_weights, num_padded = \
        kernel_mod.moe_sort_tokens(topk_idx, topk_weights, 0, BLOCK_M, T, device)
    
    # GEMM1
    gemm1_out = torch.empty((num_padded, 4096), dtype=torch.bfloat16, device=device)
    grid1 = lambda META: (triton.cdiv(num_padded, META['BLOCK_M']) * (4096 // META['BLOCK_N']),)
    kernel_mod._grouped_gemm_fp8_kernel[grid1](
        hidden_states, gemm1_weights, gemm1_out,
        hidden_states_scale, gemm1_weights_scale,
        sorted_token_ids, block_expert_ids,
        T, 7168, 4096,
        hidden_states.stride(0), hidden_states.stride(1),
        gemm1_weights.stride(0), gemm1_weights.stride(1), gemm1_weights.stride(2),
        gemm1_out.stride(0), gemm1_out.stride(1),
        hidden_states_scale.stride(0), hidden_states_scale.stride(1),
        gemm1_weights_scale.stride(0), gemm1_weights_scale.stride(1), gemm1_weights_scale.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=128, BLOCK_K=128, GROUP_SIZE_M=8,
    )
    
    # SwiGLU
    swiglu_out = torch.empty((num_padded, 2048), dtype=torch.bfloat16, device=device)
    grid_swiglu = lambda META: (triton.cdiv(num_padded, META['BLOCK_M']), triton.cdiv(2048, META['BLOCK_I']))
    kernel_mod._swiglu_kernel[grid_swiglu](
        gemm1_out, swiglu_out, num_padded, 2048,
        BLOCK_M=64, BLOCK_I=128,
    )
    
    # GEMM2
    gemm2_out_buf = torch.empty((num_padded, 7168), dtype=torch.bfloat16, device=device)
    grid2 = lambda META: (triton.cdiv(num_padded, META['BLOCK_M']) * triton.cdiv(7168, META['BLOCK_N']),)
    kernel_mod._grouped_gemm_bf16xfp8_kernel[grid2](
        swiglu_out, gemm2_weights, gemm2_out_buf,
        gemm2_weights_scale,
        block_expert_ids, sorted_weights,
        num_padded, 2048, 7168,
        swiglu_out.stride(0), swiglu_out.stride(1),
        gemm2_weights.stride(0), gemm2_weights.stride(1), gemm2_weights.stride(2),
        gemm2_out_buf.stride(0), gemm2_out_buf.stride(1),
        gemm2_weights_scale.stride(0), gemm2_weights_scale.stride(1), gemm2_weights_scale.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=128, BLOCK_K=128, GROUP_SIZE_M=8, MUL_WEIGHT=True,
    )
    
    # Scatter-add to output
    phase2_out = torch.zeros((T, 7168), dtype=torch.float32, device=device)
    for i in range(num_padded):
        token_id = sorted_token_ids[i].item()
        if token_id < T:
            phase2_out[token_id] += gemm2_out_buf[i].to(torch.float32)
    
    # ═══════════════════════════════════════════════
    # PyTorch baseline  
    # ═══════════════════════════════════════════════
    out_pt = torch.zeros((T, 7168), dtype=torch.float32, device=device)
    A = hidden_states.to(torch.float32)
    A_scale = hidden_states_scale.t()  # [T, 56]
    A_scale_expanded = A_scale.unsqueeze(-1).expand(T, 56, 128).reshape(T, 7168)
    A_dequant = A * A_scale_expanded
    
    W13 = gemm1_weights.to(torch.float32)
    W13_scale = gemm1_weights_scale
    W13_scale_expanded = W13_scale.unsqueeze(2).unsqueeze(-1).expand(32, 32, 128, 56, 128).reshape(32, 4096, 7168)
    W13_dequant = W13 * W13_scale_expanded
    
    W2 = gemm2_weights.to(torch.float32)  
    W2_scale = gemm2_weights_scale  
    W2_scale_expanded = W2_scale.unsqueeze(2).unsqueeze(-1).expand(32, 56, 128, 16, 128).reshape(32, 7168, 2048)
    W2_dequant = W2 * W2_scale_expanded
    
    for t in range(T):
        for i in range(8):
            expert_id = topk_idx[t, i].item()
            weight = topk_weights[t, i].item()
            if expert_id >= 32:
                continue
            h = A_dequant[t:t+1]
            w13 = W13_dequant[expert_id]
            w1, w3 = w13[:2048], w13[2048:]
            y1 = h @ w1.t()
            y3 = h @ w3.t()
            swiglu = F.silu(y3) * y1
            w2 = W2_dequant[expert_id]
            res = swiglu @ w2.t()
            out_pt[t] += (res * weight)[0]
    
    # ═══════════════════════════════════════════════
    # Compare
    # ═══════════════════════════════════════════════
    diff_p2 = (phase2_out - out_pt).abs()
    max_p2 = diff_p2.max().item()
    
    print(f"Phase 2 vs PyTorch max diff: {max_p2:.4f}")
    
    if max_p2 > 10.0:
        print("  → Phase 2 ALSO diverges from PyTorch! The PyTorch baseline dequant is probably wrong.")
    else:
        print("  → Phase 2 matches PyTorch. The bug is exclusively in Phase 3.")
    
    # Also directly compare Phase 2 vs Phase 3
    phase3_out = torch.zeros((T, 7168), dtype=torch.float32, device=device)
    # CRITICAL: Sort with BLOCK_M=64 (same as the module-level constant)
    # The fused kernel internally uses sorter_block_m=64 to index expert_ids
    sorted_token_ids_3, block_expert_ids_3, sorted_weights_3, num_padded_3 = \
        kernel_mod.moe_sort_tokens(topk_idx, topk_weights, 0, 64, T, device)
    grid_fused = lambda META: (
        triton.cdiv(num_padded_3, META['BLOCK_M']) * triton.cdiv(7168, META['BLOCK_H']),
    )
    kernel_mod._fully_fused_moe_kernel[grid_fused](
        A_ptr=hidden_states,
        A_scale_ptr=hidden_states_scale,
        B1_ptr=gemm1_weights,
        B1_scale_ptr=gemm1_weights_scale,
        B2_ptr=gemm2_weights,
        B2_scale_ptr=gemm2_weights_scale,
        C_ptr=phase3_out,
        token_weights_ptr=sorted_weights_3,
        token_ids_ptr=sorted_token_ids_3,
        expert_ids_ptr=block_expert_ids_3,
        T=T, num_padded=num_padded_3,
        H=7168, N1=4096, N1_HALF=2048,
        stride_at=hidden_states.stride(0), stride_ah=hidden_states.stride(1),
        stride_as0=hidden_states_scale.stride(0), stride_as1=hidden_states_scale.stride(1),
        stride_b1e=gemm1_weights.stride(0), stride_b1n=gemm1_weights.stride(1), stride_b1h=gemm1_weights.stride(2),
        stride_b1se=gemm1_weights_scale.stride(0), stride_b1sn=gemm1_weights_scale.stride(1), stride_b1sh=gemm1_weights_scale.stride(2),
        stride_b2e=gemm2_weights.stride(0), stride_b2n=gemm2_weights.stride(2), stride_b2h=gemm2_weights.stride(1),
        stride_b2se=gemm2_weights_scale.stride(0), stride_b2sh=gemm2_weights_scale.stride(1), stride_b2sn=gemm2_weights_scale.stride(2),
        stride_ct=phase3_out.stride(0), stride_ch=phase3_out.stride(1),
        BLOCK_M=16, BLOCK_N1=64, BLOCK_H=64
    )
    
    diff_23 = (phase2_out - phase3_out).abs()
    max_23 = diff_23.max().item()
    print(f"\nPhase 2 vs Phase 3 max diff: {max_23:.4f}")

if __name__ == '__main__':
    run()
