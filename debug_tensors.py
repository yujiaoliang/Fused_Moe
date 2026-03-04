import torch
import os
import sys

# We run inside /root, so solution is at /root/solution
sys.path.insert(0, "/root")

from solution.triton.kernel import kernel, _fused_moe_gemm1_swiglu_kernel, _fused_moe_gemm2_scatter_kernel

import torch.nn.functional as F

def baseline_phase2(routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, routed_scaling_factor, local_start):
    from solution.triton.kernel import ds_routing
    T = routing_logits.shape[0]
    device = hidden_states.device
    topk_idx, topk_weights = ds_routing(routing_logits, routing_bias, float(routed_scaling_factor))
    
    # Dequantize all weights to float32 first
    W13 = gemm1_weights.to(torch.float32)  # [E, 4096, 7168]
    W2 = gemm2_weights.to(torch.float32)   # [E, 7168, 2048]
    A = hidden_states.to(torch.float32)    # [T, 7168]
    
    # Apply block scales
    # We know block scales are size 128
    # A_scale: [56, T] -> [T, 56]
    A_scale = hidden_states_scale.t() # [T, 56]
    # expand 128
    A_scale_expanded = A_scale.unsqueeze(-1).expand(T, 56, 128).reshape(T, 7168)
    A = A * A_scale_expanded
    
    # W13_scale: [E, 32, 56]
    W13_scale_expanded = gemm1_weights_scale.unsqueeze(2).unsqueeze(-1).expand(32, 32, 128, 56, 128).reshape(32, 4096, 7168)
    W13 = W13 * W13_scale_expanded
    
    # W2_scale: [E, 56, 16]
    W2_scale_expanded = gemm2_weights_scale.unsqueeze(2).unsqueeze(-1).expand(32, 56, 128, 16, 128).reshape(32, 7168, 2048)
    W2 = W2 * W2_scale_expanded
    
    out = torch.zeros((T, 7168), dtype=torch.float32, device=device)
    
    for t in range(T):
        for i in range(8):
            expert_id = topk_idx[t, i].item()
            weight = topk_weights[t, i].item()
            if expert_id >= 32: continue
            h = A[t:t+1] # [1, 7168]
            w13_expert = W13[expert_id] # [4096, 7168]
            w1, w3 = w13_expert.chunk(2, dim=0) # w1=[2048, 7168], w3=[2048, 7168]
            
            y1 = h @ w1.t() # [1, 2048]
            y3 = h @ w3.t()
            swiglu = F.silu(y3) * y1 # [1, 2048]
            
            w2_expert = W2[expert_id] # [7168, 2048]
            res = swiglu @ w2_expert.t() # [1, 7168]
            out[t] += (res * weight)[0]
            
    return out.to(torch.bfloat16)

def main():
    print("Initializing test matrices...")
    T = 7
    E = 32
    H = 7168
    N1 = 4096
    N2 = 2048
    device = 'cuda'
    
    # Force logits so they pick experts 0 to 31 strongly
    routing_logits = torch.randn((T, 256), dtype=torch.float32, device=device)
    routing_logits[:, :32] += 20.0 # Force mapping to subset
    
    routing_bias = torch.zeros((256,), dtype=torch.bfloat16, device=device)
    hidden_states = torch.randn((T, H), dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    hidden_states_scale = torch.rand((H // 128, T), dtype=torch.float32, device=device) * 0.1
    
    gemm1_weights = torch.randn((E, N1, H), dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    gemm1_weights_scale = torch.rand((E, N1 // 128, H // 128), dtype=torch.float32, device=device) * 0.1
    
    gemm2_weights = torch.randn((E, H, N2), dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    gemm2_weights_scale = torch.rand((E, H // 128, N2 // 128), dtype=torch.float32, device=device) * 0.1
    
    local_expert_offset = 0
    routed_scaling_factor = 1.0
    
    print("Running Phase 2...")
    out_p2 = baseline_phase2(routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, routed_scaling_factor, local_expert_offset)
    
    print("Running Phase 3...")
    out_p3 = torch.empty((T, H), dtype=torch.bfloat16, device=device)
    kernel(routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, local_expert_offset, routed_scaling_factor, out_p3)
    
    print(f"Phase 2 contains NaN: {torch.isnan(out_p2).any().item()}")
    print(f"Phase 3 contains NaN: {torch.isnan(out_p3).any().item()}")
    
    # filter out nans for diff
    diff = (torch.nan_to_num(out_p2) - torch.nan_to_num(out_p3)).abs()
    max_diff = diff.max().item()
    print(f"Max absolute difference between Phase 2 and Phase 3: {max_diff:.4f}")
    
    if max_diff > 1e-1:
        print("Mismatched output examples!")
        idx = torch.nonzero(diff > 1e-1)
        for i in range(min(15, len(idx))):
            r, c = idx[i]
            print(f"  [{r}, {c}]: Phase2={out_p2[r, c].item():.4f} Phase3={out_p3[r, c].item():.4f} Diff={diff[r, c].item():.4f}")

if __name__ == "__main__":
    main()
