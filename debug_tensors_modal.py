import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("fused-moe-debug-tensors")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
    .add_local_dir(PROJECT_ROOT / "solution", remote_path="/root/solution")
    .add_local_dir(PROJECT_ROOT / "scripts", remote_path="/root/scripts")
)

@app.function(image=image, gpu="B200:1", timeout=600)
def debug_run():
    import sys
    sys.path.insert(0, "/root")
    
    import torch.nn.functional as F
    
    def baseline(routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, routed_scaling_factor, local_start):
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
        # Wait, for W2 it's [E, H//128, N_OUT//128] which is [E, 56, 16]
        W2_scale_expanded = gemm2_weights_scale.unsqueeze(2).unsqueeze(-1).expand(32, 56, 128, 16, 128).reshape(32, 7168, 2048)
        W2 = W2 * W2_scale_expanded
        
        out = torch.zeros((T, 7168), dtype=torch.float32, device=device)
        
        for t in range(T):
            for i in range(8):
                expert_id = topk_idx[t, i].item()
                weight = topk_weights[t, i].item()
                if expert_id >= 32: continue
                # hidden = A[t] (7168,)
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

    print("Initializing tensors...")
    torch.manual_seed(0)
    T, E, H, N1, N2, device = 7, 32, 7168, 4096, 2048, 'cuda'
    
    routing_logits = torch.randn((T, 256), dtype=torch.float32, device=device) * 2
    routing_logits[:, :32] += 20.0 # Force mapping to subset
    routing_bias = torch.zeros((256,), dtype=torch.bfloat16, device=device)
    hidden_states = torch.empty((T, H), dtype=torch.float8_e4m3fn, device=device)
    hidden_states.view(torch.int8).random_(-5, 5)
    hidden_states_scale = torch.rand((H // 128, T), dtype=torch.float32, device=device) * 0.1
    
    gemm1_weights = torch.empty((E, N1, H), dtype=torch.float8_e4m3fn, device=device)
    gemm1_weights.view(torch.int8).random_(-5, 5)
    gemm1_weights_scale = torch.rand((E, N1 // 128, H // 128), dtype=torch.float32, device=device) * 0.1
    
    gemm2_weights = torch.empty((E, H, N2), dtype=torch.float8_e4m3fn, device=device)
    gemm2_weights.view(torch.int8).random_(-5, 5)
    gemm2_weights_scale = torch.rand((E, H // 128, N2 // 128), dtype=torch.float32, device=device) * 0.1
    
    print("Running Phase 2...")
    out_p2 = baseline(routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, 1.0, 0)
    
    # Define variables for Phase 3 kernel call
    local_expert_offset = 0
    routed_scaling_factor = 1.0

    print("Running Phase 3...")
    from solution.triton.kernel import kernel
    out_p3 = torch.empty((T, H), dtype=torch.bfloat16, device=device)
    kernel(routing_logits, routing_bias, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale, local_expert_offset, routed_scaling_factor, out_p3)
    
    diff = (out_p2 - out_p3).abs()
    max_diff = diff.max().item()
    
    res = []
    res.append(f"Max abs diff: {max_diff}")
    
    if max_diff > 0:
        idx = torch.nonzero(diff >= max_diff * 0.99) # print top differences
        for i in range(min(5, len(idx))):
            r, c = idx[i]
            res.append(f"Diff at [{r}, {c}]: Phase 2={out_p2[r, c].item():.4f}, Phase 3={out_p3[r, c].item():.4f}, gap={diff[r, c].item():.4f}")
    return "\n".join(res)

@app.local_entrypoint()
def main():
    print(debug_run.remote())
