import torch
from solution.triton.kernel import kernel

torch.manual_seed(0)

T, H = 7, 7168
256
E = 64
K = 2048

routing_logits = torch.randn(T, 256, dtype=torch.float32, device="cuda")
# Constraint to simulate local experts bound to GPU memory limits dynamically
routing_logits[:, 32:] = -10000.0
routing_bias = torch.randn(256, dtype=torch.bfloat16, device="cuda")
hidden_states = torch.randn(T, H, device="cuda").to(torch.float8_e4m3fn)
hidden_states_scale = torch.rand(56, T, dtype=torch.float32, device="cuda")

gemm1_weights = torch.randn(32, 4096, H, device="cuda").to(torch.float8_e4m3fn)
gemm1_weights_scale = torch.rand(32, 32, 56, dtype=torch.float32, device="cuda")

gemm2_weights = torch.randn(32, H, K, device="cuda").to(torch.float8_e4m3fn)
gemm2_weights_scale = torch.rand(32, 56, 16, dtype=torch.float32, device="cuda")

local_expert_offset = 0
routed_scaling_factor = 1.0

output = torch.zeros(T, H, dtype=torch.bfloat16, device="cuda")

print("Running fused kernel...")
kernel(
    routing_logits, routing_bias,
    hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale,
    gemm2_weights, gemm2_weights_scale,
    local_expert_offset, routed_scaling_factor, output
)
print("Done! output norm:", output.norm().item())

# Correctness check against PyTorch
def pytorch_baseline(routing_logits, routing_bias, A, A_scale, W13, W13_scale, W2, W2_scale):
    T = A.shape[0]
    out = torch.zeros(T, H, dtype=torch.float32, device="cuda")
    top2_weights, top2_idx = torch.topk(routing_logits, 2, dim=1) # simplify
    for t in range(T):
        for k in range(2):
            e = top2_idx[t, k].item()
            if e >= 32: continue # Since mock uses 32 locals
            w = top2_weights[t, k].item()
            
            # GEMM1
            a = A[t].float() * A_scale[:, t].repeat_interleave(128)
            w13 = W13[e].float() * W13_scale[e].repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
            inter = torch.matmul(w13, a)
            
            # SwiGLU
            up = inter[:2048]
            gate = inter[2048:]
            swiglu = torch.nn.functional.silu(gate) * up
            
            # GEMM2
            w2 = W2[e].float() * W2_scale[e].repeat_interleave(16, dim=0).repeat_interleave(128, dim=1)
            out[t] += torch.matmul(w2, swiglu) * w
    return out.to(torch.bfloat16)

# The mock weights are initialized, wait flashinfer routes up to 256. 
# We'll use the actual local test suite directly from `test_modal` script or `run_bench`
