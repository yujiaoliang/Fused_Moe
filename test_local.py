import torch
from solution.triton.kernel import kernel

torch.manual_seed(0)

T, H = 7, 7168
256
E = 64
K = 2048

routing_logits = torch.randn(T, 256, dtype=torch.float32, device="cuda")
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
