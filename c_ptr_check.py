import torch
from solution.triton.kernel import ds_routing, moe_sort_tokens, BLOCK_M

torch.manual_seed(0)
T, H = 7, 7168
K = 2048

routing_logits = torch.randn(T, 256, dtype=torch.float32, device="cuda")
routing_logits[:, 32:] = -10000.0
routing_bias = torch.randn(256, dtype=torch.bfloat16, device="cuda")

device = routing_logits.device
topk_idx, topk_weights = ds_routing(routing_logits, routing_bias, 1.0)
sorted_token_ids, block_expert_ids, sorted_weights, num_padded = moe_sort_tokens(topk_idx, topk_weights, 0, BLOCK_M, T, device)

output_fp32 = torch.zeros((T, 7168), dtype=torch.float32, device=device)
stride_ct = output_fp32.stride(0)
stride_ch = output_fp32.stride(1)

print(f"num_padded: {num_padded}")
print(f"T: {T}")
print(f"BLOCK_M: {BLOCK_M}")
print(f"stride_ct: {stride_ct}, stride_ch: {stride_ch}")
print(f"output_fp32 size elements: {output_fp32.numel()}")

max_rh = 7167
max_safe_token_idx = T - 1 if T > 0 else 0
print(f"max offset mathematically: {max_safe_token_idx * stride_ct + max_rh * stride_ch}")
