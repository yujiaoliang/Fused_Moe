import torch
import triton
import triton.language as tl
from solution.triton.kernel import kernel

torch.manual_seed(42)

T, H = 14, 7168
K = 2048
E = 4

device = "cuda"

routing_logits = torch.zeros(T, 256, dtype=torch.float32, device=device)
routing_logits[:, :16] = 100.0
routing_bias = torch.randn(256, dtype=torch.bfloat16, device=device)
hidden_states = torch.randint(-5, 5, (T, H), device=device).to(torch.float8_e4m3fn)
hidden_states_scale = torch.rand(H//128, T, dtype=torch.float32, device=device)

gemm1_weights = torch.randint(-5, 5, (32, 4096, H), device=device).to(torch.float8_e4m3fn)
gemm1_weights_scale = torch.rand(32, 32, 56, dtype=torch.float32, device=device)

gemm2_weights = torch.randint(-5, 5, (32, H, K), device=device).to(torch.float8_e4m3fn)
gemm2_weights_scale = torch.rand(32, 56, 16, dtype=torch.float32, device=device)

output_triton = torch.zeros(T, H, dtype=torch.bfloat16, device=device)
kernel(
    routing_logits, routing_bias,
    hidden_states, hidden_states_scale,
    gemm1_weights, gemm1_weights_scale,
    gemm2_weights, gemm2_weights_scale,
    0, 1.0, output_triton
)

# NOW WRITE REFERENCE PURE EAGER TO COMPARE:
from solution.triton.kernel import ds_routing, moe_sort_tokens
topk_idx, topk_weights = ds_routing(routing_logits, routing_bias, 1.0)
sorted_token_ids, block_expert_ids, sorted_weights, num_padded = moe_sort_tokens(topk_idx, topk_weights, 0, 64, T, device)

A_fp32 = hidden_states.float()
A_scale = hidden_states_scale.float().t() # [T, 56]
A_fp32 = A_fp32 * A_scale.unsqueeze(-1).expand(-1, -1, 128).reshape(T, H)

accum = torch.zeros((T, H), dtype=torch.float32, device=device)
num_m_blocks = num_padded // 64
experts_cpu = block_expert_ids[:num_m_blocks].cpu().tolist()

seg_start = 0
for seg_end in range(1, num_m_blocks + 1):
    if seg_end == num_m_blocks or experts_cpu[seg_end] != experts_cpu[seg_start]:
        le = experts_cpu[seg_start]
        start_pos = seg_start * 64
        end_pos = seg_end * 64

        expert_token_ids = sorted_token_ids[start_pos:end_pos]
        expert_weights = sorted_weights[start_pos:end_pos]
        valid = expert_token_ids < T
        tok_ids = expert_token_ids[valid]
        tok_weights = expert_weights[valid]

        if tok_ids.numel() > 0:
            A_e = A_fp32.index_select(0, tok_ids)
            # w13
            w13_fp8 = gemm1_weights[le]
            s13 = gemm1_weights_scale[le]
            n13, k13 = s13.shape
            w13 = w13_fp8.float() * s13.view(n13, 1, k13, 1).expand(n13, 128, k13, 128).reshape(4096, H)
            G1 = torch.mm(A_e, w13.t())
            C = torch.nn.functional.silu(G1[:, 2048:]) * G1[:, :2048]
            
            # w2
            w2_fp8 = gemm2_weights[le]
            s2 = gemm2_weights_scale[le]
            n2, k2 = s2.shape
            w2 = w2_fp8.float() * s2.view(n2, 1, k2, 1).expand(n2, 128, k2, 128).reshape(H, 2048)
            O = torch.mm(C, w2.t())
            
            weighted_out = O * tok_weights.unsqueeze(1)
            accum.index_add_(0, tok_ids, weighted_out)

        seg_start = seg_end

output_ref = accum.to(torch.bfloat16)
diff = torch.abs(output_ref - output_triton)
print("max diff:", diff.max().item())
print("ref mean:", output_ref.abs().float().mean().item())
print("trit mean:", output_triton.abs().float().mean().item())
