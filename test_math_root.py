import torch
import triton
import triton.language as tl
from solution.triton.kernel import _fused_moe_gemm1_swiglu_kernel

T, H, K, N = 64, 7168, 7168, 4096
device = "cuda"

hidden_states = torch.randint(-5, 5, (T, H), device=device).to(torch.float8_e4m3fn)
hidden_states_scale = torch.rand(H//128, T, dtype=torch.float32, device="cuda")
gemm1_weights = torch.randint(-5, 5, (1, N, H), device=device).to(torch.float8_e4m3fn)
gemm1_weights_scale = torch.rand(1, N//128, H//128, dtype=torch.float32, device="cuda")

# Eager Pytorch
A_fp32 = hidden_states.float() * hidden_states_scale.float().t().unsqueeze(-1).expand(-1, -1, 128).reshape(T, H)
n_blocks, k_blocks = N//128, H//128
w13_fp32 = gemm1_weights[0].float() * gemm1_weights_scale[0].view(n_blocks, 1, k_blocks, 1).expand(-1, 128, -1, 128).reshape(N, H)

# Torch MM
G1 = torch.mm(A_fp32, w13_fp32.t())
# SwiGLU 
I_SIZE = N // 2
C_ref = torch.nn.functional.silu(G1[:, I_SIZE:]) * G1[:, :I_SIZE]

# Triton Kernel
Intermediate_bf16 = torch.zeros((T, 2048), dtype=torch.bfloat16, device=device)
expert_ids = torch.zeros(1, dtype=torch.int32, device=device)
token_ids = torch.arange(T, dtype=torch.int64, device=device)

BLOCK_M = 64
BLOCK_N = 64
BLOCK_K = 128

grid = lambda META: (triton.cdiv(T, BLOCK_M) * triton.cdiv(2048, BLOCK_N),)

_fused_moe_gemm1_swiglu_kernel[grid](
    A_ptr=hidden_states,
    A_scale_ptr=hidden_states_scale,
    B_ptr=gemm1_weights,
    C_ptr=Intermediate_bf16,
    B_scale_ptr=gemm1_weights_scale,
    token_ids_ptr=token_ids,
    expert_ids_ptr=expert_ids,
    num_padded=T, T=T, H=H, N=N, K=K,
    stride_at=hidden_states.stride(0), stride_ah=hidden_states.stride(1),
    stride_as0=hidden_states_scale.stride(0), stride_as1=hidden_states_scale.stride(1),
    stride_be=gemm1_weights.stride(0), stride_bn=gemm1_weights.stride(1), stride_bh=gemm1_weights.stride(2),
    stride_cm=Intermediate_bf16.stride(0), stride_cn=Intermediate_bf16.stride(1),
    stride_bse=gemm1_weights_scale.stride(0), stride_bsn=gemm1_weights_scale.stride(1), stride_bsh=gemm1_weights_scale.stride(2),
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=8
)

diff = torch.abs(C_ref.to(torch.bfloat16) - Intermediate_bf16)
print("Max Diff Tensor 1:", diff.max().item())
print("Ref mean:", C_ref.abs().mean().item(), "Triton mean:", Intermediate_bf16.abs().float().mean().item())
