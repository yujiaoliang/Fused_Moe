"""
Ultra-minimal: Compare Phase2 vs Phase3 GEMM1 with scales=1.0 to test raw matmul.
"""
import torch
import triton
import triton.language as tl
import solution.triton.kernel as kernel_mod

@triton.jit
def _debug_gemm1_noscale(
    A_ptr, B1_ptr, DBG_ptr,
    token_ids_ptr, expert_ids_ptr,
    T, num_padded: tl.constexpr,
    H: tl.constexpr, N1: tl.constexpr,
    stride_at, stride_ah,
    stride_b1e, stride_b1n, stride_b1h,
    BLOCK_M: tl.constexpr, BLOCK_N1: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(num_padded, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n1 = pid // num_pid_m

    sorter_block_m = 64
    stride_pid_m = sorter_block_m // BLOCK_M
    expert_id = tl.load(expert_ids_ptr + (pid_m // stride_pid_m))

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = rm < num_padded
    token_idx = tl.load(token_ids_ptr + rm, mask=m_mask, other=T)
    safe_token_idx = tl.where(token_idx < T, token_idx, 0)

    rn1 = pid_n1 * BLOCK_N1 + tl.arange(0, BLOCK_N1)
    acc = tl.zeros((BLOCK_M, BLOCK_N1), dtype=tl.float32)

    for h_in in range(0, H, BLOCK_H):
        rh_in = h_in + tl.arange(0, BLOCK_H)
        
        # Load A as fp32 (no scale)
        a_ptrs = A_ptr + safe_token_idx[:, None] * stride_at + rh_in[None, :] * stride_ah
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        a_bf16 = a.to(tl.float32).to(tl.bfloat16)
        
        # Load B1 as fp32 (no scale) 
        b1_ptrs = B1_ptr + expert_id * stride_b1e + rn1[None, :] * stride_b1n + rh_in[:, None] * stride_b1h
        b1 = tl.load(b1_ptrs)
        b1_bf16 = b1.to(tl.float32).to(tl.bfloat16)
        
        acc += tl.dot(a_bf16, b1_bf16, out_dtype=tl.float32)

    dbg_ptrs = DBG_ptr + rm[:, None] * N1 + rn1[None, :]
    n_mask = rn1 < N1
    tl.store(dbg_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


def run():
    T = 16
    device = torch.device('cuda')

    hidden_states = torch.randint(-10, 10, (T, 7168), dtype=torch.int8, device=device).to(torch.float8_e4m3fn)
    gemm1_weights = torch.randint(-10, 10, (32, 4096, 7168), dtype=torch.int8, device=device).to(torch.float8_e4m3fn)
    
    # Use uniform scales of 1.0
    hidden_states_scale = torch.ones((7168 // 128, T), dtype=torch.float32, device=device)
    gemm1_weights_scale = torch.ones((32, 4096 // 128, 7168 // 128), dtype=torch.float32, device=device)

    routing_logits = torch.randn((T, 256), dtype=torch.float32, device=device)
    routing_bias = torch.zeros((256,), dtype=torch.float32, device=device)
    topk_idx, topk_weights = kernel_mod.ds_routing(routing_logits, routing_bias, 1.0)

    # Phase 2 GEMM1 (BLOCK_M=64, scale=1.0 so dequant is just fp8->fp32)
    BLOCK_M = 64
    sorted_64, experts_64, weights_64, np_64 = \
        kernel_mod.moe_sort_tokens(topk_idx, topk_weights, 0, BLOCK_M, T, device)
    gemm1_p2 = torch.empty((np_64, 4096), dtype=torch.bfloat16, device=device)
    grid1 = lambda META: (triton.cdiv(np_64, META['BLOCK_M']) * (4096 // META['BLOCK_N']),)
    kernel_mod._grouped_gemm_fp8_kernel[grid1](
        hidden_states, gemm1_weights, gemm1_p2,
        hidden_states_scale, gemm1_weights_scale,
        sorted_64, experts_64,
        T, 7168, 4096,
        hidden_states.stride(0), hidden_states.stride(1),
        gemm1_weights.stride(0), gemm1_weights.stride(1), gemm1_weights.stride(2),
        gemm1_p2.stride(0), gemm1_p2.stride(1),
        hidden_states_scale.stride(0), hidden_states_scale.stride(1),
        gemm1_weights_scale.stride(0), gemm1_weights_scale.stride(1), gemm1_weights_scale.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=128, BLOCK_K=128, GROUP_SIZE_M=8,
    )

    # Phase 3 GEMM1 (BLOCK_M=16, no scale)
    BLOCK_M_3 = 16
    sorted_16, experts_16, weights_16, np_16 = \
        kernel_mod.moe_sort_tokens(topk_idx, topk_weights, 0, BLOCK_M_3, T, device)
    gemm1_p3 = torch.empty((np_16, 4096), dtype=torch.bfloat16, device=device)
    grid_dbg = lambda META: (
        triton.cdiv(np_16, META['BLOCK_M']) * triton.cdiv(4096, META['BLOCK_N1']),
    )
    _debug_gemm1_noscale[grid_dbg](
        A_ptr=hidden_states, B1_ptr=gemm1_weights, DBG_ptr=gemm1_p3,
        token_ids_ptr=sorted_16, expert_ids_ptr=experts_16,
        T=T, num_padded=np_16,
        H=7168, N1=4096,
        stride_at=hidden_states.stride(0), stride_ah=hidden_states.stride(1),
        stride_b1e=gemm1_weights.stride(0), stride_b1n=gemm1_weights.stride(1), stride_b1h=gemm1_weights.stride(2),
        BLOCK_M=BLOCK_M_3, BLOCK_N1=64, BLOCK_H=64,
    )

    # Compare token-by-token
    for t in range(min(T, 4)):
        p2_rows = (sorted_64 == t).nonzero(as_tuple=True)[0]
        p3_rows = (sorted_16 == t).nonzero(as_tuple=True)[0]
        if len(p2_rows) > 0 and len(p3_rows) > 0:
            p2_v = gemm1_p2[p2_rows[0]].to(torch.float32)
            p3_v = gemm1_p3[p3_rows[0]].to(torch.float32)
            diff = (p2_v - p3_v).abs()
            max_d = diff.max().item()
            mean_d = diff.mean().item()
            print(f"Token {t}: max_diff={max_d:.4f}, mean_diff={mean_d:.4f}")
            if max_d > 1.0:
                worst = diff.argmax().item()
                print(f"  col={worst}: P2={p2_v[worst]:.4f}, P3={p3_v[worst]:.4f}")

    # Also compare PyTorch reference
    A_f32 = hidden_states.to(torch.float32)
    B_f32 = gemm1_weights.to(torch.float32)
    # Token 0, first expert route 
    for t in range(min(T, 2)):
        expert = topk_idx[t, 0].item()
        if expert >= 32: continue
        pt_result = A_f32[t:t+1] @ B_f32[expert].t()
        
        p3_rows = (sorted_16 == t).nonzero(as_tuple=True)[0]
        if len(p3_rows) > 0:
            p3_v = gemm1_p3[p3_rows[0]].to(torch.float32)
            diff_pt = (pt_result[0] - p3_v).abs()
            print(f"Token {t} vs PyTorch: max_diff={diff_pt.max():.4f}")

if __name__ == '__main__':
    run()
