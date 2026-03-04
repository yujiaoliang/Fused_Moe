"""
Debug: Create a stripped-down Phase 3 kernel that ONLY does GEMM1 and writes acc_1/acc_3 to a debug buffer.
Compare against Phase 2's GEMM1 output.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import solution.triton.kernel as kernel_mod

@triton.jit
def _debug_gemm1_only(
    A_ptr, A_scale_ptr,
    B1_ptr, B1_scale_ptr,
    # Debug output: [num_padded, 4096] bf16 (same as Phase 2 GEMM1 output)
    DBG_ptr,
    token_ids_ptr,
    expert_ids_ptr,
    T, num_padded: tl.constexpr,
    H: tl.constexpr, N1: tl.constexpr, N1_HALF: tl.constexpr,
    stride_at, stride_ah,
    stride_as0, stride_as1,
    stride_b1e, stride_b1n, stride_b1h,
    stride_b1se, stride_b1sn, stride_b1sh,
    BLOCK_M: tl.constexpr, BLOCK_N1: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(num_padded, BLOCK_M)
    num_pid_n1 = tl.cdiv(N1, BLOCK_N1)
    pid_m = pid % num_pid_m
    pid_n1 = pid // num_pid_m

    sorter_block_m = 64
    stride_pid_m = sorter_block_m // BLOCK_M
    expert_id = tl.load(expert_ids_ptr + (pid_m // stride_pid_m))

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = rm < num_padded
    token_idx = tl.load(token_ids_ptr + rm, mask=m_mask, other=T)
    safe_token_idx = tl.where(token_idx < T, token_idx, 0)

    # This block computes output columns [pid_n1*BLOCK_N1 : (pid_n1+1)*BLOCK_N1]
    rn1 = pid_n1 * BLOCK_N1 + tl.arange(0, BLOCK_N1)

    acc = tl.zeros((BLOCK_M, BLOCK_N1), dtype=tl.float32)

    for h_in in range(0, H, BLOCK_H):
        rh_in = h_in + tl.arange(0, BLOCK_H)

        # Load A (same as Phase 3)
        a_ptrs = A_ptr + safe_token_idx[:, None] * stride_at + rh_in[None, :] * stride_ah
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        a_scale_ptrs = A_scale_ptr + (h_in // 128) * stride_as0 + safe_token_idx[:, None] * stride_as1
        a_scale = tl.load(a_scale_ptrs, mask=m_mask[:, None], other=0.0)
        a_fp32 = a.to(tl.float32) * a_scale

        # Load B1
        b1_ptrs = B1_ptr + expert_id * stride_b1e + rn1[None, :] * stride_b1n + rh_in[:, None] * stride_b1h
        b1 = tl.load(b1_ptrs)
        b1_scale_ptrs = B1_scale_ptr + expert_id * stride_b1se + (rn1[None, :] // 128) * stride_b1sn + ((h_in + tl.arange(0, BLOCK_H))[:, None] // 128) * stride_b1sh
        b1_scale = tl.load(b1_scale_ptrs)
        b1_fp32 = b1.to(tl.float32) * b1_scale

        a_dequant = a_fp32.to(tl.bfloat16)
        b1_dequant = b1_fp32.to(tl.bfloat16)
        acc += tl.dot(a_dequant, b1_dequant, out_dtype=tl.float32)

    # Write acc to debug buffer as bf16
    dbg_ptrs = DBG_ptr + rm[:, None] * N1 + rn1[None, :]
    n_mask = rn1 < N1
    tl.store(dbg_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None] & n_mask[None, :])


def run():
    T = 16
    device = torch.device('cuda')

    hidden_states = torch.randint(-10, 10, (T, 7168), dtype=torch.int8, device=device).to(torch.float8_e4m3fn)
    hidden_states_scale = torch.rand((7168 // 128, T), dtype=torch.float32, device=device) * 0.1

    routing_logits = torch.randn((T, 256), dtype=torch.float32, device=device)
    routing_bias = torch.zeros((256,), dtype=torch.float32, device=device)

    gemm1_weights = torch.randint(-10, 10, (32, 4096, 7168), dtype=torch.int8, device=device).to(torch.float8_e4m3fn)
    gemm1_weights_scale = torch.rand((32, 4096 // 128, 7168 // 128), dtype=torch.float32, device=device) * 0.1

    topk_idx, topk_weights = kernel_mod.ds_routing(routing_logits, routing_bias, 1.0)

    # Phase 2 GEMM1 (BLOCK_M=64)
    BLOCK_M = 64
    sorted_token_ids_64, block_expert_ids_64, sorted_weights_64, num_padded_64 = \
        kernel_mod.moe_sort_tokens(topk_idx, topk_weights, 0, BLOCK_M, T, device)

    gemm1_out_p2 = torch.empty((num_padded_64, 4096), dtype=torch.bfloat16, device=device)
    grid1 = lambda META: (triton.cdiv(num_padded_64, META['BLOCK_M']) * (4096 // META['BLOCK_N']),)
    kernel_mod._grouped_gemm_fp8_kernel[grid1](
        hidden_states, gemm1_weights, gemm1_out_p2,
        hidden_states_scale, gemm1_weights_scale,
        sorted_token_ids_64, block_expert_ids_64,
        T, 7168, 4096,
        hidden_states.stride(0), hidden_states.stride(1),
        gemm1_weights.stride(0), gemm1_weights.stride(1), gemm1_weights.stride(2),
        gemm1_out_p2.stride(0), gemm1_out_p2.stride(1),
        hidden_states_scale.stride(0), hidden_states_scale.stride(1),
        gemm1_weights_scale.stride(0), gemm1_weights_scale.stride(1), gemm1_weights_scale.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_N=128, BLOCK_K=128, GROUP_SIZE_M=8,
    )

    # Phase 3 GEMM1 (BLOCK_M=16)
    BLOCK_M_3 = 16
    sorted_token_ids_16, block_expert_ids_16, sorted_weights_16, num_padded_16 = \
        kernel_mod.moe_sort_tokens(topk_idx, topk_weights, 0, BLOCK_M_3, T, device)

    gemm1_out_p3 = torch.empty((num_padded_16, 4096), dtype=torch.bfloat16, device=device)
    grid_dbg = lambda META: (
        triton.cdiv(num_padded_16, META['BLOCK_M']) * triton.cdiv(4096, META['BLOCK_N1']),
    )
    _debug_gemm1_only[grid_dbg](
        A_ptr=hidden_states,
        A_scale_ptr=hidden_states_scale,
        B1_ptr=gemm1_weights,
        B1_scale_ptr=gemm1_weights_scale,
        DBG_ptr=gemm1_out_p3,
        token_ids_ptr=sorted_token_ids_16,
        expert_ids_ptr=block_expert_ids_16,
        T=T, num_padded=num_padded_16,
        H=7168, N1=4096, N1_HALF=2048,
        stride_at=hidden_states.stride(0), stride_ah=hidden_states.stride(1),
        stride_as0=hidden_states_scale.stride(0), stride_as1=hidden_states_scale.stride(1),
        stride_b1e=gemm1_weights.stride(0), stride_b1n=gemm1_weights.stride(1), stride_b1h=gemm1_weights.stride(2),
        stride_b1se=gemm1_weights_scale.stride(0), stride_b1sn=gemm1_weights_scale.stride(1), stride_b1sh=gemm1_weights_scale.stride(2),
        BLOCK_M=BLOCK_M_3, BLOCK_N1=64, BLOCK_H=64,
    )

    # Now unsort both and compare by token
    print(f"Phase 2: num_padded={num_padded_64}, Phase 3: num_padded={num_padded_16}")

    # Compare first few sorted rows directly
    # Both are sorted similarly, find common tokens
    for t in range(min(T, 4)):
        # Find t in Phase 2 sorted list
        p2_rows = (sorted_token_ids_64 == t).nonzero(as_tuple=True)[0]
        p3_rows = (sorted_token_ids_16 == t).nonzero(as_tuple=True)[0]
        if len(p2_rows) > 0 and len(p3_rows) > 0:
            p2_row = p2_rows[0].item()
            p3_row = p3_rows[0].item()
            p2_vals = gemm1_out_p2[p2_row].to(torch.float32)
            p3_vals = gemm1_out_p3[p3_row].to(torch.float32)
            diff = (p2_vals - p3_vals).abs()
            max_d = diff.max().item()
            mean_d = diff.mean().item()
            print(f"Token {t}: Phase2 row={p2_row}, Phase3 row={p3_row}, max_diff={max_d:.4f}, mean_diff={mean_d:.4f}")
            if max_d > 10:
                worst_col = diff.argmax().item()
                print(f"  Worst col={worst_col}: P2={p2_vals[worst_col]:.4f}, P3={p3_vals[worst_col]:.4f}")

if __name__ == '__main__':
    run()
