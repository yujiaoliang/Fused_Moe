"""Detailed correctness debug: step-by-step comparison."""
import torch
import sys
sys.path.insert(0, 'solution/triton')
import kernel as fused_moe
from flashinfer_bench.data import TraceSet
from flashinfer_bench.bench.evaluators import resolve_evaluator
from flashinfer_bench.bench.config import BenchmarkConfig

def run_test():
    torch.manual_seed(42)
    device = "cuda:0"

    ts = TraceSet.from_path("D:/Research/mlsys_note/mlsys26-contest")
    def_name = 'moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048'
    defn = ts.definitions[def_name]

    target_uuid = 'b8f4f012-a32e-4356-b4e1-7665b3d598af'
    wl = None
    for item in ts.workloads.get(def_name, []):
        if item.workload.uuid == target_uuid:
            wl = item.workload
            break

    cfg = BenchmarkConfig()
    evaluator_cls = resolve_evaluator(defn)
    baseline = evaluator_cls.build_baseline(defn, wl, cfg, device, ts.root)
    
    inp = baseline.inputs[0]
    ref_out = baseline.outputs[0]['output']
    
    T = inp['routing_logits'].shape[0]
    local_start = int(inp['local_expert_offset'])
    scale_factor = float(inp['routed_scaling_factor'])
    
    print(f"T={T}, local_start={local_start}, scale_factor={scale_factor}")
    
    # Step 1: Compare routing
    print("\n=== Step 1: Routing ===")
    topk_idx, topk_weights = fused_moe.ds_routing(
        inp['routing_logits'], inp['routing_bias'], scale_factor
    )
    print(f"topk_idx: {topk_idx}")
    print(f"topk_weights: {topk_weights}")
    
    # Check which local experts are selected
    local_end = local_start + 32
    local_mask = (topk_idx >= local_start) & (topk_idx < local_end)
    print(f"\nLocal expert mask (any per token): {local_mask.any(dim=1)}")
    num_local = local_mask.sum().item()
    print(f"Total local expert assignments: {num_local}")
    
    if num_local == 0:
        print("No local experts selected! Output should be all zeros.")
        print(f"Ref output non-zero count: {(ref_out.float().abs() > 1e-6).sum().item()}")
        return
    
    # Step 2: Token sorting
    print("\n=== Step 2: Token Sorting ===")
    BLOCK_M = 64
    sorted_token_ids, block_expert_ids, sorted_weights, num_padded = \
        fused_moe.moe_sort_tokens(topk_idx, topk_weights, local_start, BLOCK_M, T, device)
    
    if sorted_token_ids is None:
        print("No tokens sorted!")
        return
    
    print(f"num_padded: {num_padded}")
    print(f"sorted_token_ids: {sorted_token_ids}")
    print(f"block_expert_ids: {block_expert_ids}")
    print(f"sorted_weights: {sorted_weights}")
    
    # Step 3: Compare per-token error
    print("\n=== Step 3: Full kernel output comparison ===")
    our_out = fused_moe.kernel(**inp)
    
    ref_f = ref_out.float()
    our_f = our_out.float()
    
    for t in range(T):
        abs_err_t = (ref_f[t] - our_f[t]).abs()
        max_ae = abs_err_t.max().item()
        mean_ae = abs_err_t.mean().item()
        ref_norm = ref_f[t].abs().mean().item()
        our_norm = our_f[t].abs().mean().item()
        print(f"  Token {t}: max_abs_err={max_ae:.4f}, mean_abs_err={mean_ae:.4f}, "
              f"ref_norm={ref_norm:.4f}, our_norm={our_norm:.4f}")
    
    # Cleanup
    del baseline
    torch.cuda.empty_cache()

if __name__ == "__main__":
    run_test()
