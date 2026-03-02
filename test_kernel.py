"""Direct correctness test with matched_ratio computation."""
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
    print(f"T={T}")
    print(f"Inputs:")
    for k, v in inp.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: {v}")
    
    # Run our kernel
    print("\nRunning our kernel...")
    our_out = fused_moe.kernel(**inp)
    
    print(f"Ref output: {ref_out.shape} {ref_out.dtype}")
    print(f"Our output: {our_out.shape} {our_out.dtype}")
    
    # Compute error stats (same as benchmark evaluator)
    x = our_out.float()
    y = ref_out.float()
    eps = 1e-8
    abs_error = torch.abs(x - y)
    rel_error = abs_error / (torch.abs(y) + eps)
    
    atol, rtol = 0.01, 0.01
    exceeds_mask = (abs_error > atol) & (rel_error > rtol)
    total = abs_error.numel()
    exceeds_count = exceeds_mask.sum().item()
    matched_ratio = 1.0 - (exceeds_count / total)
    
    print(f"\n=== Error Stats ===")
    print(f"Max absolute error: {abs_error.max().item():.4f}")
    print(f"Max relative error: {rel_error.max().item():.4f}")
    print(f"Mean absolute error: {abs_error.mean().item():.4f}")
    print(f"Mean relative error: {rel_error.mean().item():.4f}")
    print(f"\nExceeds tolerance: {exceeds_count}/{total} elements")
    print(f"Matched ratio: {matched_ratio:.6f} (need >= 0.95)")
    print(f"PASS: {matched_ratio >= 0.95}")
    
    # Per-token breakdown
    print(f"\n=== Per-token breakdown ===")
    for t in range(T):
        abs_t = abs_error[t]
        rel_t = rel_error[t]
        exc_t = ((abs_t > atol) & (rel_t > rtol)).sum().item()
        total_t = abs_t.numel()
        ratio_t = 1.0 - exc_t / total_t
        print(f"  Token {t}: max_abs={abs_t.max().item():.2f}, max_rel={rel_t.max().item():.4f}, "
              f"exceeds={exc_t}/{total_t}, matched={ratio_t:.4f}")

    del baseline
    torch.cuda.empty_cache()

if __name__ == "__main__":
    run_test()
