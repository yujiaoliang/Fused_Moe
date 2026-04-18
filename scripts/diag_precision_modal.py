"""
Modal diagnostic: compare CuTe vs Pure Triton for T=14107.
Uses Benchmark to run both implementations and compare element-by-element.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("moe-precision-diag")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
VOLUME_MOUNT = "/data"
TRACE_SET_PATH = "/data/mlsys26-contest"

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:20260401-2c675fb")
    .entrypoint([])
    .pip_install("flashinfer-bench")
    .env({"CUDA_HOME": "/usr/local/cuda"})
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={VOLUME_MOUNT: trace_volume})
def diagnose(solution_json: str) -> str:
    import json
    import torch
    import importlib.util
    import tempfile, os

    lines = []
    def log(msg):
        lines.append(str(msg))
        print(msg)

    log(f"GPU: {torch.cuda.get_device_name(0)}")

    # Parse solution and write sources to tmp
    solution = json.loads(solution_json)
    sources = {s['path']: s['content'] for s in solution['sources']}
    tmpdir = tempfile.mkdtemp()
    for path, content in sources.items():
        fpath = os.path.join(tmpdir, path)
        with open(fpath, 'w') as f:
            f.write(content)
    sys.path.insert(0, tmpdir)

    # Load the Benchmark API just to get the workload data
    from flashinfer_bench import Solution, TraceSet
    sol = Solution.model_validate_json(solution_json)
    ts = TraceSet.from_path(TRACE_SET_PATH)
    def_name = sol.definition
    definition = ts.definitions[def_name]

    # List all workloads to find T=14107
    workloads = ts.workloads.get(def_name, [])
    log(f"Workloads: {len(workloads)}")

    # Use the benchmark runner to load workload args
    # Workload objects have 'input_tensors' method or similar
    # Let's just introspect the workload object
    target_wl = None
    for wl in workloads:
        wl_uuid = wl.workload.uuid if hasattr(wl, 'workload') else str(wl)
        if '58a34f27' in str(wl_uuid):
            target_wl = wl
            break

    if target_wl is None:
        log("Cannot find T=14107 workload, using alternative approach")
        # Just run both implementations with the full benchmark
        # and compare
    
    # Alternative: load both implementations and run the full solution
    log("\n=== Loading implementations ===")
    
    # Import pure triton
    spec_pt = importlib.util.spec_from_file_location("diag_pure_triton", os.path.join(tmpdir, "pure_triton_impl.py"))
    pure_triton = importlib.util.module_from_spec(spec_pt)
    spec_pt.loader.exec_module(pure_triton)
    log("Pure Triton loaded")

    # Import hybrid CuTe
    spec_hybrid = importlib.util.spec_from_file_location("diag_triton_impl", os.path.join(tmpdir, "triton_impl.py"))
    triton_impl = importlib.util.module_from_spec(spec_hybrid)
    spec_hybrid.loader.exec_module(triton_impl)
    log("Hybrid CuTe loaded")

    # Synthesize T=14107 workload data (matching the real MoE dimensions)
    T = 14107
    device = 'cuda'
    log(f"\n=== Synthesizing T={T} workload ===")
    
    torch.manual_seed(42)  # deterministic
    routing_logits = torch.randn(T, 256, dtype=torch.float32, device=device)
    routing_bias = torch.randn(256, dtype=torch.bfloat16, device=device)
    hidden_states = torch.randn(T, 7168, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    hidden_states_scale = torch.randn(56, T, dtype=torch.float32, device=device).abs() * 0.01 + 0.001
    gemm1_weights = torch.randn(32, 4096, 7168, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    gemm1_weights_scale = torch.randn(32, 32, 56, dtype=torch.float32, device=device).abs() * 0.01 + 0.001
    gemm2_weights = torch.randn(32, 7168, 2048, dtype=torch.float32, device=device).to(torch.float8_e4m3fn)
    gemm2_weights_scale = torch.randn(32, 56, 16, dtype=torch.float32, device=device).abs() * 0.01 + 0.001
    local_expert_offset = torch.tensor(0, dtype=torch.int32, device=device)
    routed_scaling_factor = torch.tensor(1.0, dtype=torch.float32, device=device)
    
    args = [routing_logits, routing_bias, hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
            local_expert_offset, routed_scaling_factor]

    # Run pure Triton
    log("\n=== Running Pure Triton ===")
    output_pt = torch.empty((T, 7168), dtype=torch.bfloat16, device=device)
    pure_triton.kernel(*args, output=output_pt)
    torch.cuda.synchronize()
    log(f"Pure Triton: mean={output_pt.float().abs().mean():.4f}, max={output_pt.float().abs().max():.4f}")

    # Run CuTe hybrid  
    log("\n=== Running CuTe Hybrid ===")
    output_cute = torch.empty((T, 7168), dtype=torch.bfloat16, device=device)
    triton_impl.kernel(*args, output=output_cute)
    torch.cuda.synchronize()
    log(f"CuTe: mean={output_cute.float().abs().mean():.4f}, max={output_cute.float().abs().max():.4f}")

    # Compare
    diff = (output_cute.float() - output_pt.float()).abs()
    log(f"\n=== Comparison (CuTe vs Pure Triton) ===")
    log(f"Max abs error: {diff.max():.4f}")
    log(f"Mean abs error: {diff.mean():.6f}")
    log(f"Median abs error: {diff.median():.6f}")

    # Tolerance check
    abs_ok = diff <= 1.0
    denom = output_pt.float().abs().clamp(min=1e-10)
    rel_err = diff / denom
    rel_ok = rel_err <= 0.3
    either_ok = abs_ok | rel_ok
    matched_ratio = either_ok.float().mean().item()
    log(f"Matched ratio (vs pure triton): {matched_ratio:.6f} (need >= 0.9)")
    log(f"Elements failing: {(~either_ok).sum().item()} / {either_ok.numel()}")

    # Error distribution
    for threshold in [1, 10, 100, 1000, 10000, 100000]:
        count = (diff > threshold).sum().item()
        log(f"  abs_err > {threshold:>8d}: {count:>8d} ({100*count/diff.numel():.4f}%)")

    # Worst 10 elements
    flat_diff = diff.flatten()
    worst_vals, worst_indices = flat_diff.topk(min(10, flat_diff.numel()))
    for val, idx in zip(worst_vals, worst_indices):
        row = idx.item() // 7168
        col = idx.item() % 7168
        pt_val = output_pt[row, col].float().item()
        cute_val = output_cute[row, col].float().item()
        log(f"  Worst: [{row},{col}] pt={pt_val:.2f} cute={cute_val:.2f} diff={val.item():.2f}")

    return "\n".join(lines)


@app.local_entrypoint()
def main():
    solution_path = PROJECT_ROOT / "solution.json"
    solution_json = solution_path.read_text(encoding="utf-8")
    print(f"Solution loaded ({len(solution_json)} bytes)")
    result = diagnose.remote(solution_json)
    print("\n=== DIAGNOSTIC RESULTS ===")
    print(result)
