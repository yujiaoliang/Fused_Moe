"""
GPU profiling on B200 using torch.profiler.
Uses flashinfer_bench API to load the kernel properly (avoids exec() issues with Triton @jit).
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("moe-profiling")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
VOLUME_MOUNT = "/data"
TRACE_SET_PATH = "/data/mlsys26-contest"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={VOLUME_MOUNT: trace_volume})
def run_profile(solution_json: str) -> str:
    """Profile using flashinfer_bench Benchmark API + torch.profiler."""
    import torch

    lines = []
    def log(msg):
        lines.append(str(msg))
        print(msg)

    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"PyTorch: {torch.__version__}")

    from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

    solution = Solution.model_validate_json(solution_json)
    ts = TraceSet.from_path(TRACE_SET_PATH)
    def_name = solution.definition
    definition = ts.definitions[def_name]
    traces = ts.workloads.get(def_name, [])
    workloads = []
    for t in traces:
        if hasattr(t, 'workload'):
            workloads.append(t.workload)
        else:
            workloads.append(t)

    # Write kernel source to temp file and import it
    # (Triton @jit requires functions to be defined in a file, not exec()'d)
    import json as json_mod
    import importlib.util
    import tempfile, os
    
    sol_data = json_mod.loads(solution_json)
    sources = sol_data.get('sources', [])
    if isinstance(sources, list):
        source_code = sources[0] if isinstance(sources[0], str) else sources[0].get('code', sources[0].get('content', str(sources[0])))
    elif isinstance(sources, dict):
        source_code = list(sources.values())[0]
    else:
        source_code = str(sources)
    
    # Write to temp file and import
    tmp_dir = tempfile.mkdtemp()
    kernel_path = os.path.join(tmp_dir, 'kernel.py')
    with open(kernel_path, 'w') as f:
        f.write(source_code)
    
    spec = importlib.util.spec_from_file_location("kernel_module", kernel_path)
    kernel_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel_module)
    kernel_fn = kernel_module.kernel
    
    log(f"Kernel loaded from temp file")

    # Use flashinfer_bench Benchmark to instantiate evaluators
    from flashinfer_bench import Benchmark, BenchmarkConfig
    config = BenchmarkConfig(warmup_runs=0, iterations=1, num_trials=1)
    bench_ts = TraceSet(
        root=ts.root,
        definitions={def_name: definition},
        solutions={def_name: [solution]},
        workloads={def_name: workloads},
        traces={def_name: []},
    )
    benchmark = Benchmark(bench_ts, config)

    # Pick 3 representative workloads
    workloads_sorted = sorted(workloads, key=lambda w: w.uuid)
    indices = [0, len(workloads_sorted) // 2, len(workloads_sorted) - 1]

    for idx in indices:
        wl = workloads_sorted[idx]
        log(f"\n{'='*70}")
        log(f"Workload: {wl.uuid[:8]}")
        
        # Hardcode representative values for T based on index
        # index 0 -> small T
        # index 1 -> medium T
        # index 2 -> large T
        if idx == indices[0]:
            T = 7
        elif idx == indices[1]:
            T = 128
        else:
            T = 4096
        
        # Hardcode the known shapes for this kernel:
        H = 7168
        I_SIZE = 2048
        
        # Manually create random inputs
        routing_logits = torch.randn(T, 256, dtype=torch.float32, device='cuda')
        routing_bias = torch.randn(256, dtype=torch.bfloat16, device='cuda')
        
        A_p = torch.randn(T, H, dtype=torch.float32, device='cuda')
        A_e_amax = A_p.abs().amax(dim=-1, keepdim=True)
        hidden_states = (A_p * 448.0 / A_e_amax.clamp(min=1e-12)).to(torch.float8_e4m3fn)
        hidden_states_scale = (A_e_amax / 448.0).expand(T, H//128).t().contiguous().to(torch.float32)
        
        gemm1_weights = torch.randint(-10, 10, (32, 4096, H), dtype=torch.int8, device='cuda').to(torch.float8_e4m3fn)
        gemm1_weights_scale = torch.rand(32, 32, 56, dtype=torch.float32, device='cuda') + 0.5
        
        gemm2_weights = torch.randint(-10, 10, (32, H, I_SIZE), dtype=torch.int8, device='cuda').to(torch.float8_e4m3fn)
        gemm2_weights_scale = torch.rand(32, 56, 16, dtype=torch.float32, device='cuda') + 0.5
        
        local_expert_offset = torch.tensor(0, dtype=torch.int32, device='cuda')
        routed_scaling_factor = torch.tensor(1.0, dtype=torch.float32, device='cuda')
        
        output = torch.zeros(T, H, dtype=torch.bfloat16, device='cuda')
        
        all_args = [
            routing_logits, routing_bias, 
            hidden_states, hidden_states_scale,
            gemm1_weights, gemm1_weights_scale,
            gemm2_weights, gemm2_weights_scale,
            local_expert_offset, routed_scaling_factor,
            output
        ]
        
        log(f"  T = {T}")
        
        # Warmup (including torch.compile warmup)
        for _ in range(5):
            output.zero_()
            kernel_fn(*all_args)
        torch.cuda.synchronize()

        # Profile
        from torch.profiler import profile, ProfilerActivity
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            for _ in range(5):
                output.zero_()
                kernel_fn(*all_args)
        
        torch.cuda.synchronize()
        
        # Top GPU kernels
        log(f"\n  --- Top 20 GPU Kernels (by CUDA time total) ---")
        table = prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20,
            top_level_events_only=False,
        )
        for line in table.split('\n'):
            log(f"  {line}")
        
        # Category breakdown
        log(f"\n  --- CUDA Time Summary ---")
        events = prof.key_averages()
        # In torch.profiler, average events usually have self_cuda_time_total or cuda_time_total
        # Let's use whatever works
        def get_cuda_time(e):
            return getattr(e, 'self_cuda_time_total', getattr(e, 'cuda_time_total', 0))

        total_cuda = sum(get_cuda_time(e) for e in events if get_cuda_time(e) > 0)
        
        categories = {}
        for e in events:
            t = get_cuda_time(e)
            if t <= 0:
                continue
            name = e.key.lower()
            if any(k in name for k in ['gemm', 'matmul', 'mm', 'cublas']):
                cat = 'GEMM/matmul'
            elif any(k in name for k in ['triton_', 'inductor', 'compiled']):
                cat = 'triton/inductor/compiled'
            elif any(k in name for k in ['copy', 'cast', '_to_', 'convert']):
                cat = 'copy/cast'
            elif any(k in name for k in ['index', 'scatter', 'gather']):
                cat = 'index ops'
            elif any(k in name for k in ['reduce', 'sum', 'mean', 'amax', 'topk']):
                cat = 'reduce'
            elif any(k in name for k in ['sigmoid', 'silu', 'mul', 'add', 'element']):
                cat = 'elementwise'
            elif any(k in name for k in ['zero', 'fill', 'memset']):
                cat = 'memset/zero'
            else:
                cat = 'other'
            categories[cat] = categories.get(cat, 0) + t
        
        log(f"  Total CUDA time (5 iters): {total_cuda/1000:.2f}ms")
        for cat, t in sorted(categories.items(), key=lambda x: -x[1]):
            pct = 100 * t / total_cuda if total_cuda > 0 else 0
            log(f"    {cat:25s}: {t/1000:8.2f}ms ({pct:5.1f}%)")
        
        log(f"\n  --- Memory ---")
        log(f"  Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    return "\n".join(lines)


@app.local_entrypoint()
def main():
    import subprocess

    print("Packing solution...")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "pack_solution_simple.py")],
        cwd=str(PROJECT_ROOT), check=True,
    )

    solution_path = PROJECT_ROOT / "solution.json"
    solution_json = solution_path.read_text(encoding="utf-8")
    print(f"Solution loaded ({len(solution_json)} bytes)")

    print("\nRunning profiling on Modal B200...")
    result = run_profile.remote(solution_json)
    
    # Write to file explicitly to avoid console encoding crashes
    out_path = PROJECT_ROOT / "profile_output.txt"
    out_path.write_text(result, encoding="utf-8")
    print(f"\n=== PROFILING RESULTS SAVED TO {out_path} ===")
