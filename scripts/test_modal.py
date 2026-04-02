"""
Modal test: send pre-packed solution.json, run benchmark on B200.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("fused-moe-test")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
VOLUME_MOUNT = "/data"
TRACE_SET_PATH = "/data/mlsys26-contest"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy", "cuda-python", "nvidia-cutlass-dsl>=0.1.0")
    .env({"CUTE_DEBUG_PHASE": "6", "CUDA_LAUNCH_BLOCKING": "1"})
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={VOLUME_MOUNT: trace_volume})
def run_test(solution_json: str) -> str:
    """Run benchmark on B200 with pre-packed solution JSON."""
    import torch

    lines = []
    def log(msg):
        lines.append(str(msg))
        print(msg)

    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"CUDA: {torch.cuda.get_device_capability(0)}")
    log(f"PyTorch: {torch.__version__}")

    from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

    solution = Solution.model_validate_json(solution_json)
    log(f"Solution: {solution.name}")

    ts = TraceSet.from_path(TRACE_SET_PATH)
    def_name = solution.definition
    definition = ts.definitions[def_name]
    workloads = ts.workloads.get(def_name, [])
    log(f"Definition: {def_name}")
    log(f"Workloads: {len(workloads)}")

    config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    bench_ts = TraceSet(
        root=ts.root,
        definitions={def_name: definition},
        solutions={def_name: [solution]},
        workloads={def_name: workloads},
        traces={def_name: []},
    )

    benchmark = Benchmark(bench_ts, config)
    result_ts = benchmark.run_all(dump_traces=True)

    traces = result_ts.traces.get(def_name, [])
    log(f"\n=== Results ({len(traces)} traces) ===")

    pass_count = 0
    fail_detail_count = 0
    for trace in traces:
        if trace.evaluation:
            ev = trace.evaluation
            wl_uuid = trace.workload.uuid[:8]
            status = ev.status.value
            entry = f"  {wl_uuid}: {status}"
            if ev.performance:
                entry += f" | {ev.performance.latency_ms:.3f}ms"
                if ev.performance.speedup_factor:
                    entry += f" | {ev.performance.speedup_factor:.2f}x"
            if ev.correctness:
                entry += f" | abs={ev.correctness.max_absolute_error:.2e}, rel={ev.correctness.max_relative_error:.2e}"
            log(entry)
            if status == "PASSED":
                pass_count += 1
            elif ev.log:
                fail_detail_count += 1
                log_text = ev.log.strip()
                if fail_detail_count <= 3:
                    # Show FULL log for first 3 failures
                    log(f"    === FULL ERROR LOG ({wl_uuid}) ===")
                    for line in log_text.split("\n"):
                        log(f"    {line}")
                    log(f"    === END ERROR LOG ===")
                else:
                    log_lines = log_text.split("\n")
                    for line in log_lines[-5:]:
                        log(f"    LOG: {line}")

    log(f"\n=== Summary: {pass_count}/{len(traces)} PASSED ===")
    
    import glob
    import os
    log("\n=== WORKER CRASH LOGS (/tmp/flashinfer_bench) ===")
    for log_file in glob.glob("/tmp/flashinfer_bench/*.log"):
        log(f"\n--- {log_file} ---")
        try:
            with open(log_file, "r") as f:
                content = f.read()
                log(content[-4000:]) # last 4000 chars
        except Exception as e:
            log(f"Error reading {log_file}: {e}")
            
    return "\n".join(lines)


@app.local_entrypoint()
def main():
    import subprocess

    # Pack solution
    print("Packing solution manually bypassed! Already packed.")
    # subprocess.run(
    #     [sys.executable, str(PROJECT_ROOT / "scripts" / "pack_solution_simple.py")],
    #     cwd=str(PROJECT_ROOT), check=True,
    # )

    solution_path = PROJECT_ROOT / "solution.json"
    solution_json = solution_path.read_text(encoding="utf-8")
    print(f"Solution loaded ({len(solution_json)} bytes)")

    print("\nRunning benchmark on Modal B200...")
    result = run_test.remote(solution_json)
    print("\n=== FULL RESULTS ===")
    print(result)

if __name__ == "__main__":
    with app.run():
        main()
