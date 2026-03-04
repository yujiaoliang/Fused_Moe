import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("fused-moe-check")
trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
VOLUME_MOUNT = "/data"
TRACE_SET_PATH = "/data/mlsys26-contest"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)

@app.function(image=image, gpu="B200:1", timeout=3600, volumes={VOLUME_MOUNT: trace_volume})
def run_check(solution_json: str) -> str:
    import torch
    lines = []
    def log(msg): lines.append(str(msg))

    from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
    solution = Solution.model_validate_json(solution_json)
    ts = TraceSet.from_path(TRACE_SET_PATH)
    
    def_name = solution.definition
    definition = ts.definitions[def_name]
    workloads = ts.workloads.get(def_name, [])
    
    config = BenchmarkConfig(warmup_runs=1, iterations=1, num_trials=1, use_isolated_runner=False)
    bench_ts = TraceSet(
        root=ts.root,
        definitions={def_name: definition},
        solutions={def_name: [solution]},
        workloads={def_name: workloads},
        traces={def_name: []},
    )
    benchmark = Benchmark(bench_ts, config)
    result_ts = benchmark.run_all(dump_traces=False)
    
    traces = result_ts.traces.get(def_name, [])
    
    for trace in traces:
        if trace.evaluation:
            status = trace.evaluation.status.value
            entry = f"{trace.workload.uuid[:8]}: {status}"
            if trace.evaluation.correctness:
                entry += f" | abs={trace.evaluation.correctness.max_absolute_error:.2e}, rel={trace.evaluation.correctness.max_relative_error:.2e}"
            log(entry)
            if status != "PASSED" and trace.evaluation.log:
                log(f"  LOG: {trace.evaluation.log}")
    
    return "\n".join(lines)

@app.local_entrypoint()
def main():
    import json
    solution_path = PROJECT_ROOT / "solution.json"
    print("\nRunning check on Modal B200...")
    result = run_check.remote(solution_path.read_text(encoding="utf-8"))
    print("\n=== RESULTS ===")
    print(result)

if __name__ == "__main__":
    with app.run():
        main()
