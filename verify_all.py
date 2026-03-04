import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet
from scripts.pack_solution_simple import pack_solution

def get_trace_set_path() -> str:
    return os.environ.get("FIB_DATASET_PATH")

def main():
    print("Packing solution...")
    solution_path = pack_solution()
    solution = Solution.model_validate_json(solution_path.read_text(encoding="utf-8"))
    
    config = BenchmarkConfig(warmup_runs=1, iterations=1, num_trials=1, use_isolated_runner=False)
    
    trace_set_path = get_trace_set_path()
    trace_set = TraceSet.from_path(trace_set_path)
    
    definition = trace_set.definitions[solution.definition]
    workloads = trace_set.workloads.get(solution.definition, [])
    
    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )
    
    print("Running quick local verification...")
    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)
    
    traces = result_trace_set.traces.get(definition.name, [])
    
    failed = []
    
    for trace in traces:
        if trace.evaluation:
            status = trace.evaluation.status.value
            if status != "PASSED":
                print(f"FAILED Workload: {trace.workload.uuid[:8]}")
                print(f"  Reason: {status}")
                if trace.evaluation.log:
                    print(trace.evaluation.log[:200])
                if trace.evaluation.correctness:
                    print(f"  Abs Error: {trace.evaluation.correctness.max_absolute_error:.2e}")
                failed.append(trace.workload.uuid)
            else:
                print(f"PASSED Workload: {trace.workload.uuid[:8]}")
    
    print(f"\n{len(failed)} failed out of {len(traces)}")

if __name__ == "__main__":
    main()
