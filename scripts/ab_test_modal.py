"""
A/B Test: Run baseline vs experiment solutions back-to-back on the SAME Modal B200 instance.
Eliminates environment drift by comparing within a single GPU session.

Usage:
  1. Pack baseline:     python scripts/pack_solution_simple.py && copy solution.json solution_a.json
  2. Make changes to kernel.py
  3. Pack experiment:   python scripts/pack_solution_simple.py && copy solution.json solution_b.json
  4. Run A/B test:      python -m modal run scripts/ab_test_modal.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("fused-moe-ab-test")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
VOLUME_MOUNT = "/data"
TRACE_SET_PATH = "/data/mlsys26-contest"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy", "cuda-python", "nvidia-cutlass-dsl>=0.1.0")
    .env({"CUTE_DEBUG_PHASE": "6", "CUDA_LAUNCH_BLOCKING": "1"})
)


@app.function(image=image, gpu="B200:1", timeout=7200, volumes={VOLUME_MOUNT: trace_volume})
def run_ab_test(solution_a_json: str, solution_b_json: str, label_a: str = "A (baseline)", label_b: str = "B (experiment)") -> str:
    """Run two solutions back-to-back on the SAME B200 instance."""
    import torch

    lines = []
    def log(msg):
        lines.append(str(msg))
        print(msg)

    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"CUDA: {torch.cuda.get_device_capability(0)}")
    log(f"PyTorch: {torch.__version__}")

    from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

    ts = TraceSet.from_path(TRACE_SET_PATH)
    config = BenchmarkConfig(warmup_runs=3, iterations=100, num_trials=5)

    results = {}

    for label, sol_json in [(label_a, solution_a_json), (label_b, solution_b_json)]:
        log(f"\n{'='*60}")
        log(f"Running: {label}")
        log(f"{'='*60}")

        solution = Solution.model_validate_json(sol_json)
        def_name = solution.definition
        definition = ts.definitions[def_name]
        workloads = ts.workloads.get(def_name, [])
        log(f"Solution: {solution.name} | Workloads: {len(workloads)}")

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
        run_results = {}

        for trace in traces:
            if trace.evaluation:
                ev = trace.evaluation
                wl_uuid = trace.workload.uuid[:8]
                status = ev.status.value
                latency = ev.performance.latency_ms if ev.performance else None
                speedup = ev.performance.speedup_factor if ev.performance else None
                run_results[wl_uuid] = {
                    "status": status,
                    "latency": latency,
                    "speedup": speedup,
                }

                entry = f"  {wl_uuid}: {status}"
                if latency:
                    entry += f" | {latency:.3f}ms"
                if speedup:
                    entry += f" | {speedup:.2f}x"
                log(entry)

        passed = sum(1 for r in run_results.values() if r["status"] == "PASSED")
        log(f"  => {passed}/{len(run_results)} PASSED")
        results[label] = run_results

    # ── Comparison ──
    log(f"\n{'='*60}")
    log(f"A/B COMPARISON: {label_a} vs {label_b}")
    log(f"{'='*60}")
    log(f"{'Workload':<12} {'A speed':>10} {'B speed':>10} {'Delta':>8} {'Verdict':>10}")
    log(f"{'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")

    a_results = results.get(label_a, {})
    b_results = results.get(label_b, {})
    all_uuids = sorted(set(list(a_results.keys()) + list(b_results.keys())))

    improvements = 0
    regressions = 0
    neutral = 0

    for uuid in all_uuids:
        a = a_results.get(uuid, {})
        b = b_results.get(uuid, {})
        a_spd = a.get("speedup")
        b_spd = b.get("speedup")

        if a_spd and b_spd:
            delta = (b_spd - a_spd) / a_spd * 100
            if delta > 2:
                verdict = "✅ BETTER"
                improvements += 1
            elif delta < -2:
                verdict = "❌ WORSE"
                regressions += 1
            else:
                verdict = "≈ SAME"
                neutral += 1
            log(f"{uuid:<12} {a_spd:>9.2f}x {b_spd:>9.2f}x {delta:>+7.1f}% {verdict:>10}")
        else:
            a_status = a.get("status", "N/A")
            b_status = b.get("status", "N/A")
            log(f"{uuid:<12} {a_status:>10} {b_status:>10}  {'N/A':>7}  {'N/A':>10}")

    a_speeds = [a_results[u]["speedup"] for u in all_uuids if a_results.get(u, {}).get("speedup")]
    b_speeds = [b_results[u]["speedup"] for u in all_uuids if b_results.get(u, {}).get("speedup")]

    if a_speeds and b_speeds:
        a_mean = sum(a_speeds) / len(a_speeds)
        b_mean = sum(b_speeds) / len(b_speeds)
        mean_delta = (b_mean - a_mean) / a_mean * 100
        log(f"\nMean speedup: A={a_mean:.2f}x  B={b_mean:.2f}x  Delta={mean_delta:+.1f}%")

    log(f"\nSummary: {improvements} improved, {regressions} regressed, {neutral} neutral (±2% threshold)")

    return "\n".join(lines)


@app.local_entrypoint()
def main():
    solution_a_path = PROJECT_ROOT / "solution_a.json"
    solution_b_path = PROJECT_ROOT / "solution_b.json"

    if not solution_a_path.exists():
        print(f"ERROR: {solution_a_path} not found!")
        print("  Pack baseline first: python scripts/pack_solution_simple.py")
        print("  Then: copy solution.json solution_a.json")
        return

    if not solution_b_path.exists():
        print(f"ERROR: {solution_b_path} not found!")
        print("  Make changes, then pack: python scripts/pack_solution_simple.py")
        print("  Then: copy solution.json solution_b.json")
        return

    solution_a_json = solution_a_path.read_text(encoding="utf-8")
    solution_b_json = solution_b_path.read_text(encoding="utf-8")
    print(f"Solution A: {len(solution_a_json)} bytes")
    print(f"Solution B: {len(solution_b_json)} bytes")

    print("\nRunning A/B test on Modal B200 (same instance)...")
    result = run_ab_test.remote(solution_a_json, solution_b_json)
    print("\n=== FULL A/B RESULTS ===")
    print(result)
