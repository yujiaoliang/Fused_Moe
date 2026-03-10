"""
Accurate Modal profiler for Track-A MoE Triton kernel.

This script focuses on reliable timing data:
1) End-to-end wall time (CUDA events)
2) Per-kernel CUDA time breakdown (torch.profiler)
3) FLOPs/TFLOPS based on actual num_padded from Triton sort kernel
4) Optional NCU counters if `ncu` is available in the Modal runtime
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = modal.App("moe-ncu-profile-yjl")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
VOLUME_MOUNT = "/data"
TRACE_SET_PATH = "/data/mlsys26-contest"
TRACE_DEF_NAME = "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


def _get_cuda_us(event) -> float:
    for attr in (
        "self_cuda_time_total",
        "cuda_time_total",
        "self_device_time_total",
        "device_time_total",
    ):
        value = getattr(event, attr, None)
        if value is not None:
            return float(value)
    return 0.0


def _classify_event(name: str) -> str:
    name_l = name.lower()
    if "_fused_moe_gemm1_swiglu_kernel" in name_l:
        return "gemm1"
    if "_fused_moe_gemm2_scatter_kernel" in name_l:
        return "gemm2"
    if "triton_ds_routing_kernel" in name_l:
        return "routing"
    if "triton_sort_and_scatter_kernel" in name_l:
        return "sorting"
    if any(x in name_l for x in ("copy", "zero_", "fill_", "memset", "memcpy")):
        return "memops"
    return "other"


def _find_ncu_binary() -> str | None:
    candidates = (
        "/usr/local/cuda/bin/ncu",
        "/usr/bin/ncu",
        "/opt/nvidia/nsight-compute/ncu",
    )
    for path in candidates:
        if os.path.exists(path):
            return path
    try:
        result = subprocess.run(["which", "ncu"], capture_output=True, text=True, check=False)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        return None
    return None


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={VOLUME_MOUNT: trace_volume})
def run_profile(solution_json: str) -> str:
    import importlib.util

    import torch
    from torch.profiler import ProfilerActivity, profile
    from flashinfer_bench.bench.config import BenchmarkConfig
    from flashinfer_bench.bench.evaluators import resolve_evaluator
    from flashinfer_bench.data import TraceSet

    logs: list[str] = []

    def log(msg: str) -> None:
        logs.append(str(msg))
        print(msg)

    # Runtime info
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"PyTorch: {torch.__version__}")
    log(f"Triton: {__import__('triton').__version__}")

    # Load kernel module from packed solution JSON
    solution = json.loads(solution_json)
    entry_point = solution.get("spec", {}).get("entry_point", "kernel.py::kernel")
    entry_file = entry_point.split("::")[0]
    source_code = None
    for src in solution.get("sources", []):
        if isinstance(src, dict) and src.get("path") == entry_file:
            source_code = src.get("content", "")
            break
    if source_code is None:
        raise RuntimeError(f"Cannot find source for entry file: {entry_file}")

    tmp_dir = tempfile.mkdtemp(prefix="moe_profile_")
    kernel_path = os.path.join(tmp_dir, "kernel.py")
    with open(kernel_path, "w", encoding="utf-8") as f:
        f.write(source_code)

    spec = importlib.util.spec_from_file_location("kernel_module", kernel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    kernel_fn = module.kernel
    log("Kernel loaded")

    # Optional NCU availability
    ncu_path = _find_ncu_binary()
    if ncu_path:
        ncu_ver = subprocess.run([ncu_path, "--version"], capture_output=True, text=True, check=False)
        log(f"NCU: {ncu_ver.stdout.strip() or 'available'}")
    else:
        log("NCU: not found (skip hardware counters)")

    # Shapes from definition
    H = int(getattr(module, "H", 7168))
    I_SIZE = int(getattr(module, "I_SIZE", 2048))
    E_LOCAL = int(getattr(module, "E_LOCAL", 32))
    TOP_K = int(getattr(module, "TOP_K", 8))
    QBLOCK = int(getattr(module, "QBLOCK", 128))
    BLOCK_M = int(getattr(module, "BLOCK_M", 64))

    DEFAULT_T_VALUES = [7, 14, 64, 128, 512, 1024, 4096]
    WARMUP = 15
    ITERS = 80

    summary_rows = []

    def build_inputs_synthetic(t: int):
        routing_logits = torch.randn(t, 256, dtype=torch.float32, device="cuda")
        routing_bias = torch.randn(256, dtype=torch.bfloat16, device="cuda")

        a_fp32 = torch.randn(t, H, dtype=torch.float32, device="cuda")
        a_amax = a_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        hidden_states = (a_fp32 * 448.0 / a_amax).to(torch.float8_e4m3fn)
        hidden_states_scale = (a_amax / 448.0).expand(t, H // QBLOCK).t().contiguous().float()

        gemm1_weights = (
            torch.randint(-10, 10, (E_LOCAL, 4096, H), dtype=torch.int8, device="cuda")
            .to(torch.float8_e4m3fn)
        )
        gemm1_weights_scale = (
            torch.rand(E_LOCAL, 4096 // QBLOCK, H // QBLOCK, dtype=torch.float32, device="cuda") + 0.5
        )

        gemm2_weights = (
            torch.randint(-10, 10, (E_LOCAL, H, I_SIZE), dtype=torch.int8, device="cuda")
            .to(torch.float8_e4m3fn)
        )
        gemm2_weights_scale = (
            torch.rand(E_LOCAL, H // QBLOCK, I_SIZE // QBLOCK, dtype=torch.float32, device="cuda") + 0.5
        )

        local_expert_offset = torch.tensor(0, dtype=torch.int32, device="cuda")
        routed_scaling_factor = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        output = torch.zeros(t, H, dtype=torch.bfloat16, device="cuda")

        return [
            routing_logits,
            routing_bias,
            hidden_states,
            hidden_states_scale,
            gemm1_weights,
            gemm1_weights_scale,
            gemm2_weights,
            gemm2_weights_scale,
            local_expert_offset,
            routed_scaling_factor,
            output,
        ]

    def _extract_tensor(x):
        if torch.is_tensor(x):
            return x
        if isinstance(x, list) and x and torch.is_tensor(x[0]):
            return x[0]
        if isinstance(x, tuple) and x and torch.is_tensor(x[0]):
            return x[0]
        if isinstance(x, dict):
            for v in x.values():
                if torch.is_tensor(v):
                    return v
        return None

    def _normalize_kernel_args(inp, ref_out):
        # Kernel signature order (without output)
        ordered_keys = [
            "routing_logits",
            "routing_bias",
            "hidden_states",
            "hidden_states_scale",
            "gemm1_weights",
            "gemm1_weights_scale",
            "gemm2_weights",
            "gemm2_weights_scale",
            "local_expert_offset",
            "routed_scaling_factor",
        ]
        if isinstance(inp, (list, tuple)):
            args = list(inp)
        elif isinstance(inp, dict):
            args = [inp[k] for k in ordered_keys]
        else:
            raise TypeError(f"Unsupported baseline input type: {type(inp)}")

        if len(args) < 10:
            raise RuntimeError(f"Expected at least 10 input args, got {len(args)}")

        out_tensor = _extract_tensor(ref_out)
        if out_tensor is None:
            raise RuntimeError("Cannot infer output tensor shape/dtype from baseline output")
        output = torch.zeros_like(out_tensor)

        if len(args) == 10:
            args.append(output)
        else:
            args[10] = output
        return args

    def build_trace_inputs_by_t():
        trace_inputs = {}
        try:
            ts = TraceSet.from_path(TRACE_SET_PATH)
            if TRACE_DEF_NAME not in ts.definitions:
                log(f"Trace definition not found: {TRACE_DEF_NAME}")
                return trace_inputs
            defn = ts.definitions[TRACE_DEF_NAME]
            workloads = ts.workloads.get(TRACE_DEF_NAME, [])
            evaluator_cls = resolve_evaluator(defn)
            cfg = BenchmarkConfig()

            for item in workloads:
                wl = item.workload
                baseline = evaluator_cls.build_baseline(defn, wl, cfg, "cuda:0", ts.root)
                inp = baseline.inputs[0]
                ref_out = baseline.outputs[0]
                args = _normalize_kernel_args(inp, ref_out)
                t_val = int(args[0].shape[0])
                if t_val not in trace_inputs:
                    trace_inputs[t_val] = args
            log(f"Trace samples loaded for T values: {sorted(trace_inputs.keys())}")
        except Exception as exc:
            log(f"Trace loading failed, fallback to synthetic inputs: {exc}")
        return trace_inputs

    def infer_num_padded(args) -> tuple[int | None, int | None]:
        # Uses the same Triton kernels as runtime path, so this is exact for current input.
        if not hasattr(module, "ds_routing") or not hasattr(module, "triton_sort_and_scatter_kernel"):
            return None, None
        try:
            def _to_int(v):
                if torch.is_tensor(v):
                    return int(v.item())
                return int(v)

            def _to_float(v):
                if torch.is_tensor(v):
                    return float(v.item())
                return float(v)

            routing_logits = args[0]
            routing_bias = args[1]
            local_expert_offset = _to_int(args[8])
            routed_scaling_factor = _to_float(args[9])
            t = int(routing_logits.shape[0])

            topk_idx, topk_wts = module.ds_routing(routing_logits, routing_bias, routed_scaling_factor)

            max_padded = t * TOP_K + E_LOCAL * BLOCK_M
            sorted_token_ids = torch.empty((max_padded,), dtype=torch.int64, device="cuda")
            sorted_weights = torch.empty((max_padded,), dtype=torch.float32, device="cuda")
            block_offsets = torch.empty((E_LOCAL + 1,), dtype=torch.int32, device="cuda")
            total_blocks = torch.empty((1,), dtype=torch.int32, device="cuda")
            counts_workspace = torch.empty((E_LOCAL,), dtype=torch.int32, device="cuda")

            module.triton_sort_and_scatter_kernel[(1,)](
                topk_idx,
                topk_wts,
                sorted_token_ids,
                sorted_weights,
                block_offsets,
                total_blocks,
                counts_workspace,
                local_expert_offset,
                t,
                TOP_K,
                E_LOCAL,
                BLOCK_M,
                max_padded,
                num_warps=8,
            )
            tb = int(total_blocks.item())
            return tb, tb * BLOCK_M
        except Exception:
            return None, None

    trace_inputs_by_t = build_trace_inputs_by_t()
    if trace_inputs_by_t:
        run_t_values = sorted(trace_inputs_by_t.keys())
        log(f"Using trace T values: {run_t_values}")
    else:
        run_t_values = DEFAULT_T_VALUES
        log(f"Using fallback T values: {run_t_values}")

    for t in run_t_values:
        log("")
        log("=" * 96)
        log(f"T = {t}")
        log("=" * 96)

        if t in trace_inputs_by_t:
            args = trace_inputs_by_t[t]
            log(f"Input source: trace set (T={t})")
        else:
            args = build_inputs_synthetic(t)
            log(f"Input source: synthetic fallback (T={t})")
        output = args[-1]

        total_blocks, num_padded = infer_num_padded(args)
        if total_blocks is not None:
            log(f"Inferred padding: total_blocks={total_blocks}, num_padded={num_padded}")
        else:
            log("Inferred padding: unavailable (fallback to N/A FLOP metrics)")

        # Warmup (JIT/autotune)
        for _ in range(WARMUP):
            output.zero_()
            kernel_fn(*args)
        torch.cuda.synchronize()

        # Wall time (end-to-end)
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_ev.record()
        for _ in range(ITERS):
            output.zero_()
            kernel_fn(*args)
        end_ev.record()
        torch.cuda.synchronize()
        wall_ms = start_ev.elapsed_time(end_ev) / ITERS
        toks_per_s = t / (wall_ms * 1e-3)
        log(f"Wall: {wall_ms:.3f} ms/iter | Throughput: {toks_per_s:,.0f} tok/s")

        # Kernel breakdown
        prof_kwargs = dict(activities=[ProfilerActivity.CUDA], record_shapes=False)
        try:
            with profile(**prof_kwargs, acc_events=True) as prof:
                for _ in range(ITERS):
                    output.zero_()
                    kernel_fn(*args)
        except TypeError:
            with profile(**prof_kwargs) as prof:
                for _ in range(ITERS):
                    output.zero_()
                    kernel_fn(*args)
        torch.cuda.synchronize()

        totals_us = {
            "gemm1": 0.0,
            "gemm2": 0.0,
            "routing": 0.0,
            "sorting": 0.0,
            "memops": 0.0,
            "other": 0.0,
        }
        top_kernels = []
        for event in prof.key_averages():
            cuda_us = _get_cuda_us(event)
            if cuda_us <= 0:
                continue
            category = _classify_event(event.key)
            totals_us[category] += cuda_us
            top_kernels.append((event.key, category, cuda_us / ITERS, event.count / ITERS))

        total_cuda_ms = sum(totals_us.values()) / ITERS / 1000.0
        cpu_overhead_ms = max(0.0, wall_ms - total_cuda_ms)

        def pct(us_val: float) -> float:
            denom = sum(totals_us.values())
            return 100.0 * us_val / denom if denom > 0 else 0.0

        log("CUDA breakdown (per iter):")
        log(f"  GEMM1   : {totals_us['gemm1']/ITERS/1000.0:8.3f} ms ({pct(totals_us['gemm1']):5.1f}%)")
        log(f"  GEMM2   : {totals_us['gemm2']/ITERS/1000.0:8.3f} ms ({pct(totals_us['gemm2']):5.1f}%)")
        log(f"  Routing : {totals_us['routing']/ITERS/1000.0:8.3f} ms ({pct(totals_us['routing']):5.1f}%)")
        log(f"  Sorting : {totals_us['sorting']/ITERS/1000.0:8.3f} ms ({pct(totals_us['sorting']):5.1f}%)")
        log(f"  MemOps  : {totals_us['memops']/ITERS/1000.0:8.3f} ms ({pct(totals_us['memops']):5.1f}%)")
        log(f"  Other   : {totals_us['other']/ITERS/1000.0:8.3f} ms ({pct(totals_us['other']):5.1f}%)")
        log(f"  Total CUDA: {total_cuda_ms:.3f} ms/iter")
        log(f"  CPU overhead: {cpu_overhead_ms:.3f} ms/iter ({(cpu_overhead_ms / wall_ms * 100.0):.1f}%)")

        top_kernels.sort(key=lambda x: -x[2])
        log("Top kernels:")
        for name, category, us, count in top_kernels[:8]:
            log(f"  [{category:7s}] {us:8.2f} us x {count:.1f} | {name[:90]}")

        # FLOP-based metrics using actual num_padded
        gemm1_tflops = None
        gemm2_tflops = None
        if num_padded is not None and num_padded > 0:
            gemm1_flops = 2.0 * num_padded * 4096 * H
            gemm2_flops = 2.0 * num_padded * H * I_SIZE
            gemm1_ms = totals_us["gemm1"] / ITERS / 1000.0
            gemm2_ms = totals_us["gemm2"] / ITERS / 1000.0
            if gemm1_ms > 0:
                gemm1_tflops = gemm1_flops / (gemm1_ms * 1e-3) / 1e12
            if gemm2_ms > 0:
                gemm2_tflops = gemm2_flops / (gemm2_ms * 1e-3) / 1e12
            log(
                f"Effective TFLOPS (actual num_padded={num_padded}): "
                f"GEMM1={gemm1_tflops:.1f}T, GEMM2={gemm2_tflops:.1f}T"
            )
        else:
            log("Effective TFLOPS: N/A (num_padded unavailable)")

        summary_rows.append(
            {
                "T": t,
                "wall_ms": wall_ms,
                "cuda_ms": total_cuda_ms,
                "cpu_ms": cpu_overhead_ms,
                "routing_ms": totals_us["routing"] / ITERS / 1000.0,
                "sorting_ms": totals_us["sorting"] / ITERS / 1000.0,
                "gemm1_ms": totals_us["gemm1"] / ITERS / 1000.0,
                "gemm2_ms": totals_us["gemm2"] / ITERS / 1000.0,
                "num_padded": num_padded if num_padded is not None else -1,
                "gemm1_tflops": gemm1_tflops if gemm1_tflops is not None else -1.0,
                "gemm2_tflops": gemm2_tflops if gemm2_tflops is not None else -1.0,
            }
        )

    # Summary table
    log("")
    log("=" * 96)
    log("SUMMARY")
    log("=" * 96)
    log(
        f"{'T':>6s} | {'Wall':>7s} | {'CUDA':>7s} | {'CPU':>7s} | {'Route':>7s} | "
        f"{'Sort':>7s} | {'GEMM1':>7s} | {'GEMM2':>7s} | {'num_pad':>8s} | {'G1 TF':>8s} | {'G2 TF':>8s}"
    )
    for row in summary_rows:
        g1 = f"{row['gemm1_tflops']:.1f}" if row["gemm1_tflops"] >= 0 else "N/A"
        g2 = f"{row['gemm2_tflops']:.1f}" if row["gemm2_tflops"] >= 0 else "N/A"
        npad = str(row["num_padded"]) if row["num_padded"] >= 0 else "N/A"
        log(
            f"{row['T']:6d} | {row['wall_ms']:7.3f} | {row['cuda_ms']:7.3f} | {row['cpu_ms']:7.3f} | "
            f"{row['routing_ms']:7.3f} | {row['sorting_ms']:7.3f} | {row['gemm1_ms']:7.3f} | "
            f"{row['gemm2_ms']:7.3f} | {npad:>8s} | {g1:>8s} | {g2:>8s}"
        )

    # Optional hardware counters
    if ncu_path:
        log("")
        log("=" * 96)
        log("NCU COUNTERS (if available in runtime)")
        log("=" * 96)
        ncu_script_path = os.path.join(tmp_dir, "ncu_once.py")
        with open(ncu_script_path, "w", encoding="utf-8") as f:
            f.write(
                "import torch, importlib.util\n"
                f"spec=importlib.util.spec_from_file_location('km','{kernel_path}')\n"
                "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
                "k=m.kernel\n"
                "T=1024; H=7168\n"
                "rl=torch.randn(T,256,device='cuda',dtype=torch.float32)\n"
                "rb=torch.randn(256,device='cuda',dtype=torch.bfloat16)\n"
                "a=torch.randn(T,H,device='cuda',dtype=torch.float32)\n"
                "amax=a.abs().amax(dim=-1,keepdim=True).clamp(min=1e-12)\n"
                "hs=(a*448.0/amax).to(torch.float8_e4m3fn)\n"
                "hss=(amax/448.0).expand(T,H//128).t().contiguous().float()\n"
                "g1w=torch.randint(-10,10,(32,4096,H),device='cuda',dtype=torch.int8).to(torch.float8_e4m3fn)\n"
                "g1s=torch.rand(32,32,56,device='cuda',dtype=torch.float32)+0.5\n"
                "g2w=torch.randint(-10,10,(32,H,2048),device='cuda',dtype=torch.int8).to(torch.float8_e4m3fn)\n"
                "g2s=torch.rand(32,56,16,device='cuda',dtype=torch.float32)+0.5\n"
                "leo=torch.tensor(0,device='cuda',dtype=torch.int32)\n"
                "rsf=torch.tensor(1.0,device='cuda',dtype=torch.float32)\n"
                "out=torch.zeros(T,H,device='cuda',dtype=torch.bfloat16)\n"
                "args=[rl,rb,hs,hss,g1w,g1s,g2w,g2s,leo,rsf,out]\n"
                "for _ in range(10): out.zero_(); k(*args)\n"
                "torch.cuda.synchronize(); out.zero_(); k(*args); torch.cuda.synchronize()\n"
            )
        ncu_csv = os.path.join(tmp_dir, "ncu_metrics.csv")
        cmd = [
            ncu_path,
            "--kernel-name",
            "regex:.*(fused_moe|triton_ds_routing_kernel|triton_sort_and_scatter_kernel).*",
            "--metrics",
            ",".join(
                [
                    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                    "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
                    "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
                ]
            ),
            "--csv",
            "--log-file",
            ncu_csv,
            "--target-processes",
            "all",
            sys.executable,
            ncu_script_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
            log(f"NCU exit code: {result.returncode}")
            if result.returncode != 0 and result.stderr:
                for line in result.stderr.splitlines()[:12]:
                    log(f"NCU stderr: {line}")
            if os.path.exists(ncu_csv):
                log("NCU CSV (first 30 lines):")
                with open(ncu_csv, "r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f):
                        if i >= 30:
                            break
                        log(f"  {line.rstrip()}")
        except Exception as exc:
            log(f"NCU profiling failed: {exc}")

    return "\n".join(logs)


@app.local_entrypoint()
def main():
    print("Packing solution...")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "pack_solution_simple.py")],
        cwd=str(PROJECT_ROOT),
        check=True,
    )

    solution_path = PROJECT_ROOT / "solution.json"
    solution_json = solution_path.read_text(encoding="utf-8")
    print(f"Loaded packed solution: {solution_path}")

    print("Running profile on Modal B200...")
    report = run_profile.remote(solution_json)

    out_path = PROJECT_ROOT / "ncu_profiler_yjl.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"Saved report: {out_path}")
    print(report)
