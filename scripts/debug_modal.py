"""
Debug Modal: step-by-step comparison of Triton vs reference on B200.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("fused-moe-debug")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
VOLUME_MOUNT = "/data"
TRACE_SET_PATH = "/data/mlsys26-contest"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


@app.function(image=image, gpu="B200:1", timeout=600, volumes={VOLUME_MOUNT: trace_volume})
def debug_kernel(kernel_source: str) -> str:
    import torch
    import os
    import traceback
    import sys

    lines = []
    def log(msg):
        lines.append(str(msg))

    log(f"GPU: {torch.cuda.get_device_name(0)}")

    # Write kernel
    os.makedirs("/tmp/solution/triton", exist_ok=True)
    with open("/tmp/solution/triton/kernel.py", "w") as f:
        f.write(kernel_source)
    sys.path.insert(0, "/tmp/solution/triton")
    import kernel as my_kernel

    from flashinfer_bench.data import TraceSet
    from flashinfer_bench.bench.evaluators import resolve_evaluator
    from flashinfer_bench.bench.config import BenchmarkConfig

    ts = TraceSet.from_path(TRACE_SET_PATH)
    def_name = 'moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048'
    defn = ts.definitions[def_name]
    workloads = ts.workloads.get(def_name, [])

    # Pick smallest workload
    wl = workloads[0].workload
    cfg = BenchmarkConfig()
    evaluator_cls = resolve_evaluator(defn)
    baseline = evaluator_cls.build_baseline(defn, wl, cfg, "cuda:0", ts.root)

    inp = baseline.inputs[0]
    ref_out_raw = baseline.outputs[0]

    log(f"inp type: {type(inp)}")
    if isinstance(inp, list):
        log(f"inp is list of {len(inp)} items")
        for i, item in enumerate(inp):
            if isinstance(item, torch.Tensor):
                log(f"  inp[{i}]: {item.shape} {item.dtype}")
            else:
                log(f"  inp[{i}]: {type(item)} = {item}")
    elif isinstance(inp, dict):
        for k, v in inp.items():
            if isinstance(v, torch.Tensor):
                log(f"  inp[{k}]: {v.shape} {v.dtype}")
            else:
                log(f"  inp[{k}]: {type(v)} = {v}")

    log(f"ref_out_raw type: {type(ref_out_raw)}")
    if isinstance(ref_out_raw, list):
        ref_out = ref_out_raw[0]
    elif isinstance(ref_out_raw, torch.Tensor):
        ref_out = ref_out_raw
    else:
        ref_out = list(ref_out_raw.values())[0]
    log(f"ref_out: {ref_out.shape} {ref_out.dtype}")

    # Run our kernel with the same inputs
    try:
        if isinstance(inp, list):
            # List-based inputs: positional
            output = torch.empty_like(ref_out)
            my_kernel.kernel(*inp, output)
        else:
            output = torch.empty_like(ref_out)
            my_kernel.kernel(**inp, output=output)

        # Compare
        x = output.float()
        y = ref_out.float()
        abs_err = torch.abs(x - y)
        rel_err = abs_err / (torch.abs(y) + 1e-8)

        log(f"\nOUR output: mean={x.mean():.4f}, std={x.std():.4f}, min={x.min():.4f}, max={x.max():.4f}")
        log(f"REF output: mean={y.mean():.4f}, std={y.std():.4f}, min={y.min():.4f}, max={y.max():.4f}")
        log(f"ABS error: mean={abs_err.mean():.4f}, max={abs_err.max():.4f}")
        log(f"REL error: mean={rel_err.mean():.4f}, max={rel_err.max():.4f}")

        # Check per-row
        for row in range(min(ref_out.shape[0], 5)):
            row_abs = abs_err[row].max().item()
            row_our = x[row, :5].tolist()
            row_ref = y[row, :5].tolist()
            log(f"  Row {row}: max_abs={row_abs:.2f}, our[:5]={[f'{v:.1f}' for v in row_our]}, ref[:5]={[f'{v:.1f}' for v in row_ref]}")

    except Exception as e:
        log(f"ERROR: {e}")
        log(traceback.format_exc())

    return "\n".join(lines)


@app.local_entrypoint()
def main():
    kernel_path = PROJECT_ROOT / "solution" / "triton" / "kernel.py"
    source = kernel_path.read_text(encoding="utf-8")
    print("Running debug on Modal B200...")
    result = debug_kernel.remote(source)
    print(result)
