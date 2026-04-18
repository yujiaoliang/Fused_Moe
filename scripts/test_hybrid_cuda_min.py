"""
Minimal Modal-based check for the hybrid Python + CUDA path.

Run with:
  modal run scripts/test_hybrid_cuda_min.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent.parent

app = modal.App("hybrid-cuda-min-check")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("build-essential", "ninja-build")
    .pip_install("torch", "triton", "numpy")
    .env({"CUDA_HOME": "/usr/local/cuda"})
)


def _materialize_solution_sources(solution: dict, root_dir: str) -> str:
    spec = solution.get("spec", {})
    entry_point = spec.get("entry_point", "kernel.py::kernel")
    entry_file = entry_point.split("::")[0]

    found_entry = False
    for src in solution.get("sources", []):
        if not isinstance(src, dict):
            continue
        rel_path = src.get("path")
        if not rel_path:
            continue
        abs_path = Path(root_dir) / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(src.get("content", ""), encoding="utf-8")
        if rel_path == entry_file:
            found_entry = True

    if not found_entry:
        raise RuntimeError(f"Cannot find source for entry file: {entry_file}")

    return str(Path(root_dir) / entry_file)


@app.function(image=image, gpu="B200:1", timeout=1800)
def check_cuda_toolchain(solution_json: str) -> str:
    import importlib.util

    import torch

    logs: list[str] = []

    def log(msg: str) -> None:
        logs.append(str(msg))
        print(msg)

    for cmd in (
        ["bash", "-lc", "echo CUDA_HOME=$CUDA_HOME"],
        ["bash", "-lc", "which nvcc || true"],
        ["bash", "-lc", "nvcc --version || true"],
        ["bash", "-lc", "ls -l /usr/local/cuda/include/cuda_runtime_api.h || true"],
    ):
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        log("$ " + " ".join(cmd))
        if result.stdout.strip():
            log(result.stdout.strip())
        if result.stderr.strip():
            log(result.stderr.strip())

    solution = json.loads(solution_json)
    tmp_dir = tempfile.mkdtemp(prefix="hybrid_cuda_min_")
    kernel_path = _materialize_solution_sources(solution, tmp_dir)

    spec = importlib.util.spec_from_file_location("hybrid_kernel", kernel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    log(f"Initial runtime status: {json.dumps(module.get_runtime_status(), ensure_ascii=False)}")

    t = 1
    h = int(module.H)
    qblock = int(module.QBLOCK)
    e_local = int(module.E_LOCAL)
    i_size = int(module.I_SIZE)

    routing_logits = torch.randn(t, 256, dtype=torch.float32, device="cuda")
    routing_bias = torch.randn(256, dtype=torch.bfloat16, device="cuda")

    a_fp32 = torch.randn(t, h, dtype=torch.float32, device="cuda")
    a_amax = a_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    hidden_states = (a_fp32 * 448.0 / a_amax).to(torch.float8_e4m3fn)
    hidden_states_scale = (a_amax / 448.0).expand(t, h // qblock).t().contiguous().float()

    gemm1_weights = (
        torch.randint(-10, 10, (e_local, 4096, h), dtype=torch.int8, device="cuda")
        .to(torch.float8_e4m3fn)
    )
    gemm1_weights_scale = torch.rand(e_local, 4096 // qblock, h // qblock, dtype=torch.float32, device="cuda") + 0.5

    gemm2_weights = (
        torch.randint(-10, 10, (e_local, h, i_size), dtype=torch.int8, device="cuda")
        .to(torch.float8_e4m3fn)
    )
    gemm2_weights_scale = torch.rand(e_local, h // qblock, i_size // qblock, dtype=torch.float32, device="cuda") + 0.5

    local_expert_offset = torch.tensor(0, dtype=torch.int32, device="cuda")
    routed_scaling_factor = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    output = torch.zeros(t, h, dtype=torch.bfloat16, device="cuda")

    module.kernel(
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
    )
    torch.cuda.synchronize()

    log(f"Output sum: {float(output.float().sum().item()):.6f}")
    log(f"Runtime status after one call: {json.dumps(module.get_runtime_status(), ensure_ascii=False)}")
    return "\n".join(logs)


@app.local_entrypoint()
def main():
    print("Packing hybrid Python solution from solution/python...")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "pack_solution_simple.py")],
        cwd=str(PROJECT_ROOT),
        check=True,
    )

    solution_path = PROJECT_ROOT / "solution.json"
    solution_json = solution_path.read_text(encoding="utf-8")
    print(f"Loaded packed solution: {solution_path}")

    check_cuda_toolchain.remote(solution_json)
    print("Minimal hybrid CUDA check completed.")


if __name__ == "__main__":
    print("Use `modal run scripts/test_hybrid_cuda_min.py` to execute this check.")
