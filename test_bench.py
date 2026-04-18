"""
Run the official-style flashinfer-bench CLI on Modal.

Run with:
  modal run test_bench.py
"""

import json
import subprocess
import sys
from pathlib import Path

import modal

PROJECT_ROOT = Path(__file__).parent

app = modal.App("flashinfer-moe-eval")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
VOLUME_MOUNT = "/data"
TRACE_SET_PATH = "/data/mlsys26-contest"

image = (
    modal.Image.from_registry("flashinfer/flashinfer-ci-cu132:20260401-2c675fb")
    .entrypoint([])
    .env({"CUDA_HOME": "/usr/local/cuda"})
)


@app.function(image=image, gpu="B200:1", timeout=7200, volumes={VOLUME_MOUNT: trace_volume})
def run_cli_benchmark(solution_json: str) -> dict:
    import glob
    import os
    import re
    import shlex
    import shutil
    import tempfile

    def _find_flashinfer_bench() -> str:
        found = shutil.which("flashinfer-bench")
        if found:
            return found

        search_patterns = [
            "/opt/conda/envs/*/bin/flashinfer-bench",
            "/opt/conda/bin/flashinfer-bench",
            "/opt/venv/bin/flashinfer-bench",
            "/usr/local/bin/flashinfer-bench",
            "/usr/bin/flashinfer-bench",
            "/root/.local/bin/flashinfer-bench",
        ]
        for pattern in search_patterns:
            matches = sorted(glob.glob(pattern))
            if matches:
                return matches[0]
        return "flashinfer-bench"

    def _pip_install(run_dir: Path, packages: list[str], no_deps: bool = False) -> None:
        install_cmd = ["python", "-m", "pip", "install"]
        if no_deps:
            install_cmd.append("--no-deps")
        install_cmd.extend(packages)
        print("$ " + " ".join(install_cmd))
        result = subprocess.run(
            install_cmd,
            cwd=str(run_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())
        if result.returncode != 0:
            raise RuntimeError(
                f"pip install failed for {packages}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )

    def _ensure_flashinfer_bench_imports(run_dir: Path) -> None:
        module_to_package = {
            "click": "click",
            "docstring_parser": "docstring-parser",
            "filelock": "filelock",
            "jinja2": "jinja2",
            "markdown_it": "markdown-it-py",
            "mdurl": "mdurl",
            "numpy": "numpy",
            "packaging": "packaging",
            "pydantic": "pydantic",
            "pydantic_core": "pydantic-core",
            "pygments": "pygments",
            "rich": "rich",
            "safetensors": "safetensors",
            "shellingham": "shellingham",
            "tabulate": "tabulate",
            "tqdm": "tqdm",
            "typer": "typer",
            "typing_extensions": "typing-extensions",
            "yaml": "pyyaml",
        }

        probe_cmd = [
            "python",
            "-c",
            "from flashinfer_bench.cli.main import cli; print('flashinfer-bench import ok')",
        ]
        for _ in range(12):
            result = subprocess.run(
                probe_cmd,
                cwd=str(run_dir),
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout.strip():
                print(result.stdout.strip())
            if result.returncode == 0:
                return

            combined = "\n".join(x for x in (result.stdout, result.stderr) if x)
            print(combined.strip())
            match = re.search(r"No module named ['\"]([^'\"]+)['\"]", combined)
            if not match:
                raise RuntimeError(
                    "flashinfer-bench import failed for a non-missing-module reason: "
                    f"{combined.strip()}"
                )

            module_name = match.group(1).split(".", 1)[0]
            package_name = module_to_package.get(module_name)
            if package_name is None:
                raise RuntimeError(
                    f"flashinfer-bench is missing module {module_name!r}, "
                    "and test_bench.py does not know the package name to install."
                )
            _pip_install(run_dir, [package_name])

        raise RuntimeError("flashinfer-bench import dependency resolution did not converge.")

    def _install_flashinfer_bench_cli_if_missing(run_dir: Path) -> str:
        before = _find_flashinfer_bench()
        if before != "flashinfer-bench":
            _ensure_flashinfer_bench_imports(run_dir)
            return before

        _pip_install(run_dir, ["flashinfer-bench"], no_deps=True)
        _ensure_flashinfer_bench_imports(run_dir)

        after = _find_flashinfer_bench()
        if after == "flashinfer-bench":
            raise RuntimeError(
                "flashinfer-bench CLI was not present in the image and could not be "
                "installed with `python -m pip install --no-deps flashinfer-bench`."
            )
        return after

    run_dir = Path(tempfile.mkdtemp(prefix="flashinfer_eval_"))
    dataset_root = run_dir / "flashinfer-trace"
    dataset_root.mkdir(parents=True, exist_ok=True)

    solution_obj = json.loads(solution_json)
    solution_name = solution_obj.get("name", "packed_solution")

    for dirname in ("definitions", "workloads", "blob"):
        src = Path(TRACE_SET_PATH) / dirname
        dst = dataset_root / dirname
        if src.exists() and not dst.exists():
            os.symlink(src, dst, target_is_directory=True)

    traces_dir = dataset_root / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    solutions_dir = dataset_root / "solutions"
    solutions_dir.mkdir(parents=True, exist_ok=True)
    solution_path = solutions_dir / f"{solution_name}.json"
    solution_path.write_text(solution_json, encoding="utf-8")

    setup_cmds = [
        ["bash", "-lc", "pwd"],
        ["bash", "-lc", "echo PATH=$PATH"],
        ["bash", "-lc", "echo CUDA_HOME=$CUDA_HOME"],
        ["bash", "-lc", "which flashinfer-bench || true"],
        ["bash", "-lc", "find /opt /usr/local /usr /root -path '*/bin/flashinfer-bench' -type f 2>/dev/null | head -20 || true"],
        ["bash", "-lc", "find /opt /usr/local /usr /root -name '*flashinfer*bench*' 2>/dev/null | head -50 || true"],
        ["bash", "-lc", "python -m pip show flashinfer-bench || python -m pip show flashinfer_bench || true"],
        ["bash", "-lc", "python -c 'import cutlass, cuda.bindings.driver as cuda; print(\"cutlass/cuda.bindings import ok\")' || true"],
        ["bash", "-lc", "which nvcc || true"],
        ["bash", "-lc", "nvidia-smi"],
        ["bash", "-lc", "ls -la"],
        ["bash", "-lc", f"find {dataset_root} -maxdepth 2 -type d | sort"],
        ["bash", "-lc", f"ls -la {solutions_dir}"],
    ]

    setup_logs = []
    for cmd in setup_cmds:
        result = subprocess.run(
            cmd,
            cwd=str(run_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        setup_logs.append(
            {
                "cmd": " ".join(cmd),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        print(f"$ {' '.join(cmd)}")
        if result.stdout.strip():
            print(result.stdout.strip())
        if result.stderr.strip():
            print(result.stderr.strip())

    flashinfer_bench = _install_flashinfer_bench_cli_if_missing(run_dir)
    bench_dir = str(Path(flashinfer_bench).parent)
    os.environ["PATH"] = bench_dir + os.pathsep + os.environ.get("PATH", "")
    print(f"Resolved flashinfer-bench: {flashinfer_bench}")

    bench_cmd = (
        f"{shlex.quote(flashinfer_bench)} run "
        f"--local {dataset_root} "
        "--definitions moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 "
        f"--solutions {solution_name} "
        "--save-results --use-isolated-runner --log-level INFO --resume --timeout 300 "
        "--atol 1 --rtol 0.3 --required-matched-ratio 0.9"
    )

    print("$ " + bench_cmd)
    bench_result = subprocess.run(
        ["bash", "-lc", bench_cmd],
        cwd=str(run_dir),
        capture_output=True,
        text=True,
        check=False,
    )

    if bench_result.stdout.strip():
        print(bench_result.stdout.strip())
    if bench_result.stderr.strip():
        print(bench_result.stderr.strip())

    result_files = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            result_files.append(str(path.relative_to(run_dir)).replace("\\", "/"))

    return {
        "run_dir": str(run_dir),
        "dataset_root": str(dataset_root),
        "solution_path": str(solution_path),
        "dataset_path": TRACE_SET_PATH,
        "solution_name": solution_name,
        "flashinfer_bench": flashinfer_bench,
        "setup_logs": setup_logs,
        "bench": {
            "cmd": bench_cmd,
            "returncode": bench_result.returncode,
            "stdout": bench_result.stdout,
            "stderr": bench_result.stderr,
        },
        "result_files": result_files,
    }


def _print_report(report: dict) -> None:
    print("")
    print(f"Run dir: {report.get('run_dir', 'unknown')}")
    print(f"Dataset root: {report.get('dataset_root', 'unknown')}")
    print(f"Solution path: {report.get('solution_path', 'unknown')}")
    print(f"Dataset path: {report.get('dataset_path', 'unknown')}")
    print(f"Solution name: {report.get('solution_name', 'unknown')}")
    print("")
    print("CLI command:")
    print("  " + report.get("bench", {}).get("cmd", ""))
    print("")
    print(f"Return code: {report.get('bench', {}).get('returncode', -1)}")
    print("")

    stdout = report.get("bench", {}).get("stdout", "")
    stderr = report.get("bench", {}).get("stderr", "")
    if stdout.strip():
        print("STDOUT:")
        print(stdout)
    if stderr.strip():
        print("STDERR:")
        print(stderr)

    files = report.get("result_files", [])
    if files:
        print("Generated files:")
        for item in files:
            print(f"  {item}")


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

    print("Running flashinfer-bench CLI on Modal B200...")
    report = run_cli_benchmark.remote(solution_json)

    out_path = PROJECT_ROOT / "test_bench_cli_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved report: {out_path}")
    _print_report(report)


if __name__ == "__main__":
    print("Use `modal run test_bench.py` to execute this benchmark.")
