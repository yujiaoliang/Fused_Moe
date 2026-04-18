"""Pack solution source files into solution.json.

Usage:
  python scripts/pack_solution_simple.py          # Pack configured solution
  python scripts/pack_solution_simple.py --cuda   # Pack pure CUDA solution
"""

import argparse
import json
from pathlib import Path

try:
    import tomli
except ImportError:
    import tomllib as tomli

from flashinfer_bench.data import BuildSpec, Solution, SourceFile

PROJECT_ROOT = Path(__file__).parent.parent


def _gather_sources(language: str, source_dir_name: str) -> list[SourceFile]:
    source_dir = PROJECT_ROOT / "solution" / source_dir_name
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    sources: list[SourceFile] = []
    if source_dir_name == "python":
        patterns = ("*.py", "*.cu", "*.cpp")
    else:
        patterns = ("*.cu", "*.cpp", "*.py") if language == "cuda" else ("*.py",)
    for pattern in patterns:
        for p in sorted(source_dir.rglob(pattern)):
            rel = p.relative_to(source_dir).as_posix()
            sources.append(SourceFile(path=rel, content=p.read_text(encoding="utf-8")))

    return sources


def pack_solution(use_cuda: bool = False) -> Path:
    config_path = PROJECT_ROOT / "config.toml"
    with open(config_path, "rb") as f:
        config = tomli.load(f)

    if use_cuda:
        lang = "cuda"
        source_dir_name = "cuda"
        entry_point = config["build"].get("cuda_entry_point", "binding.py::kernel")
        binding = config["build"].get("cuda_binding", "torch")
        sources = _gather_sources(lang, source_dir_name)
        print("[CUDA mode] Packing solution/cuda")
    else:
        lang = config["build"]["language"]
        source_dir_name = config["build"].get("source_dir", lang)
        entry_point = config["build"]["entry_point"]
        binding = None
        sources = _gather_sources(lang, source_dir_name)

    solution = Solution(
        name=config["solution"]["name"],
        definition=config["solution"]["definition"],
        author=config["solution"]["author"],
        spec=BuildSpec(
            language=lang,
            target_hardware=["cuda"],
            entry_point=entry_point,
        ),
        sources=sources,
    )

    payload = json.loads(solution.model_dump_json(indent=2))
    if binding:
        payload["spec"]["binding"] = binding

    out_path = PROJECT_ROOT / "solution.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Solution packed: {out_path}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {lang}")
    if binding:
        print(f"  Binding: {binding}")
    if not use_cuda:
        print(f"  Source dir: {source_dir_name}")
    print(f"  Sources: {[s.path for s in sources]}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack solution for submission")
    parser.add_argument("--cuda", action="store_true",
                        help="Pack pure CUDA solution from solution/cuda")
    args = parser.parse_args()
    pack_solution(use_cuda=args.cuda)
