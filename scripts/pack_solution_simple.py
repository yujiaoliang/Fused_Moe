"""Pack solution source files into solution.json.

Usage:
  python scripts/pack_solution_simple.py          # Pack Triton solution (default)
  python scripts/pack_solution_simple.py --cuda    # Pack CUDA baseline (binding.py)
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


def pack_solution(use_cuda: bool = False) -> Path:
    config_path = PROJECT_ROOT / "config.toml"
    with open(config_path, "rb") as f:
        config = tomli.load(f)

    if use_cuda:
        # Pack CUDA baseline (binding.py) as a triton solution
        # The benchmark framework only runs .py via triton builder,
        # so we pretend it's triton and rename binding.py → kernel.py
        lang = "triton"
        entry_point = "kernel.py::kernel"
        cuda_binding = PROJECT_ROOT / "solution" / "cuda" / "binding.py"
        sources = [SourceFile(path="kernel.py", content=cuda_binding.read_text(encoding="utf-8"))]
        print("[CUDA mode] Packing solution/cuda/binding.py as kernel.py")
    else:
        lang = config["build"]["language"]
        entry_point = config["build"]["entry_point"]
        src_dir = PROJECT_ROOT / "solution" / lang

        sources = []
        for p in sorted(src_dir.rglob("*.py" if lang == "triton" else "*.cu")):
            rel = p.relative_to(src_dir).as_posix()
            sources.append(SourceFile(path=rel, content=p.read_text(encoding="utf-8")))
        if lang == "cuda":
            for p in sorted(src_dir.rglob("*.py")):
                rel = p.relative_to(src_dir).as_posix()
                sources.append(SourceFile(path=rel, content=p.read_text(encoding="utf-8")))

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

    out_path = PROJECT_ROOT / "solution.json"
    out_path.write_text(solution.model_dump_json(indent=2), encoding="utf-8")
    print(f"Solution packed: {out_path}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {lang}")
    print(f"  Sources: {[s.path for s in sources]}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pack solution for submission")
    parser.add_argument("--cuda", action="store_true",
                        help="Pack CUDA baseline (solution/cuda/binding.py) instead of Triton kernel")
    args = parser.parse_args()
    pack_solution(use_cuda=args.cuda)
