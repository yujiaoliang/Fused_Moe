"""Pack solution source files into solution.json."""

import json
from pathlib import Path

try:
    import tomli
except ImportError:
    import tomllib as tomli

from flashinfer_bench.data import BuildSpec, Solution, SourceFile

PROJECT_ROOT = Path(__file__).parent.parent


def pack_solution() -> Path:
    config_path = PROJECT_ROOT / "config.toml"
    with open(config_path, "rb") as f:
        config = tomli.load(f)

    lang = config["build"]["language"]
    src_dir = PROJECT_ROOT / "solution" / lang

    sources = []
    for p in sorted(src_dir.rglob("*.py" if lang == "triton" else "*.cu")):
        rel = p.relative_to(src_dir).as_posix()
        sources.append(SourceFile(path=rel, content=p.read_text()))
    # Also grab .py files for CUDA binding
    if lang == "cuda":
        for p in sorted(src_dir.rglob("*.py")):
            rel = p.relative_to(src_dir).as_posix()
            sources.append(SourceFile(path=rel, content=p.read_text()))

    solution = Solution(
        name=config["solution"]["name"],
        definition=config["solution"]["definition"],
        author=config["solution"]["author"],
        spec=BuildSpec(
            language=lang,
            target_hardware=["cuda"],
            entry_point=config["build"]["entry_point"],
        ),
        sources=sources,
    )

    out_path = PROJECT_ROOT / "solution.json"
    out_path.write_text(solution.model_dump_json(indent=2))
    print(f"Solution packed: {out_path}")
    print(f"  Name: {solution.name}")
    print(f"  Definition: {solution.definition}")
    print(f"  Author: {solution.author}")
    print(f"  Language: {lang}")
    print(f"  Sources: {[s.path for s in sources]}")
    return out_path


if __name__ == "__main__":
    pack_solution()
