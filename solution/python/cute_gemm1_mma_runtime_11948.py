import importlib.util
import sys
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

_CUTE_GEMM1_MOD = None
_CUTE_GEMM1_ERROR = None
_CUTE_GEMM1_TARGET_T = 11948
_CUTE_GEMM1_BLOCK_M = 128


def _load_local_module(module_name: str, local_name: str):
    path = _THIS_DIR / local_name
    if not path.exists():
        raise FileNotFoundError(f"Unable to locate module source for {module_name}: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_gemm1_module():
    global _CUTE_GEMM1_MOD, _CUTE_GEMM1_ERROR

    if _CUTE_GEMM1_MOD is not None:
        return _CUTE_GEMM1_MOD
    if _CUTE_GEMM1_ERROR is not None:
        raise RuntimeError(f"cached CuTe GEMM1 MMA T=11948 failure: {_CUTE_GEMM1_ERROR!r}")

    try:
        _CUTE_GEMM1_MOD = _load_local_module("hybrid_cute_gemm1_mma_t11948", "cute_gemm1_mma.py")
        return _CUTE_GEMM1_MOD
    except Exception as exc:
        _CUTE_GEMM1_ERROR = exc
        raise


def should_use_cute_gemm1_mma(t: int, block_m: int) -> bool:
    if int(t) != _CUTE_GEMM1_TARGET_T or int(block_m) != _CUTE_GEMM1_BLOCK_M:
        return False
    if _CUTE_GEMM1_ERROR is not None:
        return False
    return True


def run_cute_gemm1_mma(
    a_sorted_fp16,
    gemm1_weights,
    gemm1_weights_scale,
    gemm1_raw,
    block_offsets,
    total_tokens,
    block_m,
    block_offsets_host=None,
):
    if not should_use_cute_gemm1_mma(int(total_tokens), int(block_m)):
        raise RuntimeError("CuTe GEMM1 MMA T=11948 runtime is not selected for this workload")

    mod = _load_gemm1_module()
    mod.run(
        a_sorted_fp16,
        gemm1_weights,
        gemm1_weights_scale,
        gemm1_raw,
        block_offsets,
        int(total_tokens),
        int(block_m),
        block_offsets_host=block_offsets_host,
    )
    return gemm1_raw
