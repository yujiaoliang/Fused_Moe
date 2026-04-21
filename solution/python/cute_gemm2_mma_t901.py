import importlib.util
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
_IMPL = None


def _load_impl():
    global _IMPL
    if _IMPL is not None:
        return _IMPL

    path = _THIS_DIR / "cute_gemm2_mma.py"
    spec = importlib.util.spec_from_file_location("cute_gemm2_mma_t901_impl", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    # Isolate T=901 from the large-T CuTe GEMM2 state/cache.
    module.TARGET_BLOCK_M = {901: 64}
    module._COMPILED = {}
    module._COMPILE_ERROR = {}
    module._STATE = {}
    module._B_FP16_CACHE = {}
    module._METADATA_CACHE = {}
    _IMPL = module
    return module


def run(
    intermediate,
    gemm2_weights,
    gemm2_weights_scale,
    expert_out,
    block_offsets,
    total_tokens,
    block_m,
    block_offsets_host=None,
):
    if int(total_tokens) != 901 or int(block_m) != 64:
        raise ValueError("T=901 CuTe GEMM2 path only supports total_tokens=901 and block_m=64")
    return _load_impl().run(
        intermediate,
        gemm2_weights,
        gemm2_weights_scale,
        expert_out,
        block_offsets,
        int(total_tokens),
        int(block_m),
        block_offsets_host=block_offsets_host,
    )
