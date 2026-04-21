import importlib.util
import sys
from pathlib import Path


_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

TOP_K = 8
BLOCK_M_TINY = 16
BLOCK_M_SMALL = 32
BLOCK_M_LARGE = 64
BLOCK_M_XLARGE = 128
TINY_BATCH_TOPK_TOKENS = 1024
SMALL_BATCH_TOPK_TOKENS = 4096
LARGE_BATCH_TOPK_TOKENS = 16384

_CUTE_TARGET_BLOCK_M = {
    11948: 128,
    14107: 128,
}
_T901_CUTE_TARGET_BLOCK_M = {
    901: 64,
}
_PURE_TRITON_IMPL = None
_HYBRID_TRITON_IMPL = None
_T901_TRITON_IMPL = None
_PURE_TRITON_ERROR = None
_HYBRID_TRITON_ERROR = None
_T901_TRITON_ERROR = None


def _select_block_m(num_topk_tokens: int) -> int:
    if num_topk_tokens <= TINY_BATCH_TOPK_TOKENS:
        return BLOCK_M_TINY
    if num_topk_tokens <= SMALL_BATCH_TOPK_TOKENS:
        return BLOCK_M_SMALL
    if num_topk_tokens > LARGE_BATCH_TOPK_TOKENS:
        return BLOCK_M_XLARGE
    return BLOCK_M_LARGE


def _load_local_module(module_name: str, local_name: str):
    path = _THIS_DIR / local_name
    if not path.exists():
        raise FileNotFoundError(f"Unable to locate module source for {module_name}: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_pure_triton_module():
    global _PURE_TRITON_IMPL, _PURE_TRITON_ERROR
    if _PURE_TRITON_IMPL is not None:
        return _PURE_TRITON_IMPL
    if _PURE_TRITON_ERROR is not None:
        raise RuntimeError(f"cached pure Triton load failure: {_PURE_TRITON_ERROR!r}")
    try:
        _PURE_TRITON_IMPL = _load_local_module("pure_triton_impl", "pure_triton_impl.py")
        return _PURE_TRITON_IMPL
    except Exception as exc:
        _PURE_TRITON_ERROR = exc
        raise


def _load_hybrid_triton_module():
    global _HYBRID_TRITON_IMPL, _HYBRID_TRITON_ERROR
    if _HYBRID_TRITON_IMPL is not None:
        return _HYBRID_TRITON_IMPL
    if _HYBRID_TRITON_ERROR is not None:
        raise RuntimeError(f"cached hybrid/CuTe load failure: {_HYBRID_TRITON_ERROR!r}")
    try:
        _HYBRID_TRITON_IMPL = _load_local_module("hybrid_triton_impl", "triton_impl.py")
        return _HYBRID_TRITON_IMPL
    except Exception as exc:
        _HYBRID_TRITON_ERROR = exc
        raise


def _load_t901_triton_module():
    global _T901_TRITON_IMPL, _T901_TRITON_ERROR
    if _T901_TRITON_IMPL is not None:
        return _T901_TRITON_IMPL
    if _T901_TRITON_ERROR is not None:
        raise RuntimeError(f"cached T=901 CuTe experiment load failure: {_T901_TRITON_ERROR!r}")
    try:
        _T901_TRITON_IMPL = _load_local_module("t901_triton_impl", "triton_impl_t901.py")
        return _T901_TRITON_IMPL
    except Exception as exc:
        _T901_TRITON_ERROR = exc
        raise


def kernel(
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
    output=None,
):
    T = int(routing_logits.shape[0])
    t901_block_m = _T901_CUTE_TARGET_BLOCK_M.get(T)
    if t901_block_m is not None and _select_block_m(T * TOP_K) == t901_block_m:
        return _load_t901_triton_module().kernel(
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

    cute_block_m = _CUTE_TARGET_BLOCK_M.get(T)
    if cute_block_m is not None and _select_block_m(T * TOP_K) == cute_block_m:
        return _load_hybrid_triton_module().kernel(
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

    return _load_pure_triton_module().kernel(
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
