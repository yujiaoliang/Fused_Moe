"""
Accurate Modal profiler for Track-A MoE hybrid solutions.

This script focuses on reliable timing data:
1) End-to-end wall time (CUDA events)
2) Per-kernel CUDA time breakdown (torch.profiler)
3) FLOPs/TFLOPS based on actual num_padded when sort helpers are exposed
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
REMOTE_REPORT_PATH = f"{VOLUME_MOUNT}/ncu_profiler_yjl_latest.txt"
ENABLE_NCU_COUNTERS = os.environ.get("YJL_ENABLE_NCU_COUNTERS", "0") == "1"


def _parse_target_t_values():
    raw = os.environ.get("YJL_TARGET_T_VALUES", "").strip()
    if not raw or raw.lower() in {"all", "none", "*"}:
        return None
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values or None


TARGET_T_VALUES = _parse_target_t_values()
BIG_T_EXPERIMENT_VALUES = [11948, 14107]


def _is_modal_return_channel_error(exc: Exception) -> bool:
    msg = f"{type(exc).__name__}: {exc}"
    markers = (
        "Function call has expired",
        "APP_STATE_STOPPED",
        "ConflictError",
        "Deadline exceeded",
        "TimeoutError",
        "ConnectionError",
    )
    return any(marker in msg for marker in markers)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("build-essential", "ninja-build")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
    .env(
        {
            "CUDA_HOME": "/usr/local/cuda",
            "TRITON_PRINT_AUTOTUNING": "0",
            "TRITON_DUMP_ASSEMBLY": "0",
            "TRITON_KERNEL_DUMP": "0",
            "TRITON_CACHE_DUMP": "0",
            "TRITON_DEBUG": "0",
            "MLIR_ENABLE_DUMP": "0",
            "LLVM_IR_ENABLE_DUMP": "0",
        }
    )
)


def _quiet_compiler_dump_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "TRITON_DUMP_ASSEMBLY",
        "TRITON_KERNEL_DUMP",
        "TRITON_CACHE_DUMP",
        "TRITON_ENABLE_LLVM_DEBUG",
        "TRITON_LLVM_DEBUG_ONLY",
        "TRITON_DEBUG",
        "MLIR_ENABLE_DUMP",
        "LLVM_IR_ENABLE_DUMP",
        "LLVM_ENABLE_DUMP",
        "LLVM_VERBOSE_ASM",
        "TORCH_LOGS",
    ):
        env.pop(key, None)
        os.environ.pop(key, None)
    quiet_values = {
        "TRITON_PRINT_AUTOTUNING": "0",
        "TRITON_DUMP_ASSEMBLY": "0",
        "TRITON_KERNEL_DUMP": "0",
        "TRITON_CACHE_DUMP": "0",
        "TRITON_DEBUG": "0",
        "MLIR_ENABLE_DUMP": "0",
        "LLVM_IR_ENABLE_DUMP": "0",
        "LLVM_ENABLE_DUMP": "0",
        "LLVM_VERBOSE_ASM": "0",
    }
    env.update(quiet_values)
    os.environ.update(quiet_values)
    return env


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


def _get_cpu_us(event) -> float:
    for attr in ("self_cpu_time_total", "cpu_time_total"):
        value = getattr(event, attr, None)
        if value is not None:
            return float(value)
    return 0.0


def _classify_event(name: str) -> str:
    name_l = name.lower()
    if (
        "_fused_moe_gemm1_swiglu_kernel" in name_l
        or "_mapped_small_fused_moe_gemm1_swiglu_kernel" in name_l
        or "_t1_fused_gemm1_swiglu_kernel" in name_l
        or "_medium_no_swiglu_gemm1_kernel" in name_l
        or "gemm1_swiglu_kernel" in name_l
        or "cute_gemm1_mma" in name_l
        or "groupedgemm1kernel" in name_l
        or "grouped_gemm1" in name_l
        or "dequantize_gemm1_sorted_a" in name_l
        or "cute_gemm1_swiglu_epilogue" in name_l
    ):
        return "gemm1"
    if "_t1_fused_gemm2_reduce_kernel" in name_l:
        return "gemm2"
    if (
        "_fused_moe_gemm2_kernel" in name_l
        or "_fused_moe_gemm2_scatter_kernel" in name_l
        or "_fused_moe_gemm2_t901_kernel" in name_l
        or "gemm2_scatter_kernel" in name_l
        or "gemm2_scatter_large_kernel" in name_l
        or "gemm2_t14107" in name_l
        or "cute_gemm2_mma" in name_l
        or "cute_grouped_gemm" in name_l
        or "groupedgemm" in name_l
        or "grouped_gemm" in name_l
        or "cute_t14107" in name_l
        or "gemm2_reduce_t14107" in name_l
        or "fused_moe_gemm2_reduce_atomic" in name_l
        or "fused_moe_gemm2_single_local_direct" in name_l
    ):
        return "gemm2"
    if (
        "token_reduce_t14107" in name_l
        or "cute_token_reduce" in name_l
        or "pos_reduce_t14107" in name_l
        or "cute_pos_reduce" in name_l
        or "token_reduce_posmap" in name_l
        or "token_reduce_weighted" in name_l
        or "build_token_pos_map" in name_l
        or "token_reduce_counted" in name_l
        or "count_local_experts" in name_l
    ):
        return "routing"
    if "triton_ds_routing_kernel" in name_l or "triton_ds_routing_t1_local_kernel" in name_l:
        return "routing"
    if (
        "triton_sort_and_scatter_kernel" in name_l
        or "triton_sort_histogram_kernel" in name_l
        or "triton_sort_layout_kernel" in name_l
        or "triton_sort_scatter_kernel" in name_l
        or "triton_init_sorted_buffers_kernel" in name_l
        or "triton_dual_bucket_layout_kernel" in name_l
        or "triton_dual_sort_scatter_kernel" in name_l
        or "triton_small_expert_layout_kernel" in name_l
        or "triton_small_tile_offsets_kernel" in name_l
        or "triton_init_token_ids_kernel" in name_l
        or "triton_small_sort_scatter_kernel" in name_l
        or "triton_small_row_map_kernel" in name_l
    ):
        return "sorting"
    if any(x in name_l for x in ("copy", "zero_", "fill_", "memset", "memcpy", "cast_fp32_to_bf16")):
        return "memops"
    return "other"


def _classify_cpu_event(name: str) -> str:
    name_l = name.lower()
    if any(x in name_l for x in ("cudalaunchkernel", "cuda", "hip", "driverapi", "runtimeapi")):
        return "cuda_api"
    if any(x in name_l for x in ("aten::", "at::", "torch::", "triton_", "triton::")):
        return "framework"
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
        abs_path = os.path.join(root_dir, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(src.get("content", ""))
        if rel_path == entry_file:
            found_entry = True

    if not found_entry:
        raise RuntimeError(f"Cannot find source for entry file: {entry_file}")

    return os.path.join(root_dir, entry_file)


def _format_triton_config(cfg) -> str:
    if cfg is None:
        return "None"
    parts = []
    kwargs = getattr(cfg, "kwargs", None)
    if kwargs is not None:
        try:
            parts.append(f"kwargs={dict(kwargs)}")
        except Exception:
            parts.append(f"kwargs={kwargs!r}")
    for attr in ("num_warps", "num_stages", "num_ctas", "maxnreg"):
        value = getattr(cfg, attr, None)
        if value is not None:
            parts.append(f"{attr}={value}")
    return ", ".join(parts) if parts else repr(cfg)


def _log_named_triton_autotune(module, kernel_name: str, prefix: str, log) -> None:
    kernel = getattr(module, kernel_name, None)
    if kernel is None:
        log(f"{prefix}: kernel object not found")
        return

    log(f"{prefix} object: {type(kernel).__name__}")

    configs = getattr(kernel, "configs", None)
    if configs is not None:
        try:
            log(f"{prefix} candidate configs: {len(configs)}")
        except Exception:
            pass

    best_cfg = getattr(kernel, "best_config", None)
    if best_cfg is not None:
        log(f"{prefix} best_config: {_format_triton_config(best_cfg)}")

    cache = getattr(kernel, "cache", None)
    if isinstance(cache, dict):
        log(f"{prefix} autotune cache entries: {len(cache)}")
        for idx, (key, value) in enumerate(list(cache.items())[:4]):
            log(f"  cache[{idx}] key={key!r}")
            log(f"  cache[{idx}] value={_format_triton_config(value)}")
    elif cache is not None:
        log(f"{prefix} autotune cache type: {type(cache).__name__}")


def _log_t901_gemm2_autotune(module, log) -> None:
    _log_named_triton_autotune(module, "_fused_moe_gemm2_t901_kernel", "T901 autotune", log)
    _log_named_triton_autotune(module, "_fused_moe_gemm2_kernel", "Base GEMM2 autotune", log)


def _log_t1_autotune(module, log) -> None:
    _log_named_triton_autotune(module, "_t1_fused_gemm1_swiglu_kernel", "T=1 GEMM1 autotune", log)
    _log_named_triton_autotune(module, "_t1_fused_gemm2_reduce_kernel", "T=1 GEMM2 autotune", log)


def _log_small_t_gemm1_autotune(module, t: int, log) -> None:
    if t <= 64:
        _log_named_triton_autotune(
            module,
            "_small_medium_fused_moe_gemm1_swiglu_kernel",
            f"T={t} GEMM1 autotune",
            log,
        )
    elif t <= 128:
        _log_named_triton_autotune(
            module,
            "_medium_fused_moe_gemm1_swiglu_kernel",
            f"T={t} GEMM1 autotune",
            log,
        )
    else:
        _log_named_triton_autotune(
            module,
            "_fused_moe_gemm1_swiglu_kernel",
            f"T={t} GEMM1 autotune",
            log,
        )


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={VOLUME_MOUNT: trace_volume})
def run_profile(solution_json: str, remote_report_path: str = REMOTE_REPORT_PATH) -> str:
    import importlib.util

    quiet_env = _quiet_compiler_dump_env()

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
    spec_info = solution.get("spec", {})
    entry_point = spec_info.get("entry_point", "kernel.py::kernel")
    entry_symbol = entry_point.split("::")[1] if "::" in entry_point else "kernel"
    log(
        f"Solution spec: language={spec_info.get('language', 'unknown')}, "
        f"entry={entry_point}, binding={spec_info.get('binding', 'default')}, "
        f"source_count={len(solution.get('sources', []))}"
    )

    tmp_dir = tempfile.mkdtemp(prefix="moe_profile_")
    kernel_path = _materialize_solution_sources(solution, tmp_dir)

    spec = importlib.util.spec_from_file_location("kernel_module", kernel_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    kernel_fn = getattr(module, entry_symbol)
    log("Kernel loaded")
    if hasattr(module, "get_runtime_status"):
        try:
            log(f"Runtime status: {module.get_runtime_status()}")
        except Exception as exc:
            log(f"Runtime status unavailable: {exc!r}")

    # Optional NCU availability
    ncu_path = _find_ncu_binary()
    if ncu_path and ENABLE_NCU_COUNTERS:
        ncu_ver = subprocess.run([ncu_path, "--version"], capture_output=True, text=True, check=False)
        log(f"NCU: {ncu_ver.stdout.strip() or 'available'}")
    elif not ncu_path:
        log("NCU: not found (skip hardware counters)")
    else:
        log("NCU: available (hardware counters disabled; set YJL_ENABLE_NCU_COUNTERS=1 to enable)")

    # Shapes from definition
    H = int(getattr(module, "H", 7168))
    I_SIZE = int(getattr(module, "I_SIZE", 2048))
    E_LOCAL = int(getattr(module, "E_LOCAL", 32))
    TOP_K = int(getattr(module, "TOP_K", 8))
    QBLOCK = int(getattr(module, "QBLOCK", 128))
    BLOCK_M = int(getattr(module, "BLOCK_M_LARGE", 64))

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

    def infer_padding_stats(args) -> dict[str, int | None]:
        # Uses the same Triton kernels as runtime path, so this is exact for current input.
        stats = {
            "total_blocks": None,
            "num_padded": None,
            "num_local_rows": None,
            "tail_rows": None,
            "tail_experts": None,
            "block_m": None,
            "dual_threshold": None,
            "dual_small_experts": None,
            "dual_small_rows": None,
            "dual_small_padded": None,
            "dual_normal_experts": None,
            "dual_normal_rows": None,
            "dual_normal_padded": None,
        }
        if not hasattr(module, "ds_routing"):
            return stats
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

            if hasattr(module, "_select_block_m"):
                block_m = int(module._select_block_m(t * TOP_K))
            else:
                block_m = BLOCK_M
            stats["block_m"] = block_m

            local_end = local_expert_offset + E_LOCAL
            local_mask = (topk_idx >= local_expert_offset) & (topk_idx < local_end)
            local_ids = (topk_idx[local_mask] - local_expert_offset).to(torch.int64)
            num_local_rows = int(local_ids.numel())
            stats["num_local_rows"] = num_local_rows
            if num_local_rows > 0:
                local_counts = torch.bincount(local_ids, minlength=E_LOCAL)
            else:
                local_counts = torch.zeros((E_LOCAL,), dtype=torch.int64, device="cuda")
            tail_experts = int(((local_counts > 0) & ((local_counts % block_m) != 0)).sum().item())
            stats["tail_experts"] = tail_experts
            dual_threshold = int(getattr(module, "DUAL_BUCKET_THRESHOLD", 0))
            dual_small_block = int(getattr(module, "DUAL_SMALL_BLOCK_M", 16))
            dual_normal_block = int(getattr(module, "DUAL_NORMAL_BLOCK_M", 64))
            if dual_threshold > 0:
                small_mask = (local_counts > 0) & (local_counts <= dual_threshold)
                normal_mask = local_counts > dual_threshold
                small_rows = int(local_counts[small_mask].sum().item())
                normal_rows = int(local_counts[normal_mask].sum().item())
                small_padded = int((((local_counts[small_mask] + (dual_small_block - 1)) // dual_small_block) * dual_small_block).sum().item())
                normal_padded = int((((local_counts[normal_mask] + (dual_normal_block - 1)) // dual_normal_block) * dual_normal_block).sum().item())
                stats["dual_threshold"] = dual_threshold
                stats["dual_small_experts"] = int(small_mask.sum().item())
                stats["dual_small_rows"] = small_rows
                stats["dual_small_padded"] = small_padded
                stats["dual_normal_experts"] = int(normal_mask.sum().item())
                stats["dual_normal_rows"] = normal_rows
                stats["dual_normal_padded"] = normal_padded

                use_exact_dispatch = False
                if hasattr(module, "_use_exact_workload_dispatch"):
                    max_pid_m = (t * TOP_K + E_LOCAL * block_m) // block_m
                    use_exact_dispatch = bool(module._use_exact_workload_dispatch(t, max_pid_m))
                if hasattr(module, "_use_dual_expert_bucket_large_t_override") and bool(
                    module._use_dual_expert_bucket_large_t_override(t, block_m, use_exact_dispatch)
                ):
                    small_tail_experts = int(((local_counts[small_mask] % dual_small_block) != 0).sum().item())
                    normal_tail_experts = int(((local_counts[normal_mask] % dual_normal_block) != 0).sum().item())
                    stats["total_blocks"] = (small_padded // dual_small_block) + (normal_padded // dual_normal_block)
                    stats["num_padded"] = small_padded + normal_padded
                    stats["tail_rows"] = stats["num_padded"] - num_local_rows
                    stats["tail_experts"] = small_tail_experts + normal_tail_experts
                    return stats

            max_padded = t * TOP_K + E_LOCAL * block_m
            sorted_token_ids = torch.empty((max_padded,), dtype=torch.int64, device="cuda")
            sorted_weights = torch.empty((max_padded,), dtype=torch.float32, device="cuda")
            scatter_map = torch.empty((t * TOP_K,), dtype=torch.int32, device="cuda")
            block_offsets = torch.empty((E_LOCAL + 1,), dtype=torch.int32, device="cuda")
            total_blocks = torch.empty((1,), dtype=torch.int32, device="cuda")
            parallel_sort_min_tiles = int(getattr(module, "PARALLEL_SORT_MIN_TILES", 128))
            sort_block_items = int(getattr(module, "SORT_BLOCK_ITEMS", 256))
            if hasattr(module, "parallel_sort_and_scatter") and ((t * TOP_K + sort_block_items - 1) // sort_block_items) >= parallel_sort_min_tiles:
                num_tiles = (t * TOP_K + sort_block_items - 1) // sort_block_items
                partial_counts = torch.empty((num_tiles, E_LOCAL), dtype=torch.int32, device="cuda")
                tile_offsets = torch.empty((num_tiles, E_LOCAL), dtype=torch.int32, device="cuda")
                counts_workspace = torch.empty((E_LOCAL,), dtype=torch.int32, device="cuda")
                module.parallel_sort_and_scatter(
                    topk_idx,
                    topk_wts,
                    sorted_token_ids,
                    sorted_weights,
                    scatter_map,
                    block_offsets,
                    total_blocks,
                    partial_counts,
                    tile_offsets,
                    local_expert_offset,
                    t,
                    block_m,
                    max_padded,
                    counts_workspace,
                )
            elif hasattr(module, "triton_sort_and_scatter_kernel"):
                counts_workspace = torch.empty((E_LOCAL,), dtype=torch.int32, device="cuda")
                module.triton_sort_and_scatter_kernel[(1,)](
                    topk_idx,
                    topk_wts,
                    sorted_token_ids,
                    sorted_weights,
                    scatter_map,
                    block_offsets,
                    total_blocks,
                    counts_workspace,
                    local_expert_offset,
                    t,
                    TOP_K,
                    E_LOCAL,
                    block_m,
                    max_padded,
                    num_warps=8,
                )
            else:
                return stats
            tb = int(total_blocks.item())
            num_padded = tb * block_m
            stats["total_blocks"] = tb
            stats["num_padded"] = num_padded
            if stats["num_local_rows"] is not None:
                stats["tail_rows"] = num_padded - stats["num_local_rows"]
            return stats
        except Exception:
            return stats

    trace_inputs_by_t = build_trace_inputs_by_t()
    if trace_inputs_by_t:
        run_t_values = sorted(trace_inputs_by_t.keys())
        log(f"Using trace T values: {run_t_values}")
    else:
        run_t_values = DEFAULT_T_VALUES
        log(f"Using fallback T values: {run_t_values}")

    if TARGET_T_VALUES:
        run_t_values = [t for t in run_t_values if t in TARGET_T_VALUES]
        log(f"Filtered target T values: {run_t_values}")
    else:
        run_t_values = [t for t in run_t_values if t in BIG_T_EXPERIMENT_VALUES]
        log(f"No target T filter; profiling big-T experiment values only: {run_t_values}")

    for t in run_t_values:
        log("")
        log("=" * 96)
        log(f"T = {t}")
        log("=" * 96)
        # Previously: if t>=901: continue  — removed to enable T=901+ profiling
        if t in trace_inputs_by_t:
            args = trace_inputs_by_t[t]
            log(f"Input source: trace set (T={t})")
        else:
            args = build_inputs_synthetic(t)
            log(f"Input source: synthetic fallback (T={t})")
        output = args[-1]

        # if t!=1:
        #     continue

        padding_stats = infer_padding_stats(args)
        total_blocks = padding_stats["total_blocks"]
        num_padded = padding_stats["num_padded"]
        num_local_rows = padding_stats["num_local_rows"]
        tail_rows = padding_stats["tail_rows"]
        tail_experts = padding_stats["tail_experts"]
        block_m = padding_stats["block_m"]
        if total_blocks is not None:
            log(
                "Inferred padding: "
                f"block_m={block_m}, total_blocks={total_blocks}, num_padded={num_padded}, "
                f"num_local_rows={num_local_rows}, tail_rows={tail_rows}, tail_experts={tail_experts}"
            )
            if padding_stats["dual_threshold"] is not None:
                log(
                    "Dual-bucket stats: "
                    f"threshold<={padding_stats['dual_threshold']}, "
                    f"small_experts={padding_stats['dual_small_experts']}, "
                    f"small_rows={padding_stats['dual_small_rows']}, "
                    f"small_padded={padding_stats['dual_small_padded']}, "
                    f"normal_experts={padding_stats['dual_normal_experts']}, "
                    f"normal_rows={padding_stats['dual_normal_rows']}, "
                    f"normal_padded={padding_stats['dual_normal_padded']}"
                )
        else:
            log("Inferred padding: unavailable (fallback to N/A FLOP metrics)")

        # Warmup (JIT/autotune)
        for _ in range(WARMUP):
            output.zero_()
            kernel_fn(*args)
        torch.cuda.synchronize()
        if hasattr(module, "get_runtime_status"):
            try:
                log(f"Runtime status after warmup: {module.get_runtime_status()}")
            except Exception as exc:
                log(f"Runtime status after warmup unavailable: {exc!r}")
        if t == 1:
            _log_t1_autotune(module, log)
        if t in (32, 52, 80):
            _log_small_t_gemm1_autotune(module, t, log)
        if t == 901:
            _log_t901_gemm2_autotune(module, log)

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
        prof_kwargs = dict(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
        )
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
        cpu_totals_us = {
            "cuda_api": 0.0,
            "framework": 0.0,
            "memops": 0.0,
            "other": 0.0,
        }
        top_kernels = []
        top_cpu_events = []
        for event in prof.key_averages():
            cuda_us = _get_cuda_us(event)
            cpu_us = _get_cpu_us(event)
            if cuda_us <= 0:
                if cpu_us <= 0:
                    continue
            if cuda_us > 0:
                category = _classify_event(event.key)
                totals_us[category] += cuda_us
                top_kernels.append((event.key, category, cuda_us / ITERS, event.count / ITERS))
            if cpu_us > 0:
                cpu_category = _classify_cpu_event(event.key)
                cpu_totals_us[cpu_category] += cpu_us
                top_cpu_events.append((event.key, cpu_category, cpu_us / ITERS, event.count / ITERS))

        total_cuda_ms = sum(totals_us.values()) / ITERS / 1000.0
        cpu_overhead_ms = max(0.0, wall_ms - total_cuda_ms)
        total_cpu_profile_ms = sum(cpu_totals_us.values()) / ITERS / 1000.0

        def pct(us_val: float) -> float:
            denom = sum(totals_us.values())
            return 100.0 * us_val / denom if denom > 0 else 0.0

        def cpu_pct(us_val: float) -> float:
            denom = sum(cpu_totals_us.values())
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
        log("CPU/API breakdown (per iter):")
        log(f"  CUDA API : {cpu_totals_us['cuda_api']/ITERS/1000.0:8.3f} ms ({cpu_pct(cpu_totals_us['cuda_api']):5.1f}%)")
        log(f"  Framework: {cpu_totals_us['framework']/ITERS/1000.0:8.3f} ms ({cpu_pct(cpu_totals_us['framework']):5.1f}%)")
        log(f"  MemOps   : {cpu_totals_us['memops']/ITERS/1000.0:8.3f} ms ({cpu_pct(cpu_totals_us['memops']):5.1f}%)")
        log(f"  Other    : {cpu_totals_us['other']/ITERS/1000.0:8.3f} ms ({cpu_pct(cpu_totals_us['other']):5.1f}%)")
        log(f"  Total profiled CPU: {total_cpu_profile_ms:.3f} ms/iter")

        top_kernels.sort(key=lambda x: -x[2])
        log("Top kernels:")
        for name, category, us, count in top_kernels[:8]:
            log(f"  [{category:7s}] {us:8.2f} us x {count:.1f} | {name[:90]}")
        top_cpu_events.sort(key=lambda x: -x[2])
        log("Top CPU/API events:")
        for name, category, us, count in top_cpu_events[:8]:
            log(f"  [{category:9s}] {us:8.2f} us x {count:.1f} | {name[:90]}")

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
            gemm1_tflops_str = f"{gemm1_tflops:.1f}T" if gemm1_tflops is not None else "N/A"
            gemm2_tflops_str = f"{gemm2_tflops:.1f}T" if gemm2_tflops is not None else "N/A"
            log(
                f"Effective TFLOPS (actual num_padded={num_padded}): "
                f"GEMM1={gemm1_tflops_str}, GEMM2={gemm2_tflops_str}"
            )
        else:
            log("Effective TFLOPS: N/A (num_padded unavailable)")

        summary_rows.append(
            {
                "T": t,
                "wall_ms": wall_ms,
                "cuda_ms": total_cuda_ms,
                "cpu_ms": cpu_overhead_ms,
                "cpu_profile_ms": total_cpu_profile_ms,
                "cpu_api_ms": cpu_totals_us["cuda_api"] / ITERS / 1000.0,
                "routing_ms": totals_us["routing"] / ITERS / 1000.0,
                "sorting_ms": totals_us["sorting"] / ITERS / 1000.0,
                "gemm1_ms": totals_us["gemm1"] / ITERS / 1000.0,
                "gemm2_ms": totals_us["gemm2"] / ITERS / 1000.0,
                "num_local_rows": num_local_rows if num_local_rows is not None else -1,
                "num_padded": num_padded if num_padded is not None else -1,
                "tail_rows": tail_rows if tail_rows is not None else -1,
                "tail_experts": tail_experts if tail_experts is not None else -1,
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
        f"{'T':>6s} | {'Wall':>7s} | {'CUDA':>7s} | {'CPU':>7s} | {'CPU API':>7s} | {'CPU Prf':>7s} | {'Route':>7s} | "
        f"{'Sort':>7s} | {'GEMM1':>7s} | {'GEMM2':>7s} | {'local':>8s} | {'num_pad':>8s} | "
        f"{'tail':>8s} | {'tail_e':>6s} | {'G1 TF':>8s} | {'G2 TF':>8s}"
    )
    for row in summary_rows:
        g1 = f"{row['gemm1_tflops']:.1f}" if row["gemm1_tflops"] >= 0 else "N/A"
        g2 = f"{row['gemm2_tflops']:.1f}" if row["gemm2_tflops"] >= 0 else "N/A"
        nlocal = str(row["num_local_rows"]) if row["num_local_rows"] >= 0 else "N/A"
        npad = str(row["num_padded"]) if row["num_padded"] >= 0 else "N/A"
        tail = str(row["tail_rows"]) if row["tail_rows"] >= 0 else "N/A"
        tail_e = str(row["tail_experts"]) if row["tail_experts"] >= 0 else "N/A"
        log(
            f"{row['T']:6d} | {row['wall_ms']:7.3f} | {row['cuda_ms']:7.3f} | {row['cpu_ms']:7.3f} | "
            f"{row['cpu_api_ms']:7.3f} | {row['cpu_profile_ms']:7.3f} | "
            f"{row['routing_ms']:7.3f} | {row['sorting_ms']:7.3f} | {row['gemm1_ms']:7.3f} | "
            f"{row['gemm2_ms']:7.3f} | {nlocal:>8s} | {npad:>8s} | {tail:>8s} | {tail_e:>6s} | {g1:>8s} | {g2:>8s}"
        )

    # Optional hardware counters
    if ncu_path and ENABLE_NCU_COUNTERS:
        log("")
        log("=" * 96)
        log("NCU COUNTERS (if available in runtime)")
        log("=" * 96)
        ncu_script_path = os.path.join(tmp_dir, "ncu_once.py")
        with open(ncu_script_path, "w", encoding="utf-8") as f:
            f.write(
                "import os\n"
                "for key in ('TRITON_DUMP_ASSEMBLY','TRITON_KERNEL_DUMP','TRITON_CACHE_DUMP','TRITON_ENABLE_LLVM_DEBUG','TRITON_LLVM_DEBUG_ONLY','TRITON_DEBUG','MLIR_ENABLE_DUMP','LLVM_IR_ENABLE_DUMP','LLVM_ENABLE_DUMP','LLVM_VERBOSE_ASM','TORCH_LOGS'):\n"
                "    os.environ.pop(key, None)\n"
                "os.environ.update({'TRITON_PRINT_AUTOTUNING':'0','TRITON_DUMP_ASSEMBLY':'0','TRITON_KERNEL_DUMP':'0','TRITON_CACHE_DUMP':'0','TRITON_DEBUG':'0','MLIR_ENABLE_DUMP':'0','LLVM_IR_ENABLE_DUMP':'0','LLVM_ENABLE_DUMP':'0','LLVM_VERBOSE_ASM':'0'})\n"
                "import torch, importlib.util\n"
                f"spec=importlib.util.spec_from_file_location('km','{kernel_path}')\n"
                "m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)\n"
                f"k=getattr(m,'{entry_symbol}')\n"
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
            "regex:.*(fused_moe|fused_moe_gemm2_reduce_atomic|fused_moe_gemm2_single_local_direct|token_reduce_counted|token_reduce_posmap|token_reduce_weighted|build_token_pos_map|count_local_experts|t1_fused_gemm2_reduce|triton_ds_routing|triton_sort_|triton_dual_|gemm1_swiglu_kernel|dequantize_gemm1_sorted_a|cute_gemm1_swiglu_epilogue|cute_gemm1_mma|groupedgemm1kernel|grouped_gemm1|gemm2_scatter_kernel|gemm2_scatter_large_kernel|gemm2_t14107|cute_gemm2_mma|cute_grouped_gemm|grouped_gemm|cute_t14107|token_reduce_t14107|cute_token_reduce|pos_reduce_t14107|cute_pos_reduce|dequant_fp8_blockscale_kernel).*",
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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False, env=quiet_env)
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

    report = "\n".join(logs)
    try:
        Path(remote_report_path).write_text(report, encoding="utf-8")
        if remote_report_path != REMOTE_REPORT_PATH:
            Path(REMOTE_REPORT_PATH).write_text(report, encoding="utf-8")
        trace_volume.commit()
    except Exception as exc:
        print(f"Warning: failed to persist remote report: {exc!r}")
    return report


@app.function(image=image, timeout=300, volumes={VOLUME_MOUNT: trace_volume})
def read_latest_remote_report(remote_report_path: str = REMOTE_REPORT_PATH) -> str:
    path = Path(remote_report_path)
    if not path.exists():
        raise FileNotFoundError(f"Remote report not found: {remote_report_path}")
    return path.read_text(encoding="utf-8")


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
    import time

    remote_report_path = f"{VOLUME_MOUNT}/ncu_profiler_yjl_{int(time.time())}.txt"

    print("Running profile on Modal B200...")
    try:
        report = run_profile.remote(solution_json, remote_report_path)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        if not _is_modal_return_channel_error(exc):
            raise
        print(f"Modal return channel expired after profiling; reading persisted report from volume. ({msg})")
        last_read_error = None
        max_read_attempts = 30
        for attempt in range(1, max_read_attempts + 1):
            try:
                report = read_latest_remote_report.remote(remote_report_path)
                break
            except Exception as read_exc:
                last_read_error = read_exc
                print(
                    "Persisted report is not available yet; "
                    f"retrying ({attempt}/{max_read_attempts}). ({type(read_exc).__name__}: {read_exc})"
                )
                time.sleep(20)
        else:
            raise RuntimeError(
                "Modal return channel failed and persisted profiler report could not be read"
            ) from last_read_error

    out_path = PROJECT_ROOT / "ncu_profiler_yjl.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"Saved report: {out_path}")
    if os.environ.get("YJL_PRINT_REPORT_LOCAL", "0") == "1":
        print(report)
