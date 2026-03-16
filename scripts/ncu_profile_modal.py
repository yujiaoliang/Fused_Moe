"""
MoE Kernel Profiling on B200 — timing breakdown + analytical roofline.

Collects:
  1. Per-phase CUDA timing: GEMM1 (Triton), GEMM2 (Triton), routing, sorting, other
  2. Achieved TFLOPS for GEMM1 and GEMM2
  3. Analytical roofline: compute-bound vs memory-bound assessment
  4. NCU hardware counters (if available)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal

app = modal.App("moe-ncu-profile")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
VOLUME_MOUNT = "/data"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("flashinfer-bench", "torch", "triton", "numpy")
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={VOLUME_MOUNT: trace_volume})
def run_profile(solution_json: str) -> str:
    import torch
    import json as json_mod
    import importlib.util
    import tempfile
    import os
    import subprocess

    lines = []
    def log(msg):
        lines.append(str(msg))
        print(msg)

    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"PyTorch: {torch.__version__}")
    log(f"Triton: {__import__('triton').__version__}")

    # ── Load kernel ──
    sol_data = json_mod.loads(solution_json)
    sources = sol_data.get('sources', [])
    entry_point = sol_data.get('spec', {}).get('entry_point', 'kernel.py::kernel')
    entry_file = entry_point.split('::')[0]

    source_code = None
    for src in sources:
        if isinstance(src, dict) and src.get('path', '') == entry_file:
            source_code = src.get('content', '')
            break
    if source_code is None and sources:
        src = sources[0]
        source_code = src.get('content', str(src)) if isinstance(src, dict) else str(src)

    tmp_dir = tempfile.mkdtemp()
    kernel_path = os.path.join(tmp_dir, 'kernel.py')
    with open(kernel_path, 'w') as f:
        f.write(source_code)

    spec = importlib.util.spec_from_file_location("kernel_module", kernel_path)
    kernel_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernel_module)
    kernel_fn = kernel_module.kernel
    log("Kernel loaded")

    # ── Check NCU ──
    ncu_path = None
    for p in ['/usr/local/cuda/bin/ncu', '/usr/bin/ncu', '/opt/nvidia/nsight-compute/ncu']:
        if os.path.exists(p):
            ncu_path = p
            break
    if ncu_path is None:
        try:
            r = subprocess.run(['which', 'ncu'], capture_output=True, text=True)
            if r.returncode == 0:
                ncu_path = r.stdout.strip()
        except Exception:
            pass
    if ncu_path:
        r = subprocess.run([ncu_path, '--version'], capture_output=True, text=True)
        log(f"NCU: {r.stdout.strip()}")
    else:
        log("NCU: not found (will skip hardware counter profiling)")

    # ── B200 approximate peak performance ──
    # Dense (no sparsity): FP8 ~2500 TFLOPS, TF32 ~1250 TFLOPS, HBM3e ~8 TB/s
    PEAK_FP8 = 2500   # TFLOPS, dense
    PEAK_TF32 = 1250  # TFLOPS, dense
    PEAK_BW = 8000    # GB/s
    RIDGE_FP8 = PEAK_FP8 * 1e3 / PEAK_BW    # FLOP/B
    RIDGE_TF32 = PEAK_TF32 * 1e3 / PEAK_BW  # FLOP/B

    log(f"\n{'='*90}")
    log(f"B200 Peak (dense, approx): FP8={PEAK_FP8} TFLOPS, TF32={PEAK_TF32} TFLOPS, BW={PEAK_BW} GB/s")
    log(f"Roofline ridge: FP8={RIDGE_FP8:.0f} FLOP/B, TF32={RIDGE_TF32:.0f} FLOP/B")
    log(f"{'='*90}")

    H = 7168
    I_SIZE = 2048
    W13_N = 4096
    E_LOCAL = 32
    TOP_K = 8
    QBLOCK = 128

    T_VALUES = [7, 14, 64, 128, 512, 1024, 4096]
    NUM_ITERS = 50
    WARMUP = 15

    results = []

    for T in T_VALUES:
        log(f"\n{'─'*90}")
        log(f"T = {T}")
        log(f"{'─'*90}")

        # ── Create inputs ──
        routing_logits = torch.randn(T, 256, dtype=torch.float32, device='cuda')
        routing_bias = torch.randn(256, dtype=torch.bfloat16, device='cuda')

        A_p = torch.randn(T, H, dtype=torch.float32, device='cuda')
        A_amax = A_p.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
        hidden_states = (A_p * 448.0 / A_amax).to(torch.float8_e4m3fn)
        hidden_states_scale = (A_amax / 448.0).expand(T, H // QBLOCK).t().contiguous().float()

        gemm1_weights = torch.randint(-10, 10, (E_LOCAL, W13_N, H), dtype=torch.int8, device='cuda').to(torch.float8_e4m3fn)
        gemm1_weights_scale = torch.rand(E_LOCAL, W13_N // QBLOCK, H // QBLOCK, dtype=torch.float32, device='cuda') + 0.5

        gemm2_weights = torch.randint(-10, 10, (E_LOCAL, H, I_SIZE), dtype=torch.int8, device='cuda').to(torch.float8_e4m3fn)
        gemm2_weights_scale = torch.rand(E_LOCAL, H // QBLOCK, I_SIZE // QBLOCK, dtype=torch.float32, device='cuda') + 0.5

        local_expert_offset = torch.tensor(0, dtype=torch.int32, device='cuda')
        routed_scaling_factor = torch.tensor(1.0, dtype=torch.float32, device='cuda')
        output = torch.zeros(T, H, dtype=torch.bfloat16, device='cuda')

        all_args = [routing_logits, routing_bias, hidden_states, hidden_states_scale,
                    gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
                    local_expert_offset, routed_scaling_factor, output]

        # ── Warmup (Triton JIT + autotune) ──
        for _ in range(WARMUP):
            output.zero_()
            kernel_fn(*all_args)
        torch.cuda.synchronize()

        # ── Phase 1: CUDA Event wall-clock timing ──
        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start_ev.record()
        for _ in range(NUM_ITERS):
            output.zero_()
            kernel_fn(*all_args)
        end_ev.record()
        torch.cuda.synchronize()
        total_wall_ms = start_ev.elapsed_time(end_ev) / NUM_ITERS
        log(f"  Wall time: {total_wall_ms:.3f} ms/iter")

        # ── Phase 2: torch.profiler kernel breakdown ──
        from torch.profiler import profile, ProfilerActivity

        # acc_events=True to accumulate across iterations (PyTorch 2.10+)
        prof_kwargs = dict(activities=[ProfilerActivity.CUDA], record_shapes=False)
        try:
            # PyTorch 2.10+ supports acc_events
            with profile(**prof_kwargs, acc_events=True) as prof:
                for _ in range(NUM_ITERS):
                    output.zero_()
                    kernel_fn(*all_args)
        except TypeError:
            with profile(**prof_kwargs) as prof:
                for _ in range(NUM_ITERS):
                    output.zero_()
                    kernel_fn(*all_args)
        torch.cuda.synchronize()

        events = prof.key_averages()

        # Helper: PyTorch 2.10+ renamed self_cuda_time_total → cuda_time_total
        def get_cuda_us(e):
            for attr in ['self_cuda_time_total', 'cuda_time_total', 'self_device_time_total', 'device_time_total']:
                v = getattr(e, attr, None)
                if v is not None:
                    return v
            return 0

        phase_us = {
            'gemm1': 0,
            'gemm2': 0,
            'route': 0,
            'sort_hist': 0,
            'sort_layout': 0,
            'sort_init': 0,
            'sort_scatter': 0,
            'sort_fallback': 0,
            'reduce': 0,
            'copy': 0,
            'other': 0,
        }
        top_kernels = []

        for e in events:
            t = get_cuda_us(e)  # us, summed across NUM_ITERS
            if t <= 0:
                continue
            name = e.key
            nl = name.lower()

            if '_t1_fused_gemm2_reduce' in nl or 'token_reduce' in nl:
                phase_us['reduce'] += t
                top_kernels.append(('REDUCE', name[:80], t / NUM_ITERS, e.count / NUM_ITERS))
            elif 'triton_sort_histogram_kernel' in nl:
                phase_us['sort_hist'] += t
                top_kernels.append(('S-HIST', name[:80], t / NUM_ITERS, e.count / NUM_ITERS))
            elif 'triton_sort_layout_kernel' in nl:
                phase_us['sort_layout'] += t
                top_kernels.append(('S-LAY', name[:80], t / NUM_ITERS, e.count / NUM_ITERS))
            elif 'triton_init_sorted_buffers_kernel' in nl:
                phase_us['sort_init'] += t
                top_kernels.append(('S-INIT', name[:80], t / NUM_ITERS, e.count / NUM_ITERS))
            elif 'triton_sort_scatter_kernel' in nl:
                phase_us['sort_scatter'] += t
                top_kernels.append(('S-SCAT', name[:80], t / NUM_ITERS, e.count / NUM_ITERS))
            elif 'triton_sort_and_scatter_kernel' in nl:
                phase_us['sort_fallback'] += t
                top_kernels.append(('S-FALL', name[:80], t / NUM_ITERS, e.count / NUM_ITERS))
            elif 'triton_ds_routing' in nl or 'ds_routing_t1_local' in nl:
                phase_us['route'] += t
                top_kernels.append(('ROUTE', name[:80], t / NUM_ITERS, e.count / NUM_ITERS))
            elif 'gemm1_swiglu' in nl or '_fused_moe_gemm1' in nl or '_t1_generic_gemm1' in nl:
                phase_us['gemm1'] += t
                top_kernels.append(('GEMM1', name[:80], t / NUM_ITERS, e.count / NUM_ITERS))
            elif 'gemm2_scatter' in nl or '_fused_moe_gemm2' in nl or '_t1_generic_gemm2' in nl:
                phase_us['gemm2'] += t
                top_kernels.append(('GEMM2', name[:80], t / NUM_ITERS, e.count / NUM_ITERS))
            elif any(k in nl for k in ['sigmoid', 'topk', 'masked_fill', 'gather',
                                        'scatter_', 'mul_', 'div_', 'add_']):
                phase_us['route'] += t
            elif any(k in nl for k in ['argsort', 'sort', 'cumsum', 'repeat_interleave',
                                        'arange', 'ones_like', 'full']):
                phase_us['sort_fallback'] += t
            elif any(k in nl for k in ['copy', 'zero', 'fill', 'memset', 'memcpy']):
                phase_us['copy'] += t
            else:
                phase_us['other'] += t
                if t / NUM_ITERS > 5.0:  # >5us per iter — notable
                    top_kernels.append(('other', name[:80], t / NUM_ITERS, e.count / NUM_ITERS))

        gemm1_us = phase_us['gemm1']
        gemm2_us = phase_us['gemm2']
        routing_us = phase_us['route']
        sort_hist_us = phase_us['sort_hist']
        sort_layout_us = phase_us['sort_layout']
        sort_init_us = phase_us['sort_init']
        sort_scatter_us = phase_us['sort_scatter']
        sort_fallback_us = phase_us['sort_fallback']
        sorting_us = sort_hist_us + sort_layout_us + sort_init_us + sort_scatter_us + sort_fallback_us
        reduce_us = phase_us['reduce']
        copy_us = phase_us['copy']
        other_us = phase_us['other']
        total_cuda_us = sum(phase_us.values())

        log(f"\n  CUDA breakdown (per iter):")
        def pct(x): return 100 * x / total_cuda_us if total_cuda_us > 0 else 0
        log(f"    GEMM1 (Triton FP8):  {gemm1_us/NUM_ITERS/1000:8.3f} ms  ({pct(gemm1_us):5.1f}%)")
        log(f"    GEMM2 (Triton TF32): {gemm2_us/NUM_ITERS/1000:8.3f} ms  ({pct(gemm2_us):5.1f}%)")
        log(f"    Routing:             {routing_us/NUM_ITERS/1000:8.3f} ms  ({pct(routing_us):5.1f}%)")
        log(f"    Sort total:          {sorting_us/NUM_ITERS/1000:8.3f} ms  ({pct(sorting_us):5.1f}%)")
        log(f"      histogram:         {sort_hist_us/NUM_ITERS/1000:8.3f} ms  ({pct(sort_hist_us):5.1f}%)")
        log(f"      layout:            {sort_layout_us/NUM_ITERS/1000:8.3f} ms  ({pct(sort_layout_us):5.1f}%)")
        log(f"      init:              {sort_init_us/NUM_ITERS/1000:8.3f} ms  ({pct(sort_init_us):5.1f}%)")
        log(f"      scatter:           {sort_scatter_us/NUM_ITERS/1000:8.3f} ms  ({pct(sort_scatter_us):5.1f}%)")
        log(f"      fallback:          {sort_fallback_us/NUM_ITERS/1000:8.3f} ms  ({pct(sort_fallback_us):5.1f}%)")
        log(f"    Token reduce:        {reduce_us/NUM_ITERS/1000:8.3f} ms  ({pct(reduce_us):5.1f}%)")
        log(f"    Copy/zero/memset:    {copy_us/NUM_ITERS/1000:8.3f} ms  ({pct(copy_us):5.1f}%)")
        log(f"    Other:               {other_us/NUM_ITERS/1000:8.3f} ms  ({pct(other_us):5.1f}%)")
        log(f"    Total CUDA:          {total_cuda_us/NUM_ITERS/1000:8.3f} ms")
        log(f"    CPU overhead:        {total_wall_ms - total_cuda_us/NUM_ITERS/1000:8.3f} ms"
            f"  ({100*(total_wall_ms - total_cuda_us/NUM_ITERS/1000)/total_wall_ms:.1f}%)"
            if total_wall_ms > 0 else "")

        if top_kernels:
            log(f"\n  Kernel details:")
            for cat, name, us, cnt in sorted(top_kernels, key=lambda x: -x[2]):
                log(f"    [{cat:5s}] {us:8.1f} us x {cnt:.0f} | {name}")

        # ── Phase 3: Analytical Roofline ──
        # With random routing: ~uniform distribution → avg T*8/32 = T/4 tokens per expert
        avg_m = T * TOP_K / E_LOCAL
        # But not all experts may be active for small T
        active_experts = min(E_LOCAL, T * TOP_K)  # upper bound
        # Padded total across all experts
        total_padded = 0
        tokens_per_exp = max(1, int(avg_m))
        for _ in range(min(active_experts, E_LOCAL)):
            total_padded += ((tokens_per_exp + 63) // 64) * 64
        if total_padded == 0:
            total_padded = 64

        # GEMM1: [M, 4096] = [M, 7168] @ [7168, 4096] in FP8 × FP8
        # (both W1 and W3 computed together, so N=4096)
        gemm1_flops = 2.0 * total_padded * W13_N * H

        # Bytes: A (fp8) + B (fp8) + scales + C (fp32)
        gemm1_bytes = (total_padded * H * 1           # A reads (fp8)
                     + active_experts * W13_N * H * 1  # B reads (fp8, one expert at a time)
                     + total_padded * I_SIZE * 4)      # C writes (fp32, SwiGLU output)
        gemm1_ai = gemm1_flops / gemm1_bytes if gemm1_bytes > 0 else 0

        # GEMM2: [M, 7168] = [M, 2048] @ [2048, 7168] in FP32 × FP8
        gemm2_flops = 2.0 * total_padded * H * I_SIZE
        gemm2_bytes = (total_padded * I_SIZE * 4        # A reads (fp32)
                     + active_experts * H * I_SIZE * 1   # B reads (fp8)
                     + total_padded * H * 4)            # expert_out writes (fp32)
        gemm2_ai = gemm2_flops / gemm2_bytes if gemm2_bytes > 0 else 0
        reduce_bytes = (total_padded * H * 4            # expert_out reads (fp32)
                      + T * TOP_K * 4                    # scatter_map reads (int32)
                      + T * H * 2)                       # output writes (bf16)

        gemm1_bound = 'COMPUTE' if gemm1_ai > RIDGE_FP8 else 'MEMORY'
        gemm2_bound = 'COMPUTE' if gemm2_ai > RIDGE_TF32 else 'MEMORY'

        # Achieved TFLOPS
        gemm1_achieved = gemm1_flops / (gemm1_us / NUM_ITERS * 1e-6) / 1e12 if gemm1_us > 0 else 0
        gemm2_achieved = gemm2_flops / (gemm2_us / NUM_ITERS * 1e-6) / 1e12 if gemm2_us > 0 else 0

        # Achieved bandwidth (only meaningful if memory-bound)
        gemm1_bw = gemm1_bytes / (gemm1_us / NUM_ITERS * 1e-6) / 1e9 if gemm1_us > 0 else 0
        gemm2_bw = gemm2_bytes / (gemm2_us / NUM_ITERS * 1e-6) / 1e9 if gemm2_us > 0 else 0
        reduce_bw = reduce_bytes / (reduce_us / NUM_ITERS * 1e-6) / 1e9 if reduce_us > 0 else 0

        log(f"\n  Roofline analysis (avg {tokens_per_exp} tok/expert, {active_experts} active experts):")
        log(f"    GEMM1: {gemm1_flops/1e9:.1f} GFLOP, {gemm1_bytes/1e6:.1f} MB, "
            f"AI={gemm1_ai:.0f} FLOP/B → {gemm1_bound}-BOUND (ridge={RIDGE_FP8:.0f})")
        log(f"      Achieved: {gemm1_achieved:.0f} TFLOPS ({100*gemm1_achieved/PEAK_FP8:.1f}% FP8 peak)"
            f", BW: {gemm1_bw:.0f} GB/s ({100*gemm1_bw/PEAK_BW:.1f}% peak)")
        log(f"    GEMM2: {gemm2_flops/1e9:.1f} GFLOP, {gemm2_bytes/1e6:.1f} MB, "
            f"AI={gemm2_ai:.0f} FLOP/B → {gemm2_bound}-BOUND (ridge={RIDGE_TF32:.0f})")
        log(f"      Achieved: {gemm2_achieved:.0f} TFLOPS ({100*gemm2_achieved/PEAK_TF32:.1f}% TF32 peak)"
            f", BW: {gemm2_bw:.0f} GB/s ({100*gemm2_bw/PEAK_BW:.1f}% peak)")
        if reduce_us > 0:
            log(f"    Reduce: {reduce_bytes/1e6:.1f} MB moved, BW: {reduce_bw:.0f} GB/s"
                f" ({100*reduce_bw/PEAK_BW:.1f}% peak)")

        results.append({
            'T': T,
            'wall_ms': total_wall_ms,
            'gemm1_ms': gemm1_us / NUM_ITERS / 1000,
            'gemm2_ms': gemm2_us / NUM_ITERS / 1000,
            'route_ms': routing_us / NUM_ITERS / 1000,
            'sort_hist_ms': sort_hist_us / NUM_ITERS / 1000,
            'sort_layout_ms': sort_layout_us / NUM_ITERS / 1000,
            'sort_init_ms': sort_init_us / NUM_ITERS / 1000,
            'sort_scatter_ms': sort_scatter_us / NUM_ITERS / 1000,
            'sort_fallback_ms': sort_fallback_us / NUM_ITERS / 1000,
            'sort_ms': sorting_us / NUM_ITERS / 1000,
            'reduce_ms': reduce_us / NUM_ITERS / 1000,
            'other_ms': (copy_us + other_us) / NUM_ITERS / 1000,
            'gemm1_tflops': gemm1_achieved,
            'gemm2_tflops': gemm2_achieved,
            'gemm1_bound': gemm1_bound,
            'gemm2_bound': gemm2_bound,
        })

    # ── Summary ──
    log(f"\n{'='*90}")
    log(f"SUMMARY TABLE")
    log(f"{'='*90}")
    hdr = (f"{'T':>6s} | {'Wall':>7s} | {'GEMM1':>7s} | {'GEMM2':>7s} | {'Route':>7s} |"
           f" {'Sort':>7s} | {'Reduce':>7s} | {'Other':>7s} | {'GEMM%':>5s} | {'G1 TFLOP':>8s} | {'G2 TFLOP':>8s} |"
           f" {'G1':>7s} | {'G2':>7s}")
    log(hdr)
    log(f"{'':>6s} | {'(ms)':>7s} | {'(ms)':>7s} | {'(ms)':>7s} | {'(ms)':>7s} |"
        f" {'(ms)':>7s} | {'(ms)':>7s} | {'(ms)':>7s} | {'':>5s} | {'':>8s} | {'':>8s} |"
        f" {'bound':>7s} | {'bound':>7s}")
    log('─' * 130)
    for r in results:
        gemm_pct = 100 * (r['gemm1_ms'] + r['gemm2_ms']) / r['wall_ms'] if r['wall_ms'] > 0 else 0
        log(f"{r['T']:6d} | {r['wall_ms']:7.3f} | {r['gemm1_ms']:7.3f} | {r['gemm2_ms']:7.3f} |"
            f" {r['route_ms']:7.3f} | {r['sort_ms']:7.3f} | {r['reduce_ms']:7.3f} | {r['other_ms']:7.3f} |"
            f" {gemm_pct:4.0f}% | {r['gemm1_tflops']:7.0f}T | {r['gemm2_tflops']:7.0f}T |"
            f" {r['gemm1_bound']:>7s} | {r['gemm2_bound']:>7s}")

    log(f"\n{'='*90}")
    log(f"SORT DETAIL")
    log(f"{'='*90}")
    log(f"{'T':>6s} | {'Hist':>7s} | {'Layout':>7s} | {'Init':>7s} | {'Scatter':>7s} | {'Fallback':>8s}")
    log(f"{'':>6s} | {'(ms)':>7s} | {'(ms)':>7s} | {'(ms)':>7s} | {'(ms)':>7s} | {'(ms)':>8s}")
    log('─' * 70)
    for r in results:
        log(f"{r['T']:6d} | {r['sort_hist_ms']:7.3f} | {r['sort_layout_ms']:7.3f} |"
            f" {r['sort_init_ms']:7.3f} | {r['sort_scatter_ms']:7.3f} | {r['sort_fallback_ms']:8.3f}")

    log(f"\n{'='*90}")
    log(f"OPTIMIZATION INSIGHTS")
    log(f"{'='*90}")

    # Identify dominant bottleneck
    large_t = [r for r in results if r['T'] >= 1024]
    small_t = [r for r in results if r['T'] <= 64]

    if large_t:
        r = large_t[-1]  # T=4096
        g1_frac = r['gemm1_ms'] / r['wall_ms'] * 100 if r['wall_ms'] > 0 else 0
        g2_frac = r['gemm2_ms'] / r['wall_ms'] * 100 if r['wall_ms'] > 0 else 0
        sort_frac = r['sort_ms'] / r['wall_ms'] * 100 if r['wall_ms'] > 0 else 0
        reduce_frac = r['reduce_ms'] / r['wall_ms'] * 100 if r['wall_ms'] > 0 else 0
        overhead = r['route_ms'] + r['other_ms']
        oh_frac = overhead / r['wall_ms'] * 100 if r['wall_ms'] > 0 else 0
        log(f"\n  Large-T (T={r['T']}): GEMM1={g1_frac:.0f}%, GEMM2={g2_frac:.0f}%, sort={sort_frac:.0f}%, reduce={reduce_frac:.0f}%, misc={oh_frac:.0f}%")
        if sort_frac > 10:
            log(f"  → Parallel sort still matters. Sweep tile thresholds and histogram/scatter shape first.")
        elif reduce_frac > 10:
            log(f"  → Token-reduce is material. Autotune BLOCK_N/warps before touching GEMM math.")
        elif g2_frac > g1_frac * 1.3:
            log(f"  → GEMM2 dominates. TF32 throughput is the bottleneck (half of FP8).")
            log(f"    Options: FP8 Intermediate (failed — precision), or improve GEMM2 tile efficiency.")
        elif oh_frac > 20:
            log(f"  → Non-kernel overhead still visible. Check memset/copy and stray runtime kernels.")
        else:
            log(f"  → Balanced. Both GEMMs + overhead contribute.")

    if small_t:
        r = small_t[0]  # T=7
        overhead = r['route_ms'] + r['sort_ms'] + r['reduce_ms'] + r['other_ms']
        oh_frac = overhead / r['wall_ms'] * 100 if r['wall_ms'] > 0 else 0
        gemm_frac = (r['gemm1_ms'] + r['gemm2_ms']) / r['wall_ms'] * 100 if r['wall_ms'] > 0 else 0
        log(f"\n  Small-T (T={r['T']}): GEMM={gemm_frac:.0f}%, overhead={oh_frac:.0f}%")
        if oh_frac > 50:
            log(f"  → Non-GEMM stages dominate. Tiny-batch fast path and lighter sort/reduce are best bets.")
        else:
            log(f"  → GEMM memory-bound (loading 32 experts' weights for few tokens).")
            log(f"    Options: skip inactive experts (already done), or weight prefetching.")

    # ── NCU profiling (if available) ──
    if ncu_path:
        log(f"\n{'='*90}")
        log(f"NCU HARDWARE COUNTERS (T=4096)")
        log(f"{'='*90}")

        ncu_script = f'''
import torch
import importlib.util

spec = importlib.util.spec_from_file_location("km", "{kernel_path}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
kfn = mod.kernel

T, H = 4096, {H}
rl = torch.randn(T, 256, dtype=torch.float32, device='cuda')
rb = torch.randn(256, dtype=torch.bfloat16, device='cuda')
Ap = torch.randn(T, H, dtype=torch.float32, device='cuda')
Aa = Ap.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
hs = (Ap * 448.0 / Aa).to(torch.float8_e4m3fn)
hss = (Aa / 448.0).expand(T, H//128).t().contiguous().float()
g1w = torch.randint(-10,10,(32,4096,H),dtype=torch.int8,device='cuda').to(torch.float8_e4m3fn)
g1s = torch.rand(32,32,56,dtype=torch.float32,device='cuda')+0.5
g2w = torch.randint(-10,10,(32,H,2048),dtype=torch.int8,device='cuda').to(torch.float8_e4m3fn)
g2s = torch.rand(32,56,16,dtype=torch.float32,device='cuda')+0.5
leo = torch.tensor(0, dtype=torch.int32, device='cuda')
rsf = torch.tensor(1.0, dtype=torch.float32, device='cuda')
out = torch.zeros(T, H, dtype=torch.bfloat16, device='cuda')
args = [rl,rb,hs,hss,g1w,g1s,g2w,g2s,leo,rsf,out]

for _ in range(10):
    out.zero_(); kfn(*args)
torch.cuda.synchronize()
out.zero_(); kfn(*args)
torch.cuda.synchronize()
'''
        ncu_script_path = os.path.join(tmp_dir, 'ncu_script.py')
        with open(ncu_script_path, 'w') as f:
            f.write(ncu_script)

        ncu_out = os.path.join(tmp_dir, 'ncu.csv')
        try:
            ncu_cmd = [
                ncu_path,
                '--kernel-name', 'regex:.*fused_moe.*',
                '--metrics',
                'sm__throughput.avg.pct_of_peak_sustained_elapsed,'
                'dram__throughput.avg.pct_of_peak_sustained_elapsed,'
                'sm__warps_active.avg.pct_of_peak_sustained_elapsed,'
                'l1tex__throughput.avg.pct_of_peak_sustained_elapsed',
                '--csv',
                '--log-file', ncu_out,
                '--target-processes', 'all',
                sys.executable, ncu_script_path,
            ]
            r = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=600)
            log(f"  NCU exit code: {r.returncode}")
            if r.stderr and r.returncode != 0:
                # Show first few lines of stderr
                for line in r.stderr.strip().split('\n')[:10]:
                    log(f"  NCU stderr: {line}")
            if os.path.exists(ncu_out):
                with open(ncu_out) as f:
                    csv = f.read()
                log(f"\n  NCU CSV output:")
                for line in csv.strip().split('\n')[:30]:
                    log(f"    {line}")
            elif r.stdout:
                log(f"\n  NCU stdout:")
                for line in r.stdout.strip().split('\n')[:30]:
                    log(f"    {line}")
        except subprocess.TimeoutExpired:
            log("  NCU timed out (600s)")
        except Exception as e:
            log(f"  NCU error: {e}")

    return "\n".join(lines)


@app.local_entrypoint()
def main():
    import subprocess

    print("Packing solution...")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "pack_solution_simple.py")],
        cwd=str(PROJECT_ROOT), check=True,
    )

    solution_path = PROJECT_ROOT / "solution.json"
    solution_json = solution_path.read_text(encoding="utf-8")
    print(f"Solution loaded ({len(solution_json)} bytes)")

    print("\nRunning profiling on Modal B200...")
    result = run_profile.remote(solution_json)

    out_path = PROJECT_ROOT / "ncu_profile_output.txt"
    out_path.write_text(result, encoding="utf-8")
    print(f"\nResults saved to {out_path}")

    try:
        print(result)
    except UnicodeEncodeError:
        print(result.encode('ascii', 'replace').decode('ascii'))
