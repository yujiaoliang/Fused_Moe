# Profiling & Optimization Notes — B200

> **工具:** `scripts/ncu_profile_modal.py` — `torch.profiler` per-kernel 时间分解 + analytical roofline
> **硬件:** NVIDIA B200 (Blackwell, sm100a)
> **峰值性能:** FP8 ~2500 TFLOPS, TF32 ~1250 TFLOPS, HBM3e ~8 TB/s

---

## 1. Profiling 结果（Round 6 baseline）

### 时间分解总表

```
     T |    Wall |   GEMM1 |   GEMM2 |   Route |    Sort |  Copy/0 |   Other | CPU OH | GEMM%
       |    (ms) |    (ms) |    (ms) |    (ms) |    (ms) |    (ms) |    (ms) |        |
─────────────────────────────────────────────────────────────────────────────────────────────
     7 |   0.822 |   0.059 |   0.069 |   0.031 |   0.008 |   0.071 |   0.041 |   66%  |  16%
    14 |   0.906 |   0.061 |   0.036 |   0.030 |   0.008 |   0.066 |   0.040 |   73%  |  11%
    64 |   0.900 |   0.061 |   0.036 |   0.036 |   0.008 |   0.069 |   0.042 |   72%  |  11%
   128 |   0.406 |   0.000 |   0.000 |   0.037 |   0.001 |   0.046 |   0.024 |   73%  |   0%
   512 |   1.030 |   0.068 |   0.135 |   0.048 |   0.019 |   0.076 |   0.051 |   61%  |  20%
  1024 |   0.992 |   0.069 |   0.060 |   0.059 |   0.008 |   0.085 |   0.046 |   67%  |  13%
  4096 |   1.051 |   0.071 |   0.069 |   0.130 |   0.008 |   0.125 |   0.050 |   57%  |  13%
```

### 关键发现

1. **CPU overhead 是主要瓶颈 (56-73% of wall time)**
   - 恒定 ~0.5-0.6ms，不随 T 变化
   - 来源：Python interpreter + PyTorch dispatch (~30 routing ops × ~5-10us each) + tensor allocation

2. **GEMM kernels 本身非常高效 (13-21% of wall time)**
   - GEMM1 (FP8×FP8): 59-71us，已接近 FP8 peak
   - GEMM2 (FP32×FP8): 36-135us，大部分 workload 在 60-70us

3. **Routing 随 T 线性增长** (31us at T=7 → 130us at T=4096)
   - ~30 个 PyTorch kernel launches (sigmoid, topk×3, scatter, gather, masked_fill, etc.)
   - 每个 launch ~5-10us dispatch overhead

4. **Copy/zero 开销显著** (70-176us, 25-28% of CUDA time)
   - `output_fp32.zero_()` + `output.copy_(output_fp32)` fp32→bf16
   - 已通过 buffer cache 部分优化

5. **T=512 GEMM2 异常** (135us vs T=1024 的 60us)
   - 可能是 autotune 选择了次优 tile configuration
   - 值得专门调优

### Roofline 分析

```
Small T (≤128): MEMORY-BOUND — 加载 32 experts 的 weights 是瓶颈
  → Arithmetic Intensity < ridge point (FP8: 312, TF32: 156 FLOP/B)
  → 优化方向：weight prefetching, skip inactive experts

Large T (≥512): COMPUTE-BOUND — 充分利用 tensor cores
  → AI >> ridge point
  → 优化方向：更高效的 tile sizes, persistent kernel
```

---

## 2. CPU Overhead 分解（估算）

```
~600us total CPU overhead:
  ├── Routing Python + dispatch:  ~300us  (50%)
  │   ├── torch.topk ×3:          ~60us  (kernel launch + dispatch)
  │   ├── sigmoid, add, gather:    ~40us  (element-wise ops)
  │   ├── scatter_, masked_fill_:  ~30us
  │   └── Python interpreter:     ~170us  (control flow, dict lookup, etc.)
  ├── Sorting Python + dispatch:  ~100us  (17%)
  │   ├── argsort:                 ~20us
  │   ├── .cpu().tolist() sync:    ~30us  (GPU→CPU transfer)
  │   └── Python loop + cat:       ~50us
  ├── GEMM dispatch (2 launches):  ~30us  (5%)
  └── Framework overhead:         ~170us  (28%)
      ├── Triton autotune lookup:   ~20us
      ├── Tensor alloc/dealloc:     ~50us
      └── Other PyTorch runtime:   ~100us
```

---

## 3. Round 7 优化尝试

### 3.1 CUDA Graph for GEMMs — ❌ 无提升

**方案:** 捕获 GEMM1+GEMM2+zero+copy 为 CUDA Graph，pre-allocate persistent buffers (output_fp32, Intermediate, sorted arrays)。Warmup 3 calls for autotune，第 4 次 capture，后续 replay。

**结果:** 19/19 PASSED，avg ~9.9x（与 baseline ~10.1x 无显著差异）

**原因分析:**
- GEMM 只有 2 次 Triton launch，launch overhead ~20-50us
- 占 CPU overhead 的 <8%，graph 最多节省 ~50us
- Extra `.copy_()` for sorted data + graph state management 增加 ~25us
- 净节省 <30us out of ~900us wall time = <3%

**教训:** CUDA Graph 适用于**大量小 kernel launch** 的场景（如 routing 的 30+ ops）。对只有 2 个大 kernel 的 GEMM 部分，launch overhead 不是瓶颈。

### 3.2 torch.compile on Routing — ✅ 微小提升（保留）

**方案:** `torch.compile(_ds_routing_impl, mode="reduce-overhead", dynamic=True)` 融合 routing 中的 element-wise ops。

**测试版本:**

| Mode | Avg | Peak | Large-T |
|------|-----|------|---------|
| No compile (Round 6) | ~10.1x | ~12.1x | ~7.0-7.4x |
| `dynamic=True` (default) | ~10.0x | 12.40x | 6.82-7.26x |
| `reduce-overhead, dynamic=True` | **~10.2x** | **12.56x** | 6.73-7.11x |

**分析:**
- torch.compile 能融合 sigmoid+add, sum+div+mul 等 element-wise ops
- 但 `topk` 是不可融合的 parallel reduction，仍然生成独立 kernel
- `reduce-overhead` 模式内部使用 CUDA Graph 缓存 compiled subgraphs
- 净提升 ~1-2%，在 benchmark 噪声范围内，但无 downside → 保留

### 3.3 Pre-allocated Buffer Cache — ✅ 保留

**方案:** `output_fp32` 使用 module-level dict 缓存，同 T 跨 call 复用（`.zero_()` 替代 `torch.zeros()`）。

**分析:** PyTorch CUDA caching allocator 已经做了类似的事，实测提升可忽略。但代码更干净（明确 buffer 生命周期），保留。

---

## 4. Round 6 阶段遗留的优化方向（现已在 Round 8/9 全部解决）

### ✅ P0: Fuse Routing into Triton Kernel (Round 8 完成)

**预期与结果:** 预期节省 ~300us (routing Python + dispatch overhead)。实测通过 `triton_ds_routing_kernel` 完全消除，将峰值从 12.56x 推升至 16.73x。

**Triton topk 最终实现:**
- 放弃了原本复杂的 Bitonic Sort。
- 直接利用 `tl.reshape` 和多次 `tl.argmax` 并叠加 `tl.where(mask, -inf)` 来暴力剔除极值，硬编码实现了 Group-Level Top-2 和 Global Top-8 的无排序提取，速度极快。

### ✅ P1: Fuse Sorting into Triton Kernel (Round 9 完成)

**预期与结果:** 预期节省 ~100us (sorting Python + CPU sync)。实测通过 `triton_sort_and_scatter_kernel` 完全消除，将峰值从 16.73x 推升至 47.89x。

**实现机制:** 
摒弃 PyTorch 的 `argsort`，直接在 Triton 内遍历 `T * TOP_K` 个 Token，依靠 `tl.atomic_add` 针对 32 个 Local Expert 累加 Histogram。并巧妙使用了 Empty-Block-Skipping 让后续 GEMM 主动查表，消灭了 Python 端的 Launch 循环。

### ✅ P2: GEMM2 Tile Tuning for T=512 (Round 9 完成)

**预期与结果:** 针对 T=512 (num_padded ~32K) 开展的穷举 Autotuning。引入了极深的 `num_stages=6` 与横跨整个 L2 Cache 的 `GROUP_M=32`。最终证实当前架构已被 Memory Bandwidth 锁住上限，稳定在 ~22.5x。

---

## 5. B200 硬件参考

| 指标 | 数值 |
|------|------|
| FP8 Tensor Core (dense) | ~2500 TFLOPS |
| TF32 Tensor Core (dense) | ~1250 TFLOPS |
| HBM3e Bandwidth | ~8 TB/s |
| L2 Cache | 96 MB |
| Architecture | sm_100a (Blackwell) |

---

## 6. Round 8 & Round 9 终极优化 (Pure Triton)

**实现:**
1. **Round 8 (Routing):** 完全使用 Pure Triton 实现了 `triton_ds_routing_kernel`。用寄存器洗牌和 `tl.argmax` 从零构建了 DeepSeek-V3 的 Sigmoid -> Group Top-2 -> Global Top-8 工作流。
2. **Round 9 (Sorting):** 编写了 `triton_sort_and_scatter_kernel`，使用 Global Memory `tl.atomic_add` 实现了并行的 Token Sorting 和 Expert Offset 统计。结合 GEMM Kernels 中引入的 **Empty-Block-Skipping** 指针屏蔽机制，彻底铲除了所有 Python-Side 同步。

**优化结果 (B200):**
- **最高 Peak 暴涨:** T=7 极短序列上的 `e05c6c03` 从 16.73x (Round 8) 直接跃升至惊人的 **47.89x** (Round 9)。
- **CPU Time 彻底归零:** 将原本占耗时达 60-70% (~600us) 的 PyTorch Dims Dispatch + Allocation 完全消除，CUDA Time 等于 Wall Time。
- **内存拷贝完全移除:** 通过精心设计的 Buffer Cache，把 `output_fp32` 预分配与 GEMM 的 Atomic 积累融合，去除了冗余的 `fp32 -> bf16` casting 开销。

---

## 7. GEMM Autotuning 耗尽测试 (B200 Blackwell)

针对长期存在的 T=512 (例如 `1a4c6ba1` 和 `5e8dc11c`) 及 T=4096 大长度序列的问题，扩展了 Triton 的 Autotune 探索空间：
- 引入了 `GROUP_M=1, 16, 32`，尝试对齐或穿透整个 96MB L2 Cache。
- 将 `num_stages` 枚举提升至 `4, 5, 6` 榨干张量核心并发。

**结论 (Hardware Saturation):**
- 添加极深流水线 (`num_stages=6`) 或是对齐整个 L2 Cache 的网格映射 (`GROUP_M=32`) 未能带来更显著的性能提升。
- 最终 T=512 成绩稳固在约 **+22.5x**，而 T=4096 稳固在 **+6.5x~7.1x**。
- **分析:** 这证明了当前的 `_fused_moe_gemm1_swiglu_kernel` (N=4096) 和 `_fused_moe_gemm2_scatter_kernel` (N=2048) 已经在指令并发和 HBM 带宽上挤干了 B200 的硬件能力。考虑到 Fused FP8 GEMMs + Scatter 添加过程本身的极限 Memory Bound 物理属性，目前的结果即为基于此工程配置的数学极限。

---

## 8. Round 10: Non-Atomic GEMM2 + Reduce-Scatter (B200)

**核心发现:** 在 Round 7 的 autotune 饱和测试中，我们曾错误地认为 GEMM2 已经达到硬件极限。实际上，**瓶颈不是 GEMM 计算本身，而是 `tl.atomic_add` 的写入竞争**。

**方案:** 将 `_fused_moe_gemm2_scatter_kernel`（GEMM + atomic scatter-add）拆分为两个独立 kernel：
1. `_fused_moe_gemm2_kernel`：纯 GEMM 计算，通过非原子 `tl.store` 写入 `expert_out[num_padded, 7168]`
2. `_reduce_scatter_kernel`：从 `expert_out` 读取并通过 `tl.atomic_add` 累加到 `output_fp32[T, 7168]`

**为什么有效:**
- 原来 GEMM2 的 K-loop（16 iterations × BLOCK_K=128）在计算完成后立即 atomic_add，大量 SM 同时竞争同一 output 行
- 拆分后，GEMM2 变为纯计算 kernel（无竞争），吞吐量直接拉满
- Reduce-scatter 仅做 load+add（无 GEMM），原子操作独占带宽，竞争程度大幅降低

**结果:**
- **Peak:** 47.89x → **54.88x** (+15%)
- **中等 T (128-512):** 22x → **32-44x** (+45-57%) 🔥🔥
- **Large-T (4096):** 6.6x → **7.3-8.2x** (+11-15%)

**代价:** 额外 `expert_out[num_padded, 7168]` fp32 缓冲区。T=4096 时约 ~900MB，B200 有 180GB，完全可接受。
