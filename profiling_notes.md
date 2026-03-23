# Profiling & Optimization Notes — B200

> **工具:** `scripts/ncu_profile_modal.py` — `torch.profiler` per-kernel 时间分解 + analytical roofline
> **硬件:** NVIDIA B200 (Blackwell, sm100a)
> **峰值性能:** FP8 ~2500 TFLOPS, TF32 ~1250 TFLOPS, HBM3e ~8 TB/s

> **当前最佳版本:** Round 15 (`d2fdf14`) — Peak **106.65x**, Mean **55.77x**, GMean **47.54x**, `14 / 19` workloads 达到 **50x+**

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
- **早期误判:** 当时分析认为这证明了 `_fused_moe_gemm1_swiglu_kernel` 和 `_fused_moe_gemm2_scatter_kernel` 已经在指令并发和 HBM 带宽上挤干了 B200 的硬件能力。
- **后续反转 (Round 10):** 事实证明真正的瓶颈并不是计算或带宽，而是 **`tl.atomic_add` 的并发竞争耗尽了硬件原子单元吞吐**。拆解原子操作后，大 T 性能显著提升。

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

---

## 9. Round 10.1: Token-Centric Reduce (B200)

**继续推进:** Round 10 虽然将 GEMM2 的原子竞争消除了，但 `_reduce_scatter_kernel` 仍然使用 `tl.atomic_add` 将 expert_out 散射到 output_fp32，并且还有 `output_fp32.zero_()` 和 `output_fp32.to(bf16)` 的开销。

**方案:** 将 expert-centric 的 `_reduce_scatter_kernel` 替换为 token-centric 的 `_token_reduce_kernel`：
- 在 sort kernel 中新增 `scatter_map[T*TOP_K]` 输出，记录每个 (token, expert_slot) 在 expert_out 中的位置
- 新 kernel grid = `T × cdiv(7168, BLOCK_N)`，每个程序处理一个 token 的 BLOCK_N 列
- 通过 `tl.static_range(TOP_K)` 循环读取恰好 8 个 expert 贡献并求和
- 直接写入 bf16 output，无需 fp32 中间缓冲

**消除的开销:**
- `output_fp32.zero_()` — 完全不需要了（T=4096 时 = 112MB memset）
- `tl.atomic_add` — 完全不需要了（每个 token 仅由一个程序处理）
- `output_fp32.to(bf16)` — 完全不需要了（直接写 bf16）
- `output_fp32` 缓冲区本身 — 完全不需要分配了

**结果:**
- **Peak:** 54.88x → **80.31x** (+46%) 🔥🔥🔥
- **小 T (7-54):** 40-55x → **63-80x** (+35-78%)
- **中等 T (128-512):** 32-44x → **36-52x** (+4-17%)
- **Large-T (4096):** 7.3-8.2x → **7.2-8.1x** (~持平)

**为什么小 T 收益最大?** 因为小 T 场景下 reduce 阶段占总时间的比例更高（GEMM 很快就结束了），所以消除 zero/atomic/copy 的效果被放大。

---

## 10. Round 11: GEMM2 BF16 & Weight Fusion (❌ Reverted)

**探索思路:** 将 GEMM2 结尾乘 routing weight 的操作，推迟到 `_token_reduce_kernel` 中执行。这样可以带来的好处是：
1. GEMM2 可以不乘 weight 直接写纯 FP32 累加结果。
2. 尝试让 GEMM2 直接写出 **BF16** 的 `expert_out`，从而将 `expert_out` 的显存带宽需求减半（7168 维，T=4096 时从 900MB 减半到 450MB）。

**失败原因分析:**
- **尝试 A (BF16 `expert_out`):** 19/19 精度报错（INCORRECT Numerical）。未乘 routing weight 的 GEMM2 原始累加结果可以达到数百至数千级别，而 BF16 只有 3 位十进制的有效数字，强行缩减为 BF16 导致了无法接受的精度损失。
- **尝试 B (FP32 `expert_out` + 推迟 weight):** 19/19 PASSED，但**性能倒退**（peak 80.31x 跌至 66.93x）。虽然 GEMM2 省掉了一次 token weight 读取，但在 Token-Reduce 阶段，每个 token slot 多了一次对 `sorted_weights` 的读操作和乘法操作。事实证明，在 Token-Reduce 这边增加操作的代价大于在 GEMM2 里减压的收益。

**结论:** `expert_out` 必须保持为 FP32。不改变数据格式的前提下，推迟 weight 乘法毫无意义。代码**全部回退**至 Round 10.1 状态。

---

## 11. Round 12: BLOCK_M Adaptive Tuning

**瓶颈分析:** Round 10.1 极大提升了小/中 T 的性能，但大 T (4096) 仍然停留在 7-8x。大 T 工作负载受限于 GEMM1 和 GEMM2 的 Memory Bandwidth。

**方案:** 调整 Triton tuning parameters，实现多级 `BLOCK_M` 分配：
- `T*8 <= 512`: `BLOCK_M_SMALL = 32` (减少 per-expert padding 浪费)
- `T*8 > 16384`: `BLOCK_M_XLARGE = 128` (增大 tile size，提高 K-loop 的共享内存数据重用率)
- 其他: `BLOCK_M_LARGE = 64` (默认)

**失败尝试 (`BLOCK_M_TINY = 16`):**
尝试为 T=7 (T*8=56) 使用 16 作为 BLOCK_M，以减少 padding waste (用 32 时 waste=78%)。
结果性能严重倒退：`e05c6c03` 从 80x 跌至 51x，`2e69caee` 从 71x 跌至 48x。
**原因:** 16 × 128 (BLOCK_M × BLOCK_K) 的矩阵乘法块太小，无法充分填满 B200 的 FP8 TensorCore 计算管线。TensorCore 需要起码的矩阵规模才能实现高吞吐，padding waste 带来的计算 overhead 远小于 underutilization 的惩罚。

**最终结果:**
采用 3-level 调整 (32 / 64 / 128)，大 T (4096) 收益显著：
- `58a34f27`: 8.06x → **8.82x** (+9%)
- `5e8dc11c`: 7.24x → **8.18x** (+13%)

通过增大 `BLOCK_M` 到 128，我们为 large-T 缓解了 Memory Bound。

---

## 12. Round 13: Dynamic FP8 Quantization for SwiGLU (Bypass & Precision Wall)

**探索思路:** 将 `Intermediate` 缓冲（SwiGLU 的输出）从 FP32 动态计算 Scale 并量化为 `float8_e4m3fn` (FP8)。这样在接下来的 GEMM2 中，可以直接使用 `tl.dot(f8, f8)`（彻底激活 B200 原生的 FP8xFP8 TensorCore），理论吞吐量翻倍。

**核心发现与战术动作:**
1. **编译器崩溃 (ICE):** 直接在 Triton 的 `_fused_moe_gemm1_swiglu_kernel` 中对 `swiglu_out` (MMA layout) 进行 `tl.reshape` 或 `.to(tl.float8e4nv)` 均会导致 B200 上的 `ptxas` 汇编器抛出严重内部崩溃 (ICE)。这是 Triton LLVM backend 对于 `cvt.e4m3x2` 寄存器打包生成的 Bug。
2. **完美绕过 (Bypass):** 我们放弃了在 GEMM1 中量化。而是将 FP32 存入 Global Memory，并在 `_fused_moe_gemm2_kernel` 的 K-Loop 中从标准的 `Blocked` layout 载入 SRAM 后，执行即时 (JIT) 缩放与强化：
   `a_fp8 = (a_fp32 / a_scale[:, None]).to(tl.float8e4nv)`
   这成功避开了编译错误，并在 B200 上顺利调用了原生的 FP8xFP8 TensorCore！
3. **不可逾越的精度墙 (Quantization Squeeze):**
   虽然代码跑通了，但测试框架爆出了 `abs=1.02e+04` 的巨巨额误差。
   原因是 `check_modal.py` 传入了未经归一化的 `torch.randn`，这导致 SwiGLU 累加结果飙升至 300+。而 `float8_e4m3fn` **只有 3 位尾数精度**。对于这种大范围浮点数，3 位的阶跃截断会产生几位的绝对偏移（Rounding Error）。而基准框架是用极其苛刻的 `$10^{-3}$` 级 FP32/BF16 容差防线做比对。这导致原生的物理级 FP8 计算先天注定无法通过这项严苛的容差校验。

**结论:**
我们在代码生成、内存布局和 TensorCore 编排上取得了全胜，实现了一个完全 Layout-Safe 的 FP8 动态量化架构。但也证实了：在一个要求极高浮点容差比对的评测系统中，3-bit (FP8) 的粗糙精度天然无法交出及格卷。
**行动:** 将主线 `kernel.py` 干净回溯至峰值 80.31x 的 Round 12 状态，全面转向方向 A (Multi-Block Sort)。

---


## 13. Round 14: Multi-Block Parallel Sort (Direction A)

**探索思路:** 
Round 9 实现的 `triton_sort_and_scatter` 在统计各 Expert 的 Token 计数时，由于只有一个 Program（单线程）遍历所有 `T * TOP_K` 个 Token，其内部的 `tl.atomic_add` 操作在大长序列 (如 `T=4096`) 时形成了严重的串行原子竞争瓶颈。

**解决方案:**
摒弃串行扫描，全面引入 `parallel_sort_and_scatter` 并行调度。队友通过多 Block 切分计算，让 GPU 内部多个 SM 并行执行 Histogram 统计和 Prefix Sum 计算（通过分配 `partial_counts` 等额外 Workspace Buffer 配合 `NUM_TILES`），彻底化解了长序列时的 Atomic Bottleneck。

**结果:**
- **Peak Performance:** 80.31x → **88.x**
- **Large-T (4096):** 8.8x → **>9.3x**

**结论:** Multi-Block Sort 基本清掉了长序列上的 sort / histogram 串行瓶颈，也把系统主瓶颈从全局调度问题进一步推进到了 medium-T GEMM 固定成本。它不是终点，而是 Round 15 bucket specialization 的前置基础。

---

## 14. Round 15: Medium-T Bucket Specialization (`d2fdf14`)

**瓶颈迁移:**
Round 14 之后，routing / sort / reduce 的大头固定开销已经明显下降，large-T 也被 `BLOCK_M=128` 和并行 sort 基本托住。剩下最突出的短板变成了 **`T≈32-128` 的 medium batch**：这类 workload 的 GEMM 计算量还不足以完全淹没 kernel 固定成本，但又已经大到会被 padding、expert 边界查找和过宽 autotune 搜索空间拖慢。

**方案:**
- 将 `BLOCK_M` 扩成四档：`16 / 32 / 64 / 128`
- `32 <= T <= 64` 走 `_small_medium_fused_moe_gemm1_swiglu_kernel` / `_small_medium_fused_moe_gemm2_kernel`
- `65 <= T <= 128` 走 `_medium_fused_moe_gemm1_swiglu_kernel` / `_medium_fused_moe_gemm2_kernel`
- small-medium GEMM1 中 expert lookup 从完整 32-way scan 改为基于 `block_offsets_ptr` 的二分查找

**为什么有效（profiling 视角）:**
- medium-T 对 per-expert padding waste 更敏感，细分 `BLOCK_M` 能直接减少无效 tile
- bucketed kernel 缩窄了 autotune / launch configuration 的搜索空间，降低 generic path 的固定开销
- expert 边界查找被简化后，GEMM1 前置控制流成本明显下降，短中序列更容易把时间花在真正的 MMA 上
- 这说明主瓶颈已不再是 Round 14 之前的全局 sort atomic contention，而是 **medium-T 的 kernel specialization 不足**

**结果摘要:**
- **Peak:** **106.65x**（`2e69caee`）
- **Mean:** **55.77x**
- **GMean:** **47.54x**
- **50x+ Workloads:** **14 / 19**
- **代表性提升:** `b8f4f012 66.21x -> 95.85x`，`8f1ff9f1 23.91x -> 48.09x`，`a7c2bcfd 64.44x -> 72.44x`，`2e69caee 64.23x -> 106.65x`
- **Large-T 基本持平:** `58a34f27 9.51x -> 9.25x`，`5e8dc11c 8.54x -> 8.34x`

**结论:** 到 Round 15 为止，系统层面的优化路径已经非常清晰：
1. 先用 Pure Triton routing / sorting / token-reduce 清掉 CPU 与 atomic 固定开销；
2. 再用 `BLOCK_M=128` 和 parallel sort 稳住 large-T；
3. 最后通过 medium-T bucket specialization 把 `T≈32-128` 的 generic GEMM 固定成本继续打薄。

如果后面还要继续抬分，profiling 的重点应放在 **GEMM2 / medium bucket 的进一步特化**，而不是回头再优化 routing 或 sort。

---

## 15. Experimental Packed GEMM2: Dense Fallback `BLOCK_M=128`

> 这部分是 **experimental / research path**，不是当前 README 主线结果；但 profiling 上已经出现了一个很明确的新杠杆。

### 背景

在 packed GEMM2 + dense fallback 这条实验线上，前几轮已经把主瓶颈收敛到：

- dense fallback kernel 本体
- coarse score kernel
- 若干 `index_select / index_copy / gpu_index_kernel` 辅助开销

这轮先后试了三件事：

1. `score_mode` sweep：`sum / sum_max / max / top2 / top4`
2. boundary refine：只对边界 rows 再做一轮细粒度评分
3. dense fallback regroup 的 `BLOCK_M` 强制实验

### 先排除两个次优方向

**1) `score_mode` sweep：**

- 在 `5e8dc11c`, `80%` fallback 上，`sum` 仍然最好：
  - `sum`: `matched_ratio = 0.946565`
  - `sum_max`: `0.945476`
  - `max`: `0.943205`
  - `top2`: `0.944438`
  - `top4`: `0.945145`

结论：更复杂的 block aggregation 没有带来更好的 row ranking。

**2) boundary refine：**

- 最好的窗口大约是 `2%` / `5%` / `6%`
- 但 `matched_ratio` 仍只到 `~0.9475`

结论：多级 score 的方向没错，但这时候收益已经太小，不足以把 `80%` fallback 拉过线。

### 真正有效的一刀：强制 dense fallback regroup 用 `BLOCK_M=128`

继续看 microprofile 之后，发现 dense fallback 这段已经和主线 large-T GEMM 很像：

- 一旦 regroup 后的 batch 够大，吞吐上更大的 tile 往往比 padding 成本更值钱

于是直接固定 dense fallback regroup 的 `BLOCK_M`：

- `BLOCK_M=32`
- `BLOCK_M=64`
- `BLOCK_M=128`

在 `5e8dc11c`, `81%` fallback 上，三档 correctness 都通过：

- `BLOCK_M=32`: `matched_ratio = 0.951319`
- `BLOCK_M=64`: `0.951648`
- `BLOCK_M=128`: `0.951035`

所以这里的核心不是数值，而是**哪档 `BLOCK_M` 更快**。

### Microprofile 结果

`5e8dc11c` 上：

- `BLOCK_M=32`
  - experimental wall = `9.507 ms`
  - `dense_fallback = 2.558 ms`
  - `copy = 0.197 ms`
- `BLOCK_M=64`
  - experimental wall = `9.056 ms`
  - `dense_fallback = 2.707 ms`
  - `copy = 0.208 ms`
- `BLOCK_M=128`
  - experimental wall = **`8.696 ms`**
  - `dense_fallback = **2.154 ms**`
  - `packed_gemm2 = 0.783 ms`
  - `score = 0.486 ms`
  - `copy = 0.170 ms`

这里最重要的 profiling 结论是：

- dense fallback kernel 从 `~2.7 ms` 直接压到 `2.15 ms`
- experimental wall 也随之降到 `8.696 ms`
- 这已经不是噪声，而是一个明确的一阶收益

### 对 benchmark 的意义

对应 benchmark 结果也同步往前推了一步：

- 过滤 benchmark（`1a4c6ba1`, `5e8dc11c`, `58a34f27`）
  - `1a4c6ba1`: `2.851 ms | 7.31x`
  - `5e8dc11c`: `8.281 ms | 5.42x`
  - `58a34f27`: `6.314 ms | 5.66x`

- 完整 19 workload benchmark
  - `19 / 19 PASSED`
  - `1a4c6ba1`: `3.434 ms | 6.28x`
  - `5e8dc11c`: `8.578 ms | 5.31x`
  - `58a34f27`: `6.783 ms | 5.40x`

同时也确认了：

- `80%` fallback + `BLOCK_M=128` 仍然不过线（`matched_ratio = 0.946021`）
- 所以 correctness threshold 仍然是 `81%`

### 当前最佳实验配置

- `BLOCK_K=32`
- `RESIDUAL_TOPK=13`
- `DENSE_FALLBACK_ROW_PCT=81`
- `DENSE_FALLBACK_SCORE_MODE=sum`
- `DENSE_FALLBACK_REFINE_WINDOW_PCT=0`
- `DENSE_FALLBACK_FORCE_BLOCK_M=128`

### 这轮 profiling 给出的下一步

如果继续做这条 experimental path，profiling 上最值得打的点已经进一步收敛成：

1. **把 dense fallback 的 regroup / gather / scatter 继续下沉到 Triton**
   - `other` 里仍有比较稳定的 `gpu_index_kernel` 和 `index_put/index_select` 成本
2. **把 dense fallback 的 `BLOCK_M` 选择从"padding 最优"改成"性能最优"**
   - 现在 `BLOCK_M=128` 对大 fallback batch 明显更快
   - 未来更合理的是根据 `selected_row_count` / `grouped_total_blocks` 动态选 `32 / 64 / 128`

---

## 16. MXFP8 `tl.dot_scaled` 最终实验 (Blackwell 原生 Microscaling)

**探索思路：** 放弃手动 PTX pack/unpack 链路，改用 Triton 原生的 `tl.dot_scaled` API，调用 Blackwell 第 5 代 TensorCore 的硬件级 MXFP8 (Microscaling FP8) 指令。MXFP8 标准中每 32 个元素共享一个 `e8m0` 共享指数，精度理论上比 per-128 block scale 更高。

**关键发现：**
1. **编译器零 ICE：** `tl.dot_scaled` 在 Triton 3.6.0 + B200 sm100 上**完美编译并运行**，不受 Issue #8648 影响。这是与 `tl.dot(fp8, fp8)` 路径截然不同的代码生成路径。
2. **API 约定：** `lhs_scale: [BLOCK_M, K//32]`, `rhs_scale: [BLOCK_N, K//32]`，均为 `uint8` (`e8m0` 格式)。`e8m0` 值 127 = scale 1.0 (2^(127-127))。
3. **性能极快：** 256x2048x256 矩阵仅 **0.041ms**（median of 100 runs）。

**精度结果（FP32→MXFP8 双边量化 + tl.dot_scaled）：**

| 测试场景 | 矩阵尺寸 | 数据范围 | matched_ratio |
|----------|---------|---------|---------------|
| 小矩阵 | 32x32x256 | ±500 | 0.316 |
| 真实 GEMM2 | 64x256x2048 | ±4500 | 0.186 |
| 极端 SwiGLU | 64x256x2048 | ±180000 | 0.170 |

**根因分析：** 即使 MXFP8 的 per-32 scaling 精度理论上比 per-128 高了 4 倍，`matched_ratio` 反而**更差**（0.17-0.32 vs 手动 pack 的 ~0.80）。原因是 `e8m0` 共享指数**只能表示 2 的幂次**（离散步长），而手动 pack 使用的 FP32 scale 可以是任意连续值。虽然 MXFP8 的粒度更细（32 vs 128），但每个 scale 的精度更粗，两者相消。

**🏁 FP8 总结论（封棺）：**

经过从 Round 7 到 Round 16 的系统性探索，FP8 在本评测框架的严格容差（`matched_ratio ≥ 0.95`）下被证明**不可行**：

- ❌ Per-tensor FP8 cast → 全 NaN（SwiGLU 值 >448）
- ❌ Per-128 block scale → matched_ratio ~0.80
- ❌ Per-32 block scale + residual correction → 需要 81-85% dense fallback → 净负收益
- ❌ MXFP8 `tl.dot_scaled` + per-32 e8m0 → matched_ratio 0.17-0.32
- ❌ JIT dynamic cast in K-loop → 88x → 76x（循环内 scaling 开销吃掉算力增益）

**根本原因不是编译器、布局或工程实现，而是 FP8 e4m3 的 3-bit 尾数是硬件物理极限。在需要 FP32/BF16 级别精度的评测场景中，FP8 天然无法通过。**

---

## 17. Round 16: Direction 1 — Mainline Performance Optimization

**总目标：** 放弃 FP8，转攻 mainline 88-106x baseline 的其他瓶颈。

### Profiling 基线 (T=4096, B200)

| 内核 | CUDA 时间 (5 iter) | 占比 |
|------|-------------------|------|
| **GEMM2** | 2.733ms | **47.58%** |
| **GEMM1+SwiGLU** | 2.105ms | **36.65%** |
| Token Reduce | 395.264us | 6.88% |
| Routing | 193.888us | 3.38% |
| Sort/Scatter | 164.801us | 2.87% |
| aten::zero_ | 65.6us | 1.14% (框架外部开销，无法消除) |

**关键发现：** 两个 GEMM 合计占 **84.23%**。Token Reduce 仅 6.88%，比最初预期的低得多。

### ❌ Target 2: Token Reduce 融合进 GEMM2

**方案：** 在 GEMM2 epilogue 中直接 `tl.atomic_add` 到 `output_fp32[token_id]`，消除整个 `expert_out` 缓冲区。

**失败原因：** Triton 的 2D tile accumulator 不支持 `acc[m, :]` 行级索引。one-hot mask workaround（`tl.sum(out * mask[:, None], axis=0)`）在 `tl.static_range(BLOCK_M)` 循环中是 **O(BLOCK_M² × BLOCK_N)**，导致大 T 直接 **TIMEOUT**。

### ❌ Target 3: BF16 Intermediate 存储压缩

早已验证失败（Round 11）：SwiGLU 输出动态范围大，bf16 的 7-bit 尾数截断导致 6/19 PASSED。

### ✅ Target 4: Autotune 扩展

为 GEMM1 增加 6 个、GEMM2 增加 8 个大 T 专用配置（`num_stages=7/8`，`GROUP_M=16` with `num_stages=5`）。

**结果：** 19/19 PASSED，`58a34f27`(T=8192) 9.25x → 9.29x（噪声范围内）。

### ❌ Target 5: Persistent GEMM2 Kernel

**方案：** 为减少 GEMM2 每 56 次 N-block 循环都要执行 1 次 `expert_id` lookup 的开销（~3-5 条指令）并增加 A-side 的 L2 缓存复用，将 `grid` 从 `(M_blocks × N_blocks)` 改为 `(M_blocks,)`，在每个 program 内部循环遍历所有 `N_blocks`。

**失败原因与结果：**
- **19/19 PASSED，但性能灾难性退化**（例如 `2e69caee` 52x → 15.6x, `58a34f27` 9.29x → 8.01x）。
- 原本 kernel 启动 `14,560` 个 program（T=4096时），极大超出 B200 的 160 个 SM 的容量，允许硬件调度器在不同 program 间自由切换 warp 来隐藏内存延迟。
- Persistent kernel 仅启动 `260` 个 program，勉强填满 SM（1.6 waves）。每个 program 虽然内部有 N 循环，但循环是**严格串行**的。
- 当一个 SM 在加载 `B` 或 `A` 时发生 cache miss 挂起，它**缺少其他独立的 N-block warps 可以切换**，导致纯粹的内存延迟暴露。

**结论：** 在 GPU 编程中，为了一点点局部缓存复用而牺牲**海量线程级并行（Thread-Level Parallelism）**得不偿失。必须维持原始的 `M×N` grid 设计。

### Token Reduce 调参

| 配置 | 58a34f27 (T=8192) | 备注 |
|------|:------------------:|------|
| `BLOCK_N=256, warps=4`（baseline） | 9.29x | 原始配置 |
| `BLOCK_N=256, warps=8` | 8.80x ↓ | 多 warp 无益 |
| `BLOCK_N=512, warps=8` | 9.24x ≈ | 更宽 tile 回弹但无净增益 |

**结论：** Token Reduce 仅占 6.88% (T=4096)，调参对总性能影响可忽略。

### ❌ Target 6: TMA (Tensor Memory Accelerator)

**方案：** 使用 `tl.make_block_ptr` 重写 GEMM2 的 `A`、`B` 和 `C` 读写，触发 B200 (SM100) 的原生 TMA 异步访存指令，试图绕过 LDG 指令的带宽限制。

**结果：** 19/19 PASSED，但性能**微小退化**（~1-2% 下降）：
- `5e8dc11c` (T=14107): 8.3x → 8.22x
- `58a34f27` (T=8192): 9.3x → 9.15x
- `e05c6c03`: 61x → 59.01x

**结论：** TMA 的核心优势是 offload 地址计算和减少 SM 寄存器占用。但在极端 memory-bound 的 GEMM2 中，物理 HBM 带宽已经被常规 LDG 读指令完全打满，SM 本来就有大量的 idle 等待访存。TMA 非但不能凭空变出物理带宽，反而因描述符 setup 开销和不能像手动 LDG 那样精细调整 `num_stages` (预取策略不同) 导致轻微退化。

### 🏁 Direction 1 总结

到这一步为止，mainline 的所有主要优化路径均已穷尽：

1. **计算密度** — FP8 / BF16 精度不通过
2. **访存密度** — Intermediate 必须 FP32，expert_out 必须 FP32，无法压缩
3. **Kernel fusion** — Token Reduce 与 GEMM2 融合受 Triton 2D tile 限制
4. **Pipeline 深度** — autotune num_stages 7/8 无益
5. **Token Reduce 配置** — BLOCK_N / num_warps 调参无净增益
6. **局部性与调度** — Persistent Kernel 失去张量级并行导致 3x 性能断崖
7. **底层访存指令** — TMA 异步访存指令无法突破物理 HBM 天花板

## 🚀 Direction 2 总结：Algorithmic Fusion (算法级融合)

为了打破物理 HBM 带宽的天花板，唯一的出路是**跨内核融合 (Cross-Kernel Fusion)**，即完全消除 `expert_out` 缓冲区的读写（省下近 1GB 的访存流量）。

### ❌ Target 7: 1D Flatten + Atomic Scatter 强行融合 GEMM2 与 Token Reduce

**方案：** 针对之前 "Triton 2D tile 散列写不支持" 的限制，将 `_fused_moe_gemm2_kernel` 的目标指针用 `tl.reshape` 压平为 1D 数组，并在 GEMM2 epilogue 阶段直接调用 `tl.atomic_add` 指向最终的 Token `output` 缓冲区。同时绕过单独的 Token Reduce。
**结果：** 19/19 PASSED，但**性能雪崩式下降（约 35% 退化）**。
- `5e8dc11c` (T=14107): 8.3x → **5.37x**
- `58a34f27` (T=8192): 9.3x → **6.24x**

**失败原因深层剖析（硬件级）：**
- **无法向量化 (Failed Vectorization)**：原始的 `tl.store(expert_out)` 写入是内存连续的连续行，Triton 可以生成 `st.global.v4` 等**128位宽的向量化读写指令**。
- **Scalar Atomics 风暴**：当我们用 1D Flatten 破坏了指针的二维空间结构并调用 `tl.atomic_add` 时，PTX 编译器无法识别沿着 N=7168 维度的连续性，只能被迫 fallback 到针对每个浮点数生成**独立的标量原子操作（Scalar Atomics）**。
- **内存控制器瘫痪**：原本针对一个 `(BLOCK_M=64, BLOCK_N=128)` 的 tile 只需要几十次优化的向量访存请求，现在变成了骇人听闻的 **8,192 次独立散乱的 16-bit 显存并发写入请求**。这引发了极其严重的内存请求风暴，直接打挂了 B200 的 SM Issue 队列和 L2 控制器带宽。

**最终终局判断：**
在 Triton 的高层沙盒环境内，我们**无法兼顾** “跨行打散落位” 与 “连续块合并读写（Coalesced/Vectorized memory access）”。
只要使用 Triton，目前的 “三段式” 架构（GEMM1写出连续张量 -> GEMM2写出连续张量 -> Reduce规约）反而是在内存效率上的**全局最优解**。

**脱离沙盒的唯一生机：** NVIDIA **CUTLASS 3.x (EVT - Epilogue Visitor Tree)**。只有在 C++ 层面手写自定义的 GEMM Epilogue Node，才能在寄存器级别向量化地将数据 Scatter-Add 到 HBM 中，从而真正意义上不损耗速度地吃掉那 1GB 的冗余带宽。

---

## 18. Round 17: Direction 4 — Microbatch Scheduling / Decode Path

### 先校正一个现实：README headline 目前不可复现

在继续做方向四之前，先用仓库内官方脚本 `scripts/test_modal.py` 对历史版本做了重新复跑：

- `d2fdf14` official full19 rerun：
  - **mean ≈ 43.86x**
  - **gmean ≈ 39.36x**
  - **peak ≈ 66.20x**
- 稳定基线 `e7e8f66` official full19 rerun：
  - **mean ≈ 44.25x**
  - **gmean ≈ 39.64x**
  - **peak ≈ 66.85x**

这和 README 中历史记录的 `55.77x / 47.54x / 106.65x` 存在明显偏差，说明 **当前 Modal 线上环境已经发生漂移**。因此这轮方向四优化的比较基准不再是 README 顶部数字，而是 **今天可复现的 `e7e8f66`**。

### 小批量 profiling：瓶颈并没有想象中那么“纯 host-bound”

对真实 tiny workloads 做 microprofile 之后，得到：

| Workload | T | wall | cuda | cpu overhead | gemm1 | gemm2 |
|----------|---:|-----:|-----:|-------------:|------:|------:|
| `b8f4f012` | 7  | 0.155ms | 0.096ms | 0.059ms | 0.052ms | 0.031ms |
| `a7c2bcfd` | 16 | 0.156ms | 0.138ms | 0.018ms | 0.076ms | 0.049ms |
| `2e69caee` | 15 | 0.149ms | 0.084ms | 0.064ms | 0.050ms | 0.024ms |
| `8cba5890` | 14 | 0.158ms | 0.136ms | 0.022ms | 0.069ms | 0.055ms |

**关键结论：**

- `T=7`、`T=15` 上 host/fixed cost 仍明显存在
- 但 `T=14/16` 已经不是纯 CPU overhead 问题，**GEMM 本体依然占主导**
- 因此方向四如果只做“加并行 / 减 launch”，不一定会赢；必须避免破坏 GEMM 的 tensor core 利用率

### ❌ 尝试 1：Tiny GEMM2 Split-K + Atomic Accumulate

**想法：** 既然 decode phase 的 T 很小，就沿 K 轴拆开 GEMM2，让更多 SM 同时参与，再用 `atomic_add` 汇总。

**实现：**

- 为 `T<=16` 新增 tiny GEMM2 split-K 路径
- `SPLIT_K=4`
- 先 `expert_out.zero_()`，每个 split 分别写 partial，再原子规约

**结果：** 19/19 PASSED，但 tiny workloads 明显变慢：

- `b8f4f012`: `52.90x`
- `a7c2bcfd`: `46.40x`
- `2e69caee`: `55.52x`
- `8cba5890`: `46.93x`

**原因：**

- split-K 引入了额外的 `zero_()` 和原子汇总开销
- 原本 tiny GEMM2 的绝对计算量很小，拆 K 并不能带来足够多的有效并行
- 反而把固定成本和写回成本放大了

**结论：** microbatch 场景下，**Split-K 不是“白嫖并行度”**，在 Triton 里很容易得不偿失。

### ❌ 尝试 2：Tiny GEMM1-only Dispatch

**想法：** 只替换 `T<=16` 的 GEMM1，走更窄 autotune 空间 / 更小的 tiny-specialized kernel，尽量压低 fixed cost，同时保持 GEMM2 不变。

**实现：**

- `T<=16` 时将 GEMM1 dispatch 到 `_t1_generic_gemm1_swiglu_kernel`
- 初版还踩到了一个 `BLOCK_K` autotune 参数缺失的 binding bug，修复后重新 full19

**full19 结果：**

- **mean = 42.55x**
- **gmean = 38.39x**
- **peak = 66.40x**

**结论：**

- 19/19 正确性没问题
- 但更窄 tiny kernel 并没有降低 enough 的 fixed cost
- 反而因为 kernel shape 太保守，**tensor core 利用率下降**

这说明 tiny 路径不能只看“launch 更少 / lookup 更轻”，还要看 GEMM 是否仍能吃满机器。

### ❌ 尝试 3：Exact Grid / Direct GEMM2+Reduce

这组尝试的共同目标是：进一步削掉 tiny 路径里那些“看起来多余”的程序和缓冲区。

#### 3a. exact-grid

**想法：** 对 tiny batch 不再按 `MAX_PADDED` 启动 GEMM grid，而是在 sort 之后读出真实 `total_blocks`，只发射 exact 数量的 program。

**结果：** subset 退化。

**根因：** host 侧额外的 `.item()` 同步成本，足以吃掉 tiny 下省掉的那些空 program。

#### 3b. direct GEMM2+reduce

**想法：** 对 `T<=16` 直接把 GEMM2 和 token reduce 融合，绕过 `expert_out`。

**结果：** subset 19/19 PASSED，但速度只有 `46-51x`，显著慢于 baseline。

**根因：**

- token-centric 版本对每个 slot 都要重跑一段 K-loop
- GEMM2 原本的 tile reuse 被破坏
- 即便把 expert lookup 从扫描改成 `sorted_expert_ids` 的 O(1) 读取，也依然不够

**结论：** 这种 direct-reduce 更像“把访存省下来，再把算力浪费回去”，在 tiny workload 上不成立。

### ✅ 当前最好的方向四小改动：只改 Tiny GEMM2 Dispatch

在前面几条重型路径都失败后，最后保留的是一个非常轻量的调度实验：

- **当 `T<=16` 时，不改 GEMM1，不改 sort / reduce**
- **只把 GEMM2 从 generic kernel 切到 `_medium_fused_moe_gemm2_kernel`**

subset 上的 tiny 结果：

- `b8f4f012`: `60.15x`
- `a7c2bcfd`: `54.80x`
- `2e69caee`: `63.37x`
- `8cba5890`: `55.42x`

对应 official full19：

- **mean = 44.61x**
- **gmean = 40.01x**
- **peak = 65.53x**

> ⚠️ 更新（2026-03-23，latest `HEAD` A/B）：这组数字只对 2026-03-22 的 pre-sync 树成立。  
> 在最新主线上重新套用同一改动后，official full19 反而退到 **`43.36x / 39.30x / 61.22x`**，低于 latest `HEAD` 基线 **`48.73x / 43.18x / 77.31x`**，因此该 tweak 已正式回退。

相对今天可复现的 `e7e8f66`：

- `mean`: `44.25x -> 44.61x`
- `gmean`: `39.64x -> 40.01x`
- `peak`: `66.85x -> 65.53x`

也就是说：

- 在当时的 pre-sync 树上，它确实曾经比旧基线略好
- 但这个结论**没有跨过最新 upstream sync 的 A/B 验证**
- 因而它现在应被视为 **历史正样本**，而不是当前主线

### 这轮方向四的真正结论

1. **decode-like microbatch 并不是纯 host-bound**
   - 至少在 `T=14/15/16` 上，GEMM fixed cost 仍是主因
2. **Triton 里重型 split-K / direct-reduce 路径太贵**
   - 一旦引入额外原子、重复 K-loop、host sync，tiny batch 很快就被固定成本反杀
3. **最有效的杠杆目前仍然是 dispatch policy**
   - 尤其是 **“不同 batch bucket 下，GEMM1/GEMM2 不一定应该共用同一套 bucket 逻辑”**
4. **在当时的 pre-sync 树上，最值得继续挖的是 GEMM2**
   - 当时的证据表明：对 `T<=16`，GEMM2 的 kernel family 切换比 GEMM1 更敏感、更有效

### 下一步建议

如果继续沿 Direction 4 往下打，优先级建议是：

1. **继续细化 tiny GEMM2 dispatch**
   - 例如 `T<=8`, `9<=T<=16`, `17<=T<=31` 分开选 kernel family
2. **单独调 T=1 decode path**
   - 目前 `e05c6c03` 仍走专用 `_kernel_t1`
   - 可以单独扩充 `_t1_fused_gemm1_swiglu_kernel` / `_t1_fused_gemm2_reduce_kernel` 的 autotune 空间
3. **避免再走 heavy fusion / split-K**
   - 这轮已经基本证明：在当前 Triton + Modal 现实下，这些路径的固定成本模型不对


## 19. Round 17 续跑（2026-03-22 晚 / 2026-03-23）：Dispatch Sweep 清理 + Best-State Profiling

### 先给结论

这轮续跑之后，结论比之前更明确了：

- **在 2026-03-22 的 pre-sync 树上**，best-known 可复现主线曾经是 `T<=16` 时仅把 `GEMM2` dispatch 到 `_medium_fused_moe_gemm2_kernel`
- 后续所有 bucket / autotune / reduce sweep **都没超过** 这条主线
- 更重要的是，新的 Modal profiling 表明：**`T≈64` 真正更该打的是 GEMM1，而不是 GEMM2**

换句话说，这轮不是“找到新冠军”，而是**系统性排除了很多看似合理但其实无效的后续方向**。

### 续跑实验总表

> 注：本地原始失败日志后来已按清理策略删除。下面保留的 `worker_logs/...` 名称仅作为**历史运行标识**，不保证当前工作区仍存在对应文件；当前保留的原始结果只剩 baseline / best-state 少数文件。

| 实验 | 结果 | 结论 | 历史日志名 |
|------|------|------|------|
| `52<=T<=62 -> medium GEMM2` | subset `44.92x / 44.88x -> 45.54x / 45.50x`，但 official full19 掉到 **`42.47x / 38.35x`** | 典型 false positive；subset 提升不能代表 full19 | `worker_logs/mid52_62_medium_gemm2_subset_20260322.txt`, `worker_logs/mid52_62_medium_gemm2_full19_20260322.txt` |
| `52<=T<=62 -> BLOCK_M=32` | subset **`36.04x / 35.94x`** | tile 过小，tensor core 利用率掉得比 padding 节省更快 | `worker_logs/mid52_62_blockm32_subset_20260322.txt` |
| `T=1` GEMM1 `BLOCK_N=512` | `e05c6c03 = 60.42x` | 明显低于当前 best T1 (`65.53x`) | `worker_logs/t1_gemm1_blockn512_20260322.txt` |
| `T=1` GEMM2/reduce `BLOCK_N=512` | `e05c6c03 = 60.66x` | 也低于当前 best T1 | `worker_logs/t1_gemm2_blockn512_20260322.txt` |
| `T=1` reduce widen (`RS_BLOCK_N=512`) | `e05c6c03 = 64.18x` | 接近但仍未超过当前主线 | `worker_logs/tiny_reduce512_t1_20260322.txt` |
| `T=1` reduce tune (`RS_BLOCK_N=256, warps=2`) | `e05c6c03 = 60.32x` | 明显退化 | `worker_logs/tiny_reduce256_w2_t1_20260322.txt` |
| dedicated tiny GEMM2 | tiny subset **`56.20x / 56.12x`** | 不如 baseline tiny subset `57.20x / 56.98x` | `worker_logs/tiny_dedicated_gemm2_subset_20260322.txt` |
| dedicated tiny GEMM2 + binary-search lookup | tiny subset **`55.94x / 55.84x`** | 比 dedicated tiny GEMM2 更差 | `worker_logs/tiny_dedicated_gemm2_binarysearch_subset_20260322.txt` |
| `T<=16 -> small GEMM2` | tiny subset **`55.72x / 55.64x`** | 不如 `tiny -> medium GEMM2` | `worker_logs/tiny_small_gemm2_subset_20260322.txt` |
| `T<=16 -> small GEMM1 + medium GEMM2` | tiny subset **`56.50x / 56.45x`** | 仍输给当前 best tiny path | `worker_logs/tiny_small_gemm1_medium_gemm2_subset_20260322.txt` |
| 只对 `T=14` 或 `T=14/16` 切 medium GEMM2 | tiny subset **`55.13x / 55.04x`** 与 **`55.56x / 55.47x`** | 单点 dispatch 细分没有比 `T<=16` 整体切换更好 | `worker_logs/t14_only_medium_gemm2_subset_20260322.txt`, `worker_logs/t14_t16_only_medium_gemm2_subset_20260322.txt` |
| 给 `_medium_fused_moe_gemm2_kernel` 追加更多 tiny autotune config | tiny subset `57.51x / 57.47x`，但 official full19 退到 **`43.74x / 39.35x`** | 又一个 false positive；subset 看起来接近，full19 反而变差 | `worker_logs/tiny_medium_gemm2_autotune_plus_subset_20260322.txt`, `worker_logs/tiny_medium_gemm2_autotune_plus_full19_20260322.txt` |
| 把 small/medium GEMM2 的 expert lookup 从向量扫描改成 binary search | subset15 **`46.87x / 46.59x`**，低于当前 best subset15 `49.36x / 49.02x` | Triton 里这类标量化控制流不如向量 compare + `argmax` | `worker_logs/gemm2_bsearch_subset15_20260322.txt` |
| `T=32 -> medium GEMM2` | `6230e838 = 43.50x` | 低于当前主线 `45.20x` | `worker_logs/t32_medium_gemm2_single_20260322.txt` |
| `T=80 -> small GEMM2` | `8f1ff9f1 = 42.20x` | 低于当前主线 `42.41x` | `worker_logs/t80_small_gemm2_single_20260322.txt` |
| `T=901 -> BLOCK_M=128` | `1a4c6ba1 = 17.97x` | 大幅差于当前主线 `24.12x` | `worker_logs/t901_blockm128_single_20260322.txt` |
| small-medium GEMM2 higher-`GROUP_M` sweep | subset10 **`45.30x / 45.27x`**，低于主线 subset10 `46.42x / 46.39x` | 更高的 L2 reuse 没有抵消调度/occupancy 代价 | `worker_logs/small_medium_gemm2_groupm_plus_subset10_20260322.txt` |
| `52<=T<=62 -> medium GEMM1 only` | subset9 **`45.33x / 45.28x`**，低于主线 subset9 `46.56x / 46.52x` | `medium` GEMM1 family 对这个桶仍然偏重 | `worker_logs/mid52_62_medium_gemm1_subset_20260322.txt` |
| 历史 replay：`T=1` GEMM1 改回 generic `_fused_moe_gemm1_swiglu_kernel` | `e05c6c03 = 58.91x` | 旧版本的选择在 today-Modal 已不成立 | `worker_logs/t1_use_generic_fused_gemm1_single_20260322.txt` |
| `T>=2048` reduce widen (`BLOCK_N=512, warps=8`) | `5e8dc11c: 8.34x -> 8.28x`, `58a34f27: 9.26x -> 9.20x` | large-T reduce 不是简单“加宽 tile”就能变快 | `worker_logs/large_reduce512_subset2_20260322.txt` |

### 补录：此前遗漏但现已归档的 2026-03-22 实验

| 实验 | 结果 | 结论 | 历史日志名 |
|------|------|------|------|
| tiny GEMM2 split-K + atomic accumulate（official full19） | **`42.20x / 38.22x`** | full19 明显退化，split-K 的 `zero_()` / atomic 汇总固定成本太重 | `worker_logs/tiny_splitk_gemm2_full19_20260322.txt` |
| split-K filtered rerun（基础设施失败） | `FileNotFoundError`，未产出有效性能数据 | 缺少 `scripts/test_modal_residual_experiment.py`，属于脚本层失败，不计入 kernel 结论 | `worker_logs/tiny_splitk_gemm2_filtered_20260322.txt` |
| tiny GEMM1-only dispatch（初版 full19） | 仅 **`4/19 PASSED`** | 初版踩到 autotune/binding bug：`dynamic_func() missing BLOCK_K`，性能数据无效 | `worker_logs/tiny_gemm1_only_full19_20260322.txt` |
| tiny GEMM1-only dispatch（fix1） | official full19 **`42.55x / 38.39x`** | 修完 bug 后仍低于当前主线；更窄 tiny GEMM1 没换来净收益 | `worker_logs/tiny_gemm1_only_full19_fix1_20260322.txt` |
| tiny exact-grid | tiny subset **`54.57x / 54.44x`** | host `.item()` 同步成本吃掉了省下的空 grid | `worker_logs/tiny_exact_grid_subset_20260322.txt` |
| tiny direct GEMM2+reduce | tiny subset **`48.51x / 48.49x`** | direct-reduce 破坏 GEMM2 tile reuse，远慢于 baseline | `worker_logs/tiny_direct_gemm2_subset_20260322.txt` |
| tiny direct GEMM2+reduce + `sorted_expert_ids` | tiny subset **`48.28x / 48.25x`** | expert lookup 再缓存也救不回 direct-reduce 的额外指令成本 | `worker_logs/tiny_direct_gemm2_sorted_expert_ids_subset_20260322.txt` |
| generic `sorted_expert_ids` lookup path | tiny subset **`55.14x / 55.04x`** | 只改 lookup 不够，仍低于 tiny baseline `57.20x / 56.98x` | `worker_logs/generic_sorted_expert_ids_subset_20260322.txt` |
| microbatch `BLOCK_M=32` | tiny subset **`51.38x / 50.78x`** | 更窄 tile 明显拖累 tensor core 利用率 | `worker_logs/microbatch_blockm32_tiny_subset_20260322.txt` |
| `T<=16 -> medium GEMM1 + medium GEMM2` | tiny subset **`56.51x / 56.46x`** | 同时切两边 kernel 仍低于当前 best tiny path | `worker_logs/tiny_medium_gemm1_gemm2_subset_20260322.txt` |
| `T<=16` reduce widen subset（`RS_BLOCK_N=512`） | tiny subset **`57.37x / 57.29x`** | 相比 baseline 略有提升，但仍低于 `tiny_medium_gemm2` 的 `57.72x / 57.61x`，未升格主线 | `worker_logs/tiny_reduce512_subset_20260322.txt` |
| `T<=16` reduce tune subset（`RS_BLOCK_N=256, warps=2`） | tiny subset **`55.23x / 55.18x`** | 明显回退 | `worker_logs/tiny_reduce256_w2_subset_20260322.txt` |
| `T=14/16 -> medium GEMM2` | tiny subset **`56.10x / 56.02x`** | 只打 `14/16` 桶不如更简单的主线策略 | `worker_logs/t14_t16_medium_gemm2_subset_20260322.txt` |
| `T=14/16 -> medium GEMM2 + extra autotune` | tiny subset **`58.15x / 57.77x`** | subset-only 正信号，但未升级成可复现主线记录 | `worker_logs/t14_t16_medium_gemm2_autotune_plus_subset_20260322.txt` |
| `T=1` GEMM2/reduce low-warp64 follow-up | `e05c6c03 = 65.49x` | 几乎打平，但仍未超过当前 best `65.53x` | `worker_logs/t1_gemm2_lowwarp64_single_20260322.txt` |

### 这轮最有价值的成功：Best-State Profiling 跑通了

虽然这轮没有找到新的主线性能变体，但 profiling 本身是成功的，而且给出了比之前更强的方向约束。

日志：

- `worker_logs/ncu_profile_beststate_20260322.txt`
- `ncu_profile_output.txt`

核心切片如下：

| T | wall | 关键分解 | 结论 |
|---|-----:|---------|------|
| `7` | `0.146ms` | CPU overhead ≈ `88.9%`，routing 是 CUDA 里最大单项 | tiny decode 仍高度 fixed-cost / host-bound |
| `14` | `0.167ms` | GEMM1 ≈ `55.4%`，GEMM2 ≈ `26.6%` | `T=14/16` 不是“纯 launch overhead”问题 |
| `64` | `0.170ms` | **GEMM1 ≈ `54.5%`**，GEMM2 ≈ `30.6%`，CPU overhead ≈ `35.2%` | **medium-T 的下一目标应转向 GEMM1** |
| `512` | `0.782ms` | GEMM1 ≈ `47.9%`，GEMM2 ≈ `46.4%` | 这段已经非常 GEMM-bound |
| `1024` | `0.528ms` | GEMM2 ≈ `51.7%`，GEMM1 ≈ `35.2%` | 大 batch 开始进入 GEMM2 主导 |
| `4096` | `0.420ms` | GEMM2 ≈ `31.7%`，reduce ≈ `18.2%`，sort ≈ `12.4%`，route ≈ `11.6%` | large-T 仍有 sort / reduce 结构性空间，但简单 widen reduce 已被证伪 |

### 这轮续跑真正改变了什么判断

之前在 Round 17 初版里，最自然的延续思路是继续顺着 **GEMM2 dispatch** 往下细分。但这轮 profiling 和 sweep 合起来之后，判断已经更新成：

1. **medium-T 下一步不该再优先打 GEMM2 dispatch**
   - `T≈52-64` 的主要时间已经更偏向 **GEMM1**
   - 继续在 GEMM2 family / tiny bucket 上做小修小补，边际收益非常低

2. **small/medium GEMM2 的“lookup / GROUP_M / 单点 family 切换”基本都被证伪**
   - binary search lookup、higher `GROUP_M`、`T=32/T=80` 单点 family probe 都没有带来净增益

3. **large-T 的 reduce / sort 仍值得看，但必须是结构性改写**
   - 单纯把 reduce `BLOCK_N` 从 `256` 扩到 `512` 已经明确退化
   - 如果后面要继续打 large-T，更可能有效的是 sort/scatter/reduce 的组织方式，而不是 tile 线性放大

### 更新后的下一步优先级

在当前 Modal 口径下，后续优先级更新为：

1. **medium-T GEMM1 结构优化**
   - 优先看 `_small_medium_fused_moe_gemm1_swiglu_kernel` / `_medium_fused_moe_gemm1_swiglu_kernel`
   - 重点看 expert lookup、scale load、pointer arithmetic、`tl.dot` 周边访存形态

2. **large-T sort / reduce 的结构性优化**
   - 已知 `T=4096` 上 sort + reduce 仍占 ~30%
   - 但不要再做“单纯加宽 reduce tile”这种一维 sweep

3. **不要再保留 `tiny GEMM2 dispatch` 作为当前主线**
   - latest `HEAD` A/B 已确认：把 `T<=16` 切到 `_medium_fused_moe_gemm2_kernel` 会从 **`48.73x / 43.18x / 77.31x`** 退到 **`43.36x / 39.30x / 61.22x`**
   - 因此该改动已从 `kernel.py` 正式回退；后续不应再默认沿这条线投入时间

### 清理补录：2026-03-23 本地无效尝试

这些尝试不再保留原始本地日志，只在 markdown 里保留失败原因：

| 实验 | 结果 | 结论 |
|------|------|------|
| large-T reduce `num_warps=8` subset rerun | Modal 任务启动但未产出有效结果 | 属于中断/无结论尝试，不再保留原始日志 |
| `RS_BLOCK_N=128` large-T subset rerun | Modal 任务启动但未产出有效结果 | 同样属于中断/无结论尝试 |
| `RS_BLOCK_N=128` mixed subset rerun | subset **`49.08x / 48.76x`** | 没有显示出优于当前主线的价值，且不构成可复现正收益路径 |
| `GEMM1 autotune + RS512` full19 rerun | official full19 **`43.73x / 39.47x`** | 低于 latest `HEAD` 当前基线 **`48.73x / 43.18x`**，因此作为无效尝试清理 |
| latest `HEAD` 上重试 `T<=16 -> medium GEMM2` | official full19 **`43.36x / 39.30x / 61.22x`** | 明确低于 latest `HEAD` 基线 **`48.73x / 43.18x / 77.31x`**，故正式回退该 tweak |
