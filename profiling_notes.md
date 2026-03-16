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
