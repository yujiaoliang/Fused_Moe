# Fused MoE Kernel — Track A (FlashInfer AI Kernel Generation Contest @ MLSys 2026)

> **赛道:** Track A — Fused MoE
> **目标硬件:** NVIDIA B200 (Blackwell, sm100)
> **当前状态:** ✅ 19/19 PASSED｜最高 **12.56x** 加速｜large-T **6.8-7.3x**｜平均 **~10.2x**

---

## 当前进度

| 项目 | 状态 |
|------|------|
| DeepSeek-V3 routing | ✅ 完成 |
| Token sorting (expert 分组) | ✅ 完成（`moe_sort_tokens()` + expert 边界检测） |
| FP8 block-scale Triton Fused Kernel | ✅ 完成（`_fused_moe_gemm1_swiglu_kernel` 和 `_fused_moe_gemm2_scatter_kernel`） |
| Native FP32 精度对齐 | ✅ 完成（Triton 内全 FP32 math，通过所有数值测试） |
| Benchmark 正确性 | ✅ 19/19 PASSED (100% Numerically Correct) |

### B200 Benchmark 结果（最新）

Round 7 优化：`torch.compile(mode="reduce-overhead", dynamic=True)` 融合 routing ~14 个 PyTorch ops 的 dispatch overhead。Pre-allocated `output_fp32` buffer cache。CUDA Graph 尝试后回退（见下方分析）。平均 ~10.2x（在噪声范围内微小提升）。

| Workload | Round 1 | Round 5 | Round 6 | Round 7 | 备注 |
|----------|---------|---------|---------|---------|------|
| e05c6c03 | 12.90x | 11.91x | 12.14x | **12.56x** | 🔥 peak |
| 2e69caee | 12.21x | 11.54x | 11.97x | **12.05x** | |
| b8f4f012 (T=7) | 12.02x | 10.93x | 11.06x | **11.67x** | |
| 1a4c6ba1 | 1.35x | 10.80x | 11.34x | **11.09x** | |
| 5eadab1e | 10.06x | 9.98x | 10.76x | **10.99x** | |
| 8cba5890 | 10.14x | 10.35x | 10.78x | **10.89x** | |
| a7c2bcfd (T=128)| 10.00x | 10.00x | 10.84x | **10.83x** | |
| f7d6ac7c | 9.60x | 9.85x | 10.53x | **10.57x** | |
| e626d3e6 | 9.18x | 9.35x | 9.98x | **10.17x** | |
| 8f1ff9f1 | 8.80x | 8.95x | 10.02x | **10.06x** | |
| 4822167c | 8.75x | 9.68x | 10.19x | **10.02x** | |
| fc378037 | 9.02x | 9.52x | 10.06x | **10.02x** | |
| 74d7ff04 | 9.02x | 9.68x | 10.10x | **10.00x** | |
| 6230e838 | 8.97x | 9.88x | 10.05x | **9.98x** | |
| eedc63b2 | 9.30x | 9.32x | 10.18x | **9.92x** | |
| 76010cb4 | 8.76x | 9.39x | 10.25x | **9.86x** | |
| 81955b1e | 8.94x | 9.71x | 10.10x | **9.86x** | |
| 58a34f27 (T=4096)| 4.14x | 7.46x | 7.43x | **7.11x** | large-T |
| 5e8dc11c (T=4096)| 3.69x | 7.08x | 6.98x | **6.73x** | large-T |

---

## 环境搭建

```bash
# 1. 创建 conda 环境
conda create -n fi-bench python=3.12
conda activate fi-bench

# 2. 安装依赖
pip install flashinfer-bench modal torch triton numpy

# 3. 克隆比赛数据集
git lfs install
git clone https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest

# 4. Modal 登录（一次性）
modal setup
```

---

## 在 Modal B200 上运行测试（完整流程）

### Step 1: 创建 Modal Volume 并上传数据（一次性）

```bash
# 创建 volume
modal volume create flashinfer-trace

# 上传数据集到 volume（注意：会创建 /mlsys26-contest 子目录）
modal volume put flashinfer-trace /path/to/mlsys26-contest /
```

> ⚠️ 上传后数据路径为 `/mlsys26-contest/` 在 volume 内，volume 挂载在 `/data`，
> 所以 TraceSet 路径是 `/data/mlsys26-contest`。

### Step 2: 打包 solution

```bash
python scripts/pack_solution_simple.py
```

这会生成 `solution.json`，包含 `kernel.py` 的源码。

### Step 3: 在 B200 上运行 benchmark

```bash
# 方式1：使用 test_modal.py（推荐，输出更详细）
python -m modal run scripts/test_modal.py

# 方式2：使用 run_modal.py（完整 benchmark 框架）
python -m modal run scripts/run_modal.py
```

输出示例：
```
GPU: NVIDIA B200
CUDA: (10, 0)
PyTorch: 2.10.0+cu128

=== Results (19 traces) ===
  b8f4f012: PASSED | 8.920ms | 1.29x | abs=0.00e+00, rel=0.00e+00
  e05c6c03: PASSED | 4.560ms | 2.40x | abs=0.00e+00, rel=0.00e+00
  ...
=== Summary: 19/19 PASSED ===
```

### Step 4: 本地快速验证（可选，RTX 4080 等）

```bash
# 只测试最小 workload（T=7），不需要 benchmark 框架
python test_kernel.py
```

> 注意：本地 GPU 显存 < 24GB 跑不了大 workloads（reference 实现也会 OOM）。

---

## 项目结构

```
mlsys_note/
├── solution/
│   ├── triton/kernel.py         # ← Triton kernel（主力优化分支）
│   └── cuda/
│       ├── binding.py           # CUDA baseline：PyTorch/cuBLAS 参考实现（19/19 PASSED）
│       └── kernel.cu            # CUDA kernel 模板 + FA4 fast_sigmoid（供队友优化）
├── check_modal.py               # Modal B200 快速正确性检查
├── test_kernel.py               # 本地快速测试（直接对比 reference）
├── scripts/
│   ├── test_modal.py            # Modal B200 benchmark（推荐）
│   ├── run_modal.py             # Modal B200 完整 benchmark
│   ├── profile_modal.py         # Modal B200 NCU profiling
│   ├── debug_modal.py           # Modal B200 调试（逐步对比 reference）
│   ├── ncu_profile_modal.py     # Modal B200 torch.profiler 时间分解 + roofline 分析
│   ├── pack_solution_simple.py  # 打包 solution.json（支持 --cuda 切换打包 CUDA baseline）
│   ├── pack_solution.py         # 打包（需要 flashinfer_bench.agents）
│   ├── run_local.py             # 本地 benchmark
│   └── test_scaled_mm.py        # 测试 torch._scaled_mm API
├── config.toml                  # 配置（队名、赛道等）
├── profiling_notes.md           # B200 profiling 分析、优化尝试记录、下一步方向
├── solution.json                # 打包后的提交文件
└── mlsys26-contest/             # 比赛数据集（git submodule）
```

---

## Kernel 架构

```
kernel(routing_logits, routing_bias, hidden_states, hidden_states_scale,
       gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
       local_expert_offset, routed_scaling_factor, output)
                                                    ↑ 最后一个参数是 output（destination-passing style）
```

### 计算流程

1. **Routing** — DeepSeek-V3 no-aux routing：sigmoid → group-filter → top-8 → normalized weights
2. **Token sorting** — `moe_sort_tokens()` 按 expert 排序 token，padding 到 BLOCK_M=64
3. **Monolithic Triton Kernel 1: GEMM1 + SwiGLU**
   - `_fused_moe_gemm1_swiglu_kernel`（`@triton.autotune` 自动调参）
   - FP8 Activation & Weights 传入
   - **Native FP8 Tensor Core Dot** (`tl.dot(fp8, fp8)`) + post-dot scale multiply，2x 吞吐 vs TF32
   - 输出 fp32 `Intermediate` [num_padded, 2048]（bf16 精度不足，6/19 失败）
4. **Monolithic Triton Kernel 2: GEMM2 + Scatter Add**
   - `_fused_moe_gemm2_scatter_kernel`（`@triton.autotune` 自动调参，BLOCK_K=128）
   - Post-dot B-scale：`tl.dot(a, b.to(f32))` + `acc += partial * b_scale`，减少标量乘法并提升 TF32 精度
   - 使用 Triton `tl.atomic_add` 直接将更新写入最终的 fp32 buffer，避免低精度 rounding error，随后复制至 `bfloat16` output。

---

## CUDA Baseline（队友优化起点）

`solution/cuda/` 包含完整可运行的 CUDA 版本：

- **`binding.py`** — PyTorch/cuBLAS 参考实现（19/19 PASSED）
  - 已优化 routing（gather-first、bool mask、torch.compile）
  - Per-expert 循环：FP8 dequant(fp32) → `torch.matmul` (cuBLAS) → SwiGLU → scatter-add
  - Pre-allocated buffer cache
  - 精度优于 Triton 版（abs ~0-2K vs ~2K-4K），速度受限于每 expert ~112MB 权重物化

- **`kernel.cu`** — Fused CUDA kernel 模板
  - `fast_sigmoid()` — **FA4 启发**：Padé 有理逼近替代 SFU `expf()`（B200 上 SFU 吞吐没跟上 Tensor Core 增长）
  - `gemm1_swiglu_kernel` — Tiled GEMM1 + SwiGLU（BM=64, BN=64, BK=32, 4×4 register tiling）
  - `gemm2_scatter_kernel` — Tiled GEMM2 + weighted atomicAdd scatter
  - `extern "C"` launch wrappers（ctypes/pybind11 接口）

### 测试 CUDA Baseline

```bash
# 无需手动 copy 文件！--cuda 自动打包 solution/cuda/binding.py
python scripts/pack_solution_simple.py --cuda && python check_modal.py
```

### CUDA 优化路径

```
Step 1: 用 kernel.cu 的 fused kernel 替换 binding.py 的 per-expert PyTorch 循环
Step 2: 消除 fp32 weight 物化 — 在 GEMM tile 内在线 dequant
Step 3: CUTLASS 3.x sm100a 模板 — TMEM, TMA, wgmma
Step 4: Persistent kernel — 消除 per-expert launch overhead
Step 5: 更大的 tile sizes (BM=128, BN=256) + multi-stage pipeline
```

### 关键约束

- **Destination-passing style:** 框架调用 `kernel(*inputs, *outputs)`，output 是最后一个参数
- **正确性标准:** 95% 元素满足 `abs_err ≤ 0.01` OR `rel_err ≤ 0.01`
- **Reference 实现:** 全部 fp32 dequant + fp32 matmul（我们的实现匹配了这个策略）

---

## TODO（优化方向）

### ✅ 已完成

- [x] **Token sorting** — `moe_sort_tokens()` 按 expert 分组 + BLOCK_M padding
- [x] **Expert 边界检测** — 遍历 `block_expert_ids` 直接定位 expert 边界
- [x] **fp32 accumulation** — scatter-add 用 fp32 避免 bf16 精度损失
- [x] **view+expand dequant** — 零拷贝广播替代 `repeat_interleave`
- [x] **F.silu** — 融合 CUDA kernel 替代手动 `x*sigmoid(x)`
- [x] **torch.compile** — `max-autotune-no-cudagraphs, dynamic=True` 融合 dequant+GEMM+SwiGLU，**全部 19/19 ≥1x**
- [x] **编译 A dequant** — `_dequant_compiled` 融合 fp8→fp32 cast + scale multiply
- [x] **预转换 weight scales** — `.float()` 移到循环外，避免每 expert 转换开销
- [x] **Monolithic Fused Kernels** - 完美消除 Python 层面上基于各个 expert 的迭代 launch 惩罚，使用分组索引跨 block 读取。
- [x] **Native TF32 数值等价** - 解决 `tl.dot()` 中各种 FP8/BF16 downcast 带来的误差。完全对齐纯 Eager 模式下的精度限度 (100% Correctness通过率)。
- [x] **Triton Autotune** — `@triton.autotune` 自动选择最优 BLOCK_M/BLOCK_N/num_warps/num_stages 配置（需 `restore_value=['C_ptr']` 防止 atomic_add 在多次 trial 间累积）
- [x] **GEMM2 BLOCK_K=128** — 与 QBLOCK=128 对齐，K-loop 迭代次数减半（2048/128=16 vs 2048/64=32）
- [x] **torch.empty for Intermediate** — 避免不必要的零初始化（kernel 内 padding 行已 mask）
- [x] **Hoisted safe_token_idx** — 将 safe_token_idx 计算移出 GEMM1 K-loop，减少冗余计算
- [x] **Fused tl.dot accumulation** — `acc=` 参数融合 GEMM1 和 GEMM2 的 dot 累加
- [x] **GPU Token Sorting** — vectorized cumsum + scatter，单次 `.item()` sync（大 T 提升但小 T 回退）
- [x] **Extended Autotune** — GEMM1 +3 configs (stages=5,2), GEMM2 +4 configs (BLOCK_N=256, stages=2-4)
- [x] **fp32 Intermediate** — bf16 SwiGLU output 精度不足（6/19 PASSED），改用 fp32
- [x] **Routing gather-first** — 先 gather [T,8]，再 normalize on [T,8] 替代 [T,256] 上的 mask+mul+div+gather（最大提升 +1.2x）
- [x] **Bool group mask** — `dtype=torch.bool` + `~s_mask` 避免 float→bool 临时 tensor 分配
- [x] **Pre-allocated pad buffers** — CPU sorting path 中单次 `torch.full`/`torch.zeros` 替代每 expert 分配
- [x] **移除冗余 cast** — GEMM2 `atomic_add` 中 `.to(tl.float32)` 已冗余（out 本身即 fp32）
- [x] **FP8 Native Tensor Core Dot (GEMM1)** — `tl.dot(fp8, fp8)` + post-dot scale，Triton 3.6.0 已修复 sm100 codegen（2x 吞吐，large-T +3x，avg +1.9x）
- [x] **GEMM2 Post-dot B-scale** — `tl.dot(a, b.to(f32))` + `acc += partial * b_scale`，消除 [BLOCK_K, BLOCK_N] dequant 乘法（中等规模 +0.3-0.8x，avg +0.5x）
- [x] **Autotune 扩展** — GEMM1 增加 BLOCK_N=256 配置，GEMM2 增加 num_stages=5 和 BLOCK_N=256 配置
- [x] **NCU Profiling** — B200 上使用 `torch.profiler` 做 per-kernel 时间分解 + analytical roofline 分析（见 `profiling_notes.md`）
- [x] **torch.compile routing** — `torch.compile(mode="reduce-overhead", dynamic=True)` 融合 routing 中的 element-wise ops（微小提升 ~1-2%，在噪声范围内）
- [x] **Pre-allocated buffer cache** — `output_fp32` 跨 call 复用，避免每次 `torch.zeros` 分配

### 🟡 P0: CPU Overhead 优化（当前瓶颈）

> **关键发现（NCU Profiling）：** CPU overhead 占 wall time 的 56-73%（~600us），其中 routing ~300us（~30 PyTorch kernel launches）、sorting ~100us、Python framework ~170us。GEMM kernel 本身只占 13-21% of wall time。
>
> **结论：** 进一步优化 GEMM tile 效果有限，**必须减少 routing/sorting 的 kernel launch 数量**。

- [ ] **Fuse routing into Triton kernel** — 将 ~14 个 PyTorch ops（sigmoid, topk×3, scatter, gather, masked_fill, etc.）融合为 1 个 Triton kernel，预期节省 ~300us（最高优先级，但 topk 在 Triton 中实现复杂）
- [ ] **Fuse sorting into Triton kernel** — 替代 argsort + CPU sync（`.cpu().tolist()`），预期节省 ~100us
- [ ] **GEMM2 tile tuning for T=512** — Profiling 显示 T=512 时 GEMM2 = 135us（2x higher than T=1024 的 60us），可能是 autotune 选择了次优配置

### 🟡 P1: 进一步 GEMM 优化

- [ ] **Persistent kernels** — 通过静态 dispatch 完全抵消 Triton run 时的 CPU 端 Python 开销
- [ ] **Phase 3 Fully Fused Kernel** — 算法已验证正确（本地 sm89），等 Triton 修复 sm100 上 `tl.dot` BLOCK_H=64 codegen 后重新测试

### 🟢 P2: 已完成的调试工具

- [x] **debug_modal.py** — 逐步对比 kernel vs reference 的中间结果
- [x] **test_scaled_mm.py** — 探测 B200 `_scaled_mm` API

### ❌ 已尝试但不 work 的优化

| 尝试 | 结果 | 原因 |
|------|------|------|
| 批量 dequant 32 experts | 0.92x 回退 | 5.3GB fp32 临时数据的带宽开销 |
| Triton GEMM (bf16 dot) | abs_err ~4096 | bf16 累积56个 K-block 丢精度 |
| 编译 routing 函数 | peak 下降 ~20% | compilation 开销 > 运行时收益 |
| `_scaled_mm` BlockWise 128 | peak 5.30x (vs 8.46x) | fp32→fp8 re-quant + padding + scale 创建的开销抵消加速 |
| FP8 Native Tensor Core Dot (Round 4) | 0/19, abs_err ~1e9 | `tl.dot(fp8,fp8)` 在早期 Triton 版本上 codegen 不正确 — **Round 5 已修复并采用（Triton 3.6.0）** |
| bf16 Dequant + bf16 Dot | 0/19, rel_err 2-10x | bf16 只有 7 bit mantissa，截断后精度不满足 tolerance |
| GPU Token Sorting (per-expert .item()) | 19/19 但 peak 7.6x→11.2x | 每 expert 一次 `.item()` sync（~60次）比一次 `.cpu().tolist()` 慢得多 |
| bf16 Intermediate | 6/19 PASSED | SwiGLU 输出需要 fp32 精度，bf16 mantissa 截断导致精度不足 |
| Phase 3 Fully Fused Kernel | 0/19, abs_err ~4096 | Triton sm100 codegen bug：`tl.dot` BLOCK_H=64 场景下的精度问题（本地 sm89 正确） |
| GEMM2 FP8 Online Quantize + Native Dot | 0/19, abs_err ~10K-26K | SwiGLU 输出动态范围大，fp8 (3bit mantissa) 量化误差在 2048 维 dot product 中级联放大 |
| GEMM2 bf16 Intermediate + bf16×bf16 Dot | 3/19, rel_err ~50-1e9 | fp32 SwiGLU→bf16 截断在 GEMM2 的 16 次 K-iteration 中逐步累积，超出 tolerance |
| GEMM2 FP8 Per-128-Block-Scale Quantize (Round 7) | 0/19, abs_err ~10K+ | 即使使用 per-128-element block scaling 适应局部动态范围，fp8 (3bit mantissa) 量化噪声仍然过大。三种变体均失败：(1) per-tensor fp8 cast → 全 NaN（SwiGLU 值 >448）; (2) PyTorch block-scale quant→dequant→fp32 GEMM2 → abs=10K+; (3) Triton block-scale quant + fp8×fp8 GEMM2 → abs=10K+。**结论：GEMM2 A-side 必须保持 fp32，无法使用 fp8×fp8 tensor cores** |
| CUDA Graph for GEMM kernels (Round 7) | 19/19 但无提升 (~9.9x avg) | CUDA Graph 捕获 GEMM1+GEMM2+zero+copy 后 replay，pre-allocated persistent buffers。**但 GEMM 只有 2 次 Triton launch（~20-50us launch overhead），占 CPU overhead 的 <8%**。瓶颈是 routing 的 ~30 次 PyTorch kernel launch（~300us）。Graph 节省的 launch overhead 被 extra `.copy_()` 开销抵消 |

---

## 注意事项

1. **Modal 环境 vs 本地环境：**
   - Modal: PyTorch 2.10.0+cu128, flashinfer-bench 0.1.2, Python 3.12
   - 本地: PyTorch 2.6.0+cu124（不同版本可能有 API 差异）

2. **flashinfer-bench API 差异：**
   - `Solution.sources` 在 Modal 上是 `list` 类型（本地是 `dict`）
   - `baseline.inputs[0]` 在 Modal 上是 `list`（本地是 `dict`）
   - 用 `pack_solution_simple.py` 打包更稳定

3. **Kernel 参数顺序：**
   - 框架调用 `kernel(*inputs, *outputs)` — 所有参数是 positional
   - `output` 必须是最后一个参数
   - `local_expert_offset` 和 `routed_scaling_factor` 是 scalar tensor（不是 Python int/float）

4. **内存管理：**
   - 32 个 expert 的权重总共 ~1.8GB（FP8），dequant 到 fp32 后每个 expert ~200MB
   - B200 有 ~180GB 显存，所以不是瓶颈
   - 但 RTX 4080 (16GB) 上 reference 实现也会 OOM
