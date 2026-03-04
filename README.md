# Fused MoE Kernel — Track A (FlashInfer AI Kernel Generation Contest @ MLSys 2026)

> **赛道:** Track A — Fused MoE
> **目标硬件:** NVIDIA B200 (Blackwell, sm100)
> **当前状态:** ✅ 19/19 PASSED｜最高 **11.91x** 加速｜large-T **7.1-7.5x**｜平均 **~9.6x**

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

Round 5 优化：GEMM1 中使用 FP8 原生 Tensor Core dot（`tl.dot(fp8, fp8)` + 后乘 scale），Triton 3.6.0 已修复 sm100 codegen。2x tensor core 吞吐，消除 3 次 dequant 乘法/K-iter。Large-T 从 ~4x 跃升至 ~7x，平均从 ~7.7x 提升至 ~9.6x。

| Workload | Round 1 | Round 4 | Round 5 | 备注 |
|----------|---------|---------|---------|------|
| e05c6c03 (T=14) | 12.90x | 11.45x | **11.91x** | 🔥 peak |
| 2e69caee | 12.21x | 9.53x | **11.54x** | +2.0x |
| 1a4c6ba1 | 1.35x | 8.47x | **10.80x** | +2.3x |
| b8f4f012 (T=7) | 12.02x | 10.00x | **10.93x** | |
| 8cba5890 | 10.14x | 9.66x | **10.35x** | |
| a7c2bcfd (T=128)| 10.00x | 8.87x | **10.00x** | |
| 5eadab1e | 10.06x | 8.22x | **9.98x** | |
| 6230e838 | 8.97x | 7.16x | **9.88x** | +2.7x |
| f7d6ac7c | 9.60x | 8.54x | **9.85x** | |
| 81955b1e | 8.94x | 7.18x | **9.71x** | |
| 74d7ff04 | 9.02x | 7.35x | **9.68x** | |
| 4822167c | 8.75x | 7.35x | **9.68x** | |
| fc378037 | 9.02x | 7.58x | **9.52x** | |
| 76010cb4 | 8.76x | 7.31x | **9.39x** | |
| e626d3e6 | 9.18x | 7.49x | **9.35x** | |
| eedc63b2 | 9.30x | 7.38x | **9.32x** | |
| 8f1ff9f1 | 8.80x | 7.10x | **8.95x** | |
| 58a34f27 (T=4096)| 4.14x | 4.45x | **7.46x** | 🔥 large-T +3.0x |
| 5e8dc11c (T=4096)| 3.69x | 3.99x | **7.08x** | 🔥 large-T +3.1x |

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
├── solution/triton/kernel.py    # ← 主要编辑的文件（Triton kernel）
├── check_modal.py               # Modal B200 快速正确性检查
├── test_kernel.py               # 本地快速测试（直接对比 reference）
├── scripts/
│   ├── test_modal.py            # Modal B200 benchmark（推荐）
│   ├── run_modal.py             # Modal B200 完整 benchmark
│   ├── profile_modal.py         # Modal B200 NCU profiling
│   ├── debug_modal.py           # Modal B200 调试（逐步对比 reference）
│   ├── pack_solution_simple.py  # 打包 solution.json
│   ├── pack_solution.py         # 打包（需要 flashinfer_bench.agents）
│   ├── run_local.py             # 本地 benchmark
│   └── test_scaled_mm.py        # 测试 torch._scaled_mm API
├── config.toml                  # 配置（队名、赛道等）
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
   - Native FP32 内积与 Routing Weights 相乘
   - 使用 Triton `tl.atomic_add` 直接将更新写入最终的 fp32 buffer，避免低精度 rounding error，随后复制至 `bfloat16` output。

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

### 🟡 P1: 进一步优化

- [ ] **Persistent kernels** — 通过静态 dispatch 完全抵消 Triton run 时的 CPU 端 Python 开销
- [ ] **NCU Profiling** — 分析 B200 上的 shared memory access 和 instruction latency bottleneck
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
