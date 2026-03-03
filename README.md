# Fused MoE Kernel — Track A (FlashInfer AI Kernel Generation Contest @ MLSys 2026)

> **赛道:** Track A — Fused MoE  
> **目标硬件:** NVIDIA B200 (Blackwell, sm100)  
> **当前状态:** ✅ 19/19 PASSED｜最高 **4.13x** 加速｜13/19 workloads ≥ 1x

---

## 当前进度

| 项目 | 状态 |
|------|------|
| DeepSeek-V3 routing | ✅ 完成 |
| Token sorting (expert 分组) | ✅ 完成（`moe_sort_tokens()` + expert 边界检测） |
| FP8 block-scale 反量化 | ✅ 完成（per-expert fp32 dequant） |
| SwiGLU 激活 | ✅ 完成 |
| GEMM1/GEMM2 | ✅ PyTorch fp32 matmul + token sorting（~1.8x 提速） |
| Benchmark 正确性 | ✅ 19/19 PASSED |
| Triton GEMM | ⚠️ 已实现但精度不够（bf16 累积误差 ~4096），保留代码待优化 |

### B200 Benchmark 结果（最新）

| Workload | Latency | Speedup | Max Abs Err | 备注 |
|----------|---------|---------|-------------|------|
| e05c6c03 | 2.69ms | **4.13x** | 0 | 🔥 |
| 2e69caee | 3.81ms | **3.02x** | 0 | 🔥 |
| b8f4f012 (T=7) | 4.69ms | **2.49x** | 0 | 🔥 |
| 8cba5890 | 7.24ms | **1.72x** | 0 | |
| a7c2bcfd | 7.94ms | **1.59x** | 2 | |
| f7d6ac7c | 10.02ms | **1.32x** | 512 | |
| 5eadab1e | 10.96ms | **1.26x** | 512 | |
| eedc63b2 | 11.21ms | **1.21x** | 32 | |
| 6230e838 | 12.54ms | **1.11x** | 64 | |
| 76010cb4 | 13.30ms | **1.07x** | 1 | |
| 81955b1e | 13.99ms | **1.03x** | 8 | |
| fc378037 | 14.13ms | **1.03x** | 8 | |
| 74d7ff04 | 14.77ms | **1.00x** | 1 | |
| 4822167c | 15.02ms | **1.00x** | 32 | |
| e626d3e6 | 15.48ms | 0.99x | 64 | |
| 5e8dc11c | 47.47ms | 0.95x | 1024 | |
| 8f1ff9f1 | 16.82ms | 0.94x | 128 | |
| 58a34f27 | 38.11ms | 0.94x | 1024 | |
| 1a4c6ba1 | 23.58ms | 0.89x | 512 | |

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
├── test_kernel.py               # 本地快速测试（直接对比 reference）
├── scripts/
│   ├── test_modal.py            # Modal B200 benchmark（推荐）
│   ├── debug_modal.py           # Modal B200 调试（逐步对比 reference）
│   ├── run_modal.py             # Modal B200 完整 benchmark
│   ├── run_local.py             # 本地 benchmark
│   ├── pack_solution_simple.py  # 打包 solution.json
│   └── pack_solution.py         # 打包（需要 flashinfer_bench.agents）
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
3. **Dequant** — FP8 block-scale → fp32（block=128）
4. **Per-expert loop**（通过 `block_expert_ids` 边界检测，无需重复遍历）:
   - GEMM1: `[Tk, 7168] @ [7168, 4096]` → `[Tk, 4096]`
   - SwiGLU: `silu(second_half) * first_half` → `[Tk, 2048]`
   - GEMM2: `[Tk, 2048] @ [2048, 7168]` → `[Tk, 7168]`
5. **Weighted reduce** — fp32 `index_add_` 加权累加 → bf16 output

### 关键约束

- **Destination-passing style:** 框架调用 `kernel(*inputs, *outputs)`，output 是最后一个参数
- **正确性标准:** 95% 元素满足 `abs_err ≤ 0.01` OR `rel_err ≤ 0.01`
- **Reference 实现:** 全部 fp32 dequant + fp32 matmul（我们的实现匹配了这个策略）

---

## TODO（优化方向）

### ✅ 已完成

- [x] **Token sorting** — `moe_sort_tokens()` 按 expert 分组 + BLOCK_M padding
- [x] **Expert 边界检测** — 遍历 `block_expert_ids` 直接定位 expert 边界，~1.8x 加速
- [x] **fp32 accumulation** — scatter-add 用 fp32 避免 bf16 精度损失

### 🔴 P0: Triton GEMM 精度修复（替换 PyTorch matmul）

已有 Triton kernel 代码（`_grouped_gemm_fp8_kernel`, `_grouped_gemm_bf16xfp8_kernel`），但精度不够：

- [ ] **`tl.dot(bf16, bf16)` 累积误差** — 56 个 K-block 的 bf16 乘积累积导致 abs_err ~4096
  - 已测试方案（均失败）：
    - `tl.dot(fp8, fp8) * a_scale * b_scale` — 同样 ~4096 误差
    - 先 dequant fp8→bf16 再 `tl.dot(bf16, bf16)` — 同样 ~4096 误差
  - 待测试方案：
    - [ ] `tl.dot(fp32, fp32)` — 不用 tensor core 但精度好
    - [ ] 分段累积：每 N 个 K-block 做一次 fp32 归约
    - [ ] B200 sm100 可能有更高精度的 tensor core 配置

### 🟡 P1: 进一步优化

- [ ] **Fused SwiGLU** — 用 Triton kernel 合并进 GEMM1 epilogue
- [ ] **批量 dequant** — 一次 dequant 所有 active expert 的权重，减少重复开销
- [ ] **torch.compile** — 加速 routing + per-expert matmul
- [ ] **Persistent kernels** — 减少 kernel launch overhead

### 🟢 P2: 调试工具

- [x] **debug_modal.py** — 逐步对比 kernel vs reference 的中间结果
- [ ] **NCU Profiling** — 找到 B200 上的 bottleneck

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
