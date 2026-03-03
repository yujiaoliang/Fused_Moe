# Fused MoE Kernel — Track A (FlashInfer AI Kernel Generation Contest @ MLSys 2026)

> **赛道:** Track A — Fused MoE  
> **目标硬件:** NVIDIA B200 (Blackwell, sm100)  
> **当前状态:** ✅ 19/19 workloads PASSED（正确性通过，性能待优化）

---

## 当前进度

| 项目 | 状态 |
|------|------|
| DeepSeek-V3 routing | ✅ 完成 |
| FP8 block-scale 反量化 | ✅ 完成（per-expert fp32 dequant） |
| SwiGLU 激活 | ✅ 完成 |
| GEMM1/GEMM2 | ⚠️ PyTorch fp32 matmul（正确但慢，~0.5x） |
| Benchmark 正确性 | ✅ 19/19 PASSED |
| 性能优化 | ❌ 待做（需要 Triton GEMM） |

### B200 Benchmark 结果

| Workload | Latency | Speedup | Max Abs Err |
|----------|---------|---------|-------------|
| b8f4f012 (T=7) | 8.92ms | **1.29x** | 0.00 |
| e05c6c03 (T=14) | 4.56ms | **2.40x** | 0.00 |
| 2e69caee | 6.95ms | **1.63x** | 16 |
| a7c2bcfd | 15.55ms | 0.80x | 128 |
| 其余 workloads | 20-69ms | 0.47-0.65x | ≤1024 |

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
2. **Dequant** — FP8 block-scale → fp32（block=128）
3. **Per-expert loop:**
   - GEMM1: `[Tk, 7168] @ [7168, 4096]` → `[Tk, 4096]`
   - SwiGLU: `silu(second_half) * first_half` → `[Tk, 2048]`
   - GEMM2: `[Tk, 2048] @ [2048, 7168]` → `[Tk, 7168]`
4. **Weighted reduce** — 按 routing weight 加权累加到 output

### 关键约束

- **Destination-passing style:** 框架调用 `kernel(*inputs, *outputs)`，output 是最后一个参数
- **正确性标准:** 95% 元素满足 `abs_err ≤ 0.01` OR `rel_err ≤ 0.01`
- **Reference 实现:** 全部 fp32 dequant + fp32 matmul（我们的实现匹配了这个策略）

---

## TODO（优化方向）

### 🔴 P0: 性能优化（大 workloads 目前 ~0.5x）

- [ ] **用 Triton GEMM 替换 PyTorch matmul**
  - 当前 per-expert 循环 + PyTorch matmul 太慢
  - 需要实现 grouped GEMM：多个 expert 共享一次 kernel launch
  - B200 (sm100) 支持 FP8 tensor core + fp32 accumulation

- [ ] **FP8 dot product + block-scale 精度**
  - `tl.dot(fp8, fp8)` 在 sm89 上精度不够（93.5% matched ratio）
  - 在 B200 sm100 上精度可能更好，需要测试
  - 方案A：`tl.dot(fp8, fp8, acc_type=tl.float32) * scales`
  - 方案B：dequant fp8 → bf16，`tl.dot(bf16, bf16)`（精度更好但更慢）

- [ ] **Token sorting + grouped GEMM**
  - `moe_sort_tokens()` 已实现（按 expert 分组 + padding）
  - 已有 `_grouped_gemm_fp8_kernel` 和 `_grouped_gemm_bf16xfp8_kernel` Triton 内核
  - 需要调通精度后重新启用

### 🟡 P1: 进一步优化

- [ ] **Fused SwiGLU** — 合并进 GEMM1 output 或 GEMM2 input
- [ ] **Shared memory 优化** — 利用 B200 的 256KB shared memory
- [ ] **Persistent kernels** — 减少 kernel launch overhead
- [ ] **Auto-tuning BLOCK_M/BLOCK_N/BLOCK_K** — 针对 B200 调优

### 🟢 P2: 工程优化

- [ ] **Torch.compile** — 尝试用 torch.compile 加速 routing 部分
- [ ] **移除 per-expert 循环** — 改为批量处理所有 active expert
- [ ] **Profile** — 用 NCU 找到 bottleneck

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
