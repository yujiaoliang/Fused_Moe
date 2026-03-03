# Fused MoE Kernel — Track A (FlashInfer AI Kernel Generation Contest @ MLSys 2026)

> **赛道:** Track A — Fused MoE  
> **目标硬件:** NVIDIA B200 (Blackwell, sm100)  
> **当前状态:** ✅ 19/19 PASSED｜最高 **8.46x** 加速｜**全部 19/19 workloads ≥ 1x**

---

## 当前进度

| 项目 | 状态 |
|------|------|
| DeepSeek-V3 routing | ✅ 完成 |
| Token sorting (expert 分组) | ✅ 完成（`moe_sort_tokens()` + expert 边界检测） |
| FP8 block-scale 反量化 | ✅ 完成（per-expert fp32 dequant） |
| SwiGLU 激活 | ✅ 完成 |
| GEMM1/GEMM2 | ✅ PyTorch fp32 matmul + torch.compile(dynamic=True)（全部 ≥1x） |
| Benchmark 正确性 | ✅ 19/19 PASSED |
| torch._scaled_mm | ✅ 已测试，B200 BlockWise 128x128 fp8（**11.68x raw**）但 re-quant 开销抵消加速 |

### B200 Benchmark 结果（最新）

| Workload | Latency | Speedup | Max Abs Err | 备注 |
|----------|---------|---------|-------------|------|
| e05c6c03 | 1.23ms | **8.46x** | 512 | 🔥 |
| b8f4f012 (T=7) | 1.87ms | **5.82x** | 256 | 🔥 |
| 2e69caee | 1.89ms | **5.73x** | 512 | 🔥 |
| 8cba5890 | 2.85ms | **4.01x** | 1024 | 🔥 |
| a7c2bcfd | 3.29ms | **3.56x** | 512 | 🔥 |
| f7d6ac7c | 4.08ms | **3.01x** | 1024 | 🔥 |
| eedc63b2 | 4.71ms | **2.68x** | 512 | |
| 5eadab1e | 4.83ms | **2.66x** | 256 | |
| 6230e838 | 5.06ms | **2.55x** | 1024 | |
| 76010cb4 | 5.59ms | **2.36x** | 1024 | |
| 81955b1e | 5.99ms | **2.25x** | 1024 | |
| fc378037 | 6.08ms | **2.23x** | 512 | |
| 74d7ff04 | 6.43ms | **2.14x** | 1024 | |
| 4822167c | 6.37ms | **2.17x** | 1024 | |
| e626d3e6 | 7.16ms | **2.00x** | 1024 | |
| 8f1ff9f1 | 7.96ms | **1.87x** | 1024 | |
| 1a4c6ba1 | 13.21ms | **1.51x** | 1024 | |
| 58a34f27 | 27.96ms | **1.25x** | 2048 | |
| 5e8dc11c | 37.05ms | **1.18x** | 1024 | |

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
│   ├── test_scaled_mm.py        # 测试 torch._scaled_mm API（fp8 matmul 探针）
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
- [x] **Expert 边界检测** — 遍历 `block_expert_ids` 直接定位 expert 边界
- [x] **fp32 accumulation** — scatter-add 用 fp32 避免 bf16 精度损失
- [x] **view+expand dequant** — 零拷贝广播替代 `repeat_interleave`
- [x] **F.silu** — 融合 CUDA kernel 替代手动 `x*sigmoid(x)`
- [x] **torch.compile** — `max-autotune-no-cudagraphs, dynamic=True` 融合 dequant+GEMM+SwiGLU，**全部 19/19 ≥1x**

### 🔴 P0: 消除 re-quantization 开销

`torch._scaled_mm` BlockWise 128 fp8 matmul 在 B200 上 **raw 快 11.68x**，但当前接入后反而慢（peak 5.30x vs compiled dequant 8.46x）：

**瓶颈分析：**
1. A 需要 fp32→fp8 re-quantization（求 amax + scale + clamp + cast）
2. Tk padding 到 128 对齐
3. `torch.full()` 创建 uniformscale tensor
4. SwiGLU 中间结果 C 也需要 fp32→fp8 re-quant

- [ ] **避免 re-quant** — 直接用原始 fp8 hidden_states + block scale 进 _scaled_mm
- [ ] **预计算 weight transpose** — 在 kernel 外层一次性转置
- [ ] **融合 SwiGLU 为 Triton kernel** — 避免 GEMM1 输出落到 HBM 再 re-quant

### 🟡 P1: 进一步优化

- [ ] **Fused SwiGLU** — 用 Triton kernel 合并进 GEMM1 epilogue
- [ ] **Persistent kernels** — 减少 kernel launch overhead
- [ ] **编译 routing** — `torch.compile` routing 函数（当前 overhead > 收益，等 warmup 抵消后可能有帮助）

### 🟢 P2: 调试工具

- [x] **debug_modal.py** — 逐步对比 kernel vs reference 的中间结果
- [x] **test_scaled_mm.py** — 探测 B200 `_scaled_mm` API
- [ ] **NCU Profiling** — 找到 B200 上的 bottleneck

### ❌ 已尝试但不 work 的优化

| 尝试 | 结果 | 原因 |
|------|------|------|
| 批量 dequant 32 experts | 0.92x 回退 | 5.3GB fp32 临时数据的带宽开销 |
| Triton GEMM (bf16 dot) | abs_err ~4096 | bf16 累积56个 K-block 丢精度 |
| 编译 routing 函数 | peak 下降 ~20% | compilation 开销 > 运行时收益 |
| `_scaled_mm` BlockWise 128 | peak 5.30x (vs 8.46x) | fp32→fp8 re-quant + padding + scale 创建的开销抵消加速 |

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
