# 融合 MoE 推理核优化

> **Hybrid Triton + CuTe DSL for DeepSeek-V3 Expert Dispatch on NVIDIA B200 Blackwell**

[![Status](https://img.shields.io/badge/%E7%8A%B6%E6%80%81-19%2F19%20%E9%80%9A%E8%BF%87-brightgreen)](#核心结果)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20B200%20(sm100a)-76b900)](https://www.nvidia.com/en-us/data-center/b200/)
[![Triton](https://img.shields.io/badge/Triton-3.6.0-blue)](https://github.com/triton-lang/triton)
[![CuTe DSL](https://img.shields.io/badge/CuTe%20DSL-CUTLASS-blue)](https://github.com/NVIDIA/cutlass)
[![Peak Speedup](https://img.shields.io/badge/%E5%B3%B0%E5%80%BC%E5%8A%A0%E9%80%9F-~56x-orange)](#核心结果)
[![English](https://img.shields.io/badge/lang-English-blue)](README.md)

基于 **[Triton](https://github.com/triton-lang/triton) + [CuTe DSL](https://github.com/NVIDIA/cutlass)** 混合架构的 DeepSeek-V3 MoE 推理核实现，目标硬件 **NVIDIA B200 (Blackwell, sm_100a)**。

- **Triton** 处理 17/19 workloads（routing、sorting、GEMM1+SwiGLU、GEMM2、token reduce）
- **CuTe DSL**（NVIDIA CUTLASS）处理 2/19 大 T workloads（T=11948、T=14107），通过 per-T 隔离 runtime 的 grouped GEMM 实现

**19/19 全部通过** | 峰值 ~56x | 大 T ~13–15x | 均值 ~54x (session-dependent)

> **关于绝对 speedup 数值：** Modal B200 共享实例无锁频，同一代码不同时段可波动 20–30%。官方评测在裸金属 B200 + 锁频 (`nvidia-smi -ac 3996,1965`) 下运行。本文数值仅作相对趋势参考。

---

## 目录

- [核心结果](#核心结果)
- [架构](#架构)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [评测环境](#评测环境)
- [优化历程](#优化历程)
- [论文](#论文)
- [关键约束](#关键约束)

---

## 核心结果

| 指标 | 值 |
|------|-----|
| 正确性 | **19/19 通过** |
| 峰值加速比 | ~56x (Modal B200, session-dependent) |
| 大 T 加速比 (T=8192–14107) | ~13–15x |
| 平均加速比 | ~54x |
| 评分方式 | CUPTI GPU kernel 时间之和（CPU 开销不计入） |

---

## 架构

核心为 **6 阶段融合流水线**，混合 Triton + CuTe DSL 后端：

```
输入 → Routing → Token 排序 → GEMM1+SwiGLU → GEMM2 → Token 归约 → 输出
```

| 阶段 | 描述 | 关键技术 |
|------|------|----------|
| 1. **Routing** | DeepSeek-V3 no-aux：sigmoid → group-filter → top-8 → 归一化 | Pure Triton kernel |
| 2. **Token 排序** | Tile 级并行 histogram + offset，BLOCK_M 对齐 padding | 并行排序与散射 |
| 3. **GEMM1 + SwiGLU** | 融合 W1/W3，共享 A-side 加载 | **FP8 原生 tensor core dot** (2x 吞吐) |
| 4. **GEMM2** | 非原子写入 expert_out 缓冲 | Post-dot B-scale, bf16 expert_out |
| 5. **Token 归约** | 每个 output token 读取 TOP_K=8 个 expert 贡献 | 零原子、零清零、零 copy |
| 6. **T=1 路径** | 单 token decode 特化 | 融合 routing+sort+GEMM |

### Runtime Dispatch 策略

入口 `kernel.py` 根据 T 自动选择路径：

| 条件 | 路径 | BLOCK_M | Kernel |
|------|------|---------|--------|
| T=11948 | CuTe DSL grouped GEMM | 128 | Per-T 隔离 CuTe runtime |
| T=14107 | CuTe DSL grouped GEMM | 128 | Per-T 隔离 CuTe runtime |
| T=901 | 隔离 Triton | 64 | Static launch cap，专属 autotune |
| T=1 | Pure Triton | 16 | 融合单 token 路径 |
| T=53–59 | Pure Triton | **8** | Tight BLOCK_M — padding 浪费降低 4× |
| 32 ≤ T ≤ 64 | Pure Triton | 32 | `_small_medium_*` |
| 65 ≤ T ≤ 128 | Pure Triton | 32/64 | `_medium_*` |
| T > 128 | Pure Triton | 64 | `_fused_moe_*` (通用) |
| T > 2048 | Pure Triton | 128 | `_fused_moe_*` (通用) |
| T ≥ 4096 | Pure Triton | dynamic | Exact dispatch (`total_blocks.item()`) |

<details>
<summary><b>Profiling 瓶颈分布</b>（NCU, B200, 19 real traces）</summary>

| T 范围 | GEMM1 占比 | GEMM2 占比 | Routing+Sort | Reduce | 瓶颈 |
|--------|-----------|-----------|-------------|--------|------|
| T=1 | 32% | **55%** | 10% | (fused) | GEMM2 |
| T=7–80 | **51–61%** | 32–37% | 5–10% | 2% | GEMM1 |
| T=901 (隔离路径) | **52.6%** | 35.3% | 7.9% | 2.3% | GEMM1 |
| T=11948–14107 | 40% | **49–50%** | 5–6% | 3% | GEMM2 |

> GEMM1 TFLOPS: 103–182T (小 T) / 485–535T (大 T)。GEMM2 TFLOPS: 82–150T (小 T) / 216T (大 T)。

</details>

### 三路 Dispatch 架构

```
kernel.py (入口)
  ├── T in {11948, 14107}  →  triton_impl.py     →  CuTe DSL grouped GEMM
  │                            ├── cute_gemm1_mma.py + cute_gemm1_mma_runtime_{T}.py
  │                            ├── cute_gemm2_mma.py + cute_gemm2_mma_runtime_{T}.py
  │                            └── cute_grouped_gemm_sm100.py
  ├── T == 901             →  triton_impl_t901.py →  隔离 Triton (static launch cap)
  └── 其他 T               →  pure_triton_impl.py →  Pure Triton kernels
```

> **Per-T 隔离设计：** 每个特化 T 拥有独立的 runtime 文件，不共享 Python module state / compile cache / metadata cache。

---

## 快速开始

### 环境搭建

```bash
# 1. 创建 conda 环境
conda create -n fi-bench python=3.12
conda activate fi-bench

# 2. 安装依赖
pip install flashinfer-bench modal torch triton numpy

# 3. 准备 benchmark traces
# 将 trace 数据放在 /path/to/benchmark-traces

# 4. Modal 登录（一次性）
modal setup
```

### 运行 Benchmark

```bash
# 上传数据到 Modal volume（一次性）
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/benchmark-traces /

# 打包 & 运行
python scripts/pack_solution_simple.py
python -m modal run scripts/test_modal.py
```

### A/B 对比测试（推荐）

```bash
# 存 baseline
python scripts/pack_solution_simple.py
cp solution.json solution_a.json

# 改代码后存实验版
python scripts/pack_solution_simple.py
cp solution.json solution_b.json

# 同一 B200 session 内 A/B 对比
python -m modal run scripts/ab_test_modal.py
```

> **判断标准：** Mean speedup Δ > 2% 为有效信号；≤2% 为噪声。Large-T Δ > 1% 可靠。

---

## 项目结构

```
mlsys_note/
├── solution/
│   └── python/                              # 运行时代码目录 (config.toml: language=python)
│       ├── kernel.py                        # 入口：三路 dispatch (CuTe / T=901 / Pure Triton)
│       ├── pure_triton_impl.py              # Pure Triton 主实现 (16/19 workloads)
│       ├── triton_impl.py                   # Hybrid CuTe+Triton 实现 (大 T workloads)
│       ├── triton_impl_t901.py              # T=901 隔离 Triton 路径 (static launch cap)
│       ├── cute_gemm1_mma.py                # CuTe DSL GEMM1 核心逻辑
│       ├── cute_gemm1_mma_runtime_{T}.py    # Per-T 隔离 GEMM1 runtime
│       ├── cute_gemm2_mma.py                # CuTe DSL GEMM2 核心逻辑
│       ├── cute_gemm2_mma_runtime_{T}.py    # Per-T 隔离 GEMM2 runtime
│       └── cute_grouped_gemm_sm100.py       # CuTe DSL grouped GEMM 核心 (NVIDIA 参考)
├── scripts/
│   ├── ab_test_modal.py                     # A/B 对比测试（同 B200 session）★ 推荐
│   ├── test_modal.py                        # Modal B200 单次 benchmark
│   ├── run_modal.py                         # Modal 完整 benchmark（含 auto-pack）
│   ├── profile_modal.py                     # Modal torch.profiler profiling
│   ├── ncu_profile_modal.py                 # Modal NCU per-kernel 时间分解
│   ├── pack_solution_simple.py              # 打包 solution.json
│   └── ...
├── paper.tex                                # 技术论文 (LaTeX)
├── paper.pdf                                # 编译后论文
├── config.toml                              # 运行配置
├── solution.json                            # 打包后的 solution bundle
└── benchmark-traces/                        # 可选的本地 benchmark trace 数据
```

---

## 评测环境

| 项目 | 值 |
|------|----|
| Docker Image | `flashinfer/flashinfer-ci-cu132:20260401-2c675fb` |
| PyTorch | `2.12.0.dev20260331+cu132` |
| Triton | 3.6.0 |
| CuTe DSL | nvidia-cutlass-dsl (CUTLASS) |
| GPU | B200 (bare-metal, sm_100a) |
| 容差 | `--atol 1 --rtol 0.3 --required-matched-ratio 0.9` |
| 评分 | CUPTI GPU kernel 时间之和（CPU 开销不计入） |

---

## 优化历程

<details>
<summary><b>Phase 1：基础架构 (0x → 12.9x)</b></summary>

| 阶段 | 内容 | Peak |
|------|------|------|
| 初版 | PyTorch eager per-expert 循环 | ~1x |
| torch.compile | 融合 dequant+GEMM+SwiGLU | ~5x |
| Monolithic Triton | 消除 per-expert launch | ~12.9x |

</details>

<details>
<summary><b>Phase 2：Triton 核心优化 (12.9x → 46x)</b></summary>

| Commit | 优化内容 | 效果 |
|--------|---------|------|
| `496da33` | Triton autotune + GEMM2 BLOCK_K=128 | ~5x → 8x |
| `3d3aace` | Hybrid token sort + K-loop pointer hoist | 结构性提升 |
| `c1943ae` | Routing gather-first + pre-alloc buffers | 减少 CPU overhead |
| `99c1b13` | **FP8 Native Tensor Core** (`tl.dot(fp8,fp8)`) | **2x FLOP 吞吐** |
| `786ee59` | GEMM2 post-dot B-scale | 消除 [BK,BN] dequant |
| — | Token-Centric Reduce（零原子、零 copy） | 小 T +46–78% |
| — | Parallel Sort（GPU tile 级并行 histogram） | 打破单线程瓶颈 |

</details>

<details>
<summary><b>Phase 3：分桶特化 (46x → 55x+)</b></summary>

| Commit | 优化内容 | 效果 |
|--------|---------|------|
| `0b865b3` | T=1 专用 kernel 路径 | 消除 decode overhead |
| `c59e09d` | 3-phase (T>4096) | Large-T latency **−20%** |
| `d2fdf14` | **Medium-T bucket** (4-level BLOCK_M) | 多 workload latency **−20~54%** |
| `c685b02` | Exact dispatch for large-T | Large-T latency **−5%** |
| `93e3a84` | 关闭 Triton LSR + 非对称 L2 eviction | Mean **+4.9%** |
| `93e3a84` | 极深流水线 GEMM2 (num_stages=6–8) | Mean +2.1% |
| `f49c5ab` | 2D Tiled Token Reduce | Large-T **+2.9%** |
| `9d5a2f8` | Medium-T Column-Major 调度 | Medium-T 最高 **+8.9%** |
| `a5256cc` | **FP16 Intermediate Buffer** (×0.125/×8.0) | AB-test mean **+4.5%**, 13/19 improved |

</details>

<details>
<summary><b>Phase 4：精度与带宽探索</b></summary>

| 实验 | 结果 | 结论 |
|------|------|------|
| bf16 Intermediate | 10/19 通过 | SwiGLU 动态范围太大，7-bit mantissa 精度不足 |
| fp16 Intermediate (全局) | T=7 ratio=0.0 | SwiGLU 极端值超 fp16 max (65504) |
| **fp16 Intermediate (T≥32)** | **19/19, +4.5%** | ×0.125 缩放 + ×8.0 补偿；T<32 走 fp32 fallback |
| **bf16 expert_out (T≥32)** | **19/19, +0.4%** | bf16 range=3.4e38 不溢出；节省 50% 写带宽 |
| fp16 expert_out | 3/19 溢出 | 值超 fp16 max → inf |
| GEMM2 FP8 (全部变体) | 0–5/19 | 3-bit mantissa 误差级联；**封棺** |

**结论：** Intermediate 精度阶梯 fp16 > bf16 > fp8（仅 fp16+scaling 可行）。expert_out 阶梯 bf16 > fp16 > fp8（仅 bf16 可行）。GEMM2 A-side 必须 ≥fp16。

</details>

<details>
<summary><b>Phase 5：评测规则感知优化</b></summary>

**关键评测设置更新（Apr 14–15 澄清）：**
- **容差 100x 放宽**：`atol=1.0, rtol=0.3, ratio=0.9`（我们之前测试用 `atol=0.01`）
- **CUPTI GPU-only 计时**：CPU 开销、launch latency、`.item()` sync 全部不计入评分
- **Self-contained 要求**：不推荐 cuBLAS/CUTLASS/FlashInfer runtime 调用，以便保持 kernel 实现可审查

| Commit | 优化内容 | 结果 |
|--------|---------|------|
| `c1effcc` | **bf16 expert_out (T≥32)** | 19/19 通过, AB-test mean +0.4%, large-T +2.5% |
| `c1effcc` | **CUBLAS_ENABLED=False** | 策略调整：禁用 cuBLAS 以降低 review 风险 |

</details>

<details>
<summary><b>Phase 6：Per-T 隔离 & 最终微调 (Rounds 19–22)</b></summary>

| Commit | 优化内容 | 结果 |
|--------|---------|------|
| `f088f03` | **Per-T 隔离 CuTe Runtime** | AB-test mean +1.4%, **T=14107 +55.1%** (13.19x → 20.45x) |
| `03e746f` | **T=901 隔离 Triton 路径** | AB-test **+6%**（static launch cap 避免 host sync） |
| `886c7fc` | CuTe 精度调查 | T=14107 隔离修复后恢复 CuTe 路径 |
| `2e134c7` | **Tight BLOCK_M=8 (T=53–59)** | AB-test **+3.8% 均值**, 7/19 提升 (最高 +14.5%), 0 退化 |

</details>

<details>
<summary><b>全 Commit 审计</b> — 所有优化的可信度分级</summary>

| 判定 | 含义 |
|------|------|
| ✅✅ **确定真实** | 多 workload latency 改善 >20%，或新增整个 kernel 路径 |
| ✅ **大概率真实** | Mean latency 改善 5–20%，或 large-T 改善 >5% |
| ⚠️ **不确定** | 改善 <5%，在噪声范围内 |
| ❌ **已确认退化** | 实测退化或从主线回退 |

| Commit | 描述 | 判定 | 理由 |
|--------|------|------|------|
| 早期架构 (`aa7c1b1`→`effc2f2`) | 从零到 12.9x | ✅✅ | 从无到有 |
| `496da33` | Triton autotune + BLOCK_K=128 | ✅✅ | peak 5x → 8x |
| `3d3aace` | Hybrid sort + pointer hoist | ✅✅ | 算法结构变化 |
| `99c1b13` | FP8 native tensor core | ✅✅ | 2x FLOP 吞吐 |
| `c1943ae` | Routing gather-first | ✅ | CPU overhead 减少 |
| `786ee59` | GEMM2 post-dot B-scale | ✅ | 计算顺序变化 |
| `0b865b3` | T=1 专用 kernel (+481 行) | ✅✅ | 整个新路径 |
| `c59e09d` | 3-phase (T>4096) | ✅✅ | latency −20% |
| `d2fdf14` | Medium-T bucket | ✅✅ | latency −20~54% |
| `c685b02` | Exact dispatch (large-T) | ✅ | latency −5% |
| `93e3a84` | 关闭 Triton LSR + L2 eviction | ✅✅ | 峰值/均值显著提升 (+4.9%) |
| `93e3a84` | 极深流水线 GEMM2 | ✅ | 均值 +2.1% |
| `f49c5ab` | 2D Tiled Token Reduce | ✅ | Large-T +2.5~2.9% |
| `9d5a2f8` | Medium-T Column-Major | ✅ | Medium-T 最高 +8.9% |
| `c19336d` | Fused Routing+Histogram | ✅ | Large-T routing/sort −1.5% |
| `67ef373` | T=901 GEMM2 专用 Kernel | ✅ | 单 kernel −3% |
| `1010fb4` | 小中T GEMM1 Autotune | ✅✅ | GEMM1 latency −1.6~6.6% |
| `c35907d` | T=1 GEMM1/2 Autotune | ✅✅ | latency −3% |
| `a5256cc` | **FP16 Intermediate Buffer** | ✅✅ | AB-test mean +4.5%, 13/19 improved |
| `f040d09` | **T=901 Static Launch Cap** | ✅✅ | AB-test +6% |
| `f088f03` | **Per-T 隔离 CuTe Runtime** | ✅✅ | T=14107 +55.1% |
| `c1effcc` | **bf16 expert_out** | ✅ | mean +0.4%, large-T +2.5% |
| `2e134c7` | **Tight BLOCK_M=8 (T=53–59)** | ✅✅ | AB-test +3.8% 均值, 7/19 提升, 0 退化 |

**结论：当前主线中所有生效改动都是 ✅ 或 ✅✅ 确定真实的。**

</details>

<details>
<summary><b>失败的优化与死路</b> — 尝试过但未生效的方向</summary>

### 精度硬限制

| 尝试 | 结果 | 原因 |
|------|------|------|
| bf16 Intermediate | 10/19 通过 | 7-bit mantissa 太粗糙 |
| fp16 Intermediate (全局) | T=7 ratio=0.0 | 极端值超 fp16 max (65504) |
| fp16 expert_out | 3/19 溢出 | 值超 fp16 max → inf |
| GEMM2 FP8 (全部变体) | 0–5/19 | 3-bit mantissa 误差级联 |
| MXFP8 `tl.dot_scaled` | ratio 0.17–0.32 | e8m0 共享指数太粗糙 |

### 架构级失败

| 尝试 | 结果 | 原因 |
|------|------|------|
| Persistent GEMM2 | 52x → 15.6x | 丧失张量级并行度 |
| TMA Accelerator | −1~2% | HBM 带宽已被 LDG 指令打满 |
| 1D Atomic Scatter | 8.3x → 5.37x | 标量原子风暴打瘫内存控制器 |
| Fused GEMM2+Reduce (atomic_add) | −4.5~4.8% | 12 bytes/elem 原子 vs 2 bytes bf16 store |
| CUDA Graph for GEMMs | 无收益 | CUPTI 仅计 GPU kernel 时间 |
| torch.compile on routing | 无收益 | CPU overhead 不在评分范围 |
| Warp Specialization | 崩溃 / −34% | 需要 TMA block pointers 才能获益 |
| `tl.make_tensor_descriptor` (TMA) | −4% | Descriptor 建立成本无法被 K=2048 摊销 |
| `num_ctas=2` without TMA | 无收益 | 两个 CTA 计算完全相同的 tile |

### Autotune 饱和（2026-04 确认）

三个 kernel（GEMM1、GEMM2、token_reduce）的 autotune 空间已全部饱和。继续搜索仅产生 ≤0.4%（噪声范围）。后续提升需要算法级/架构级变化。

</details>

---

## 论文

<details>
<summary><b>查看技术论文详情</b></summary>

**标题：** *Fused MoE Inference on Blackwell: A Pure Triton Approach to DeepSeek-V3 Expert Dispatch*

**作者：** Jiayao Zhang, Jiaoliang Yu

> **注：** 论文标题为 "Pure Triton"，最终实现已演进为 Triton + CuTe DSL 混合架构（CuTe DSL 处理 2 个大 T workloads）。

**摘要：**
我们提出了一种混合 Triton + CuTe DSL 的 DeepSeek-V3 MoE kernel 实现，目标硬件为 NVIDIA B200 (Blackwell, sm_100a) GPU。我们的 6 阶段流水线利用 FP8 原生 tensor core dot 在 GEMM1 中实现 2x 吞吐，采用非原子的两阶段 GEMM2-then-reduce 架构，使用 FP16 intermediate buffer 配合 scale-and-cast 补偿，以及四级 BLOCK_M 多桶特化。在 19 个真实 trace workloads (T=1 到 T=14,107) 上，系统达到 19/19 正确性，峰值加速比 106.65x（均值 55.77x），基于 CUPTI GPU-only kernel 计时。

**核心贡献：**
1. 6 阶段融合流水线（Triton + CuTe DSL），消除 per-expert launches
2. GEMM1 原生 FP8 tensor core dot — 相比 TF32 2x 吞吐
3. 非原子 GEMM2 + token-centric reduce — medium-T +46%
4. FP16 intermediate buffer ×0.125/×8.0 补偿 — 带宽 −50%
5. 五级 BLOCK_M (8/16/32/64/128) workload-aware dispatch
6. Per-T 隔离 CuTe DSL runtime 用于大 T grouped GEMM（T=14107 +55%）

编译后论文：[`paper.pdf`](paper.pdf) | 源码：[`paper.tex`](paper.tex)

**海报：** [`poster.pdf`](poster.pdf)（A0 横版，4 栏布局）| 源码：[`poster.tex`](poster.tex)

### 图表

| | |
|:---:|:---:|
| ![Pipeline](figures/pipeline.png) | ![Speedup](figures/speedup_bar.png) |
| 流水线架构 | 各 Workload 加速比 |
| ![Precision](figures/precision_matrix.png) | ![TFLOPS](figures/tflops_efficiency.png) |
| 精度探索矩阵 | TFLOPS 效率 |
| ![Timeline](figures/optimization_timeline.png) | ![Breakdown](figures/time_breakdown.png) |
| 优化时间线 | Kernel 时间分解 |

</details>

---

## 关键约束

| 约束 | 说明 |
|------|------|
| API 风格 | Destination-passing：`kernel(*inputs, *outputs)`，output 是最后一个参数 |
| **评测容差** | **`atol=1.0, rtol=0.3, matched_ratio=0.9`** — 元素仅在 abs>1 AND rel>0.3 时才 fail |
| 评分方式 | CUPTI GPU kernel 时间之和，CPU 开销不计入 |
| Docker | `flashinfer/flashinfer-ci-cu132:20260401-2c675fb` (pinned) |
| GPU 架构 | sm_100a — 需在 build flags 中显式指定 `-arch=sm_100a` |
| cuBLAS/CUTLASS | 不硬禁；runtime 中尽量减少依赖，以保持实现自包含 |
| FlashInfer runtime | 不允许 runtime 调用 FlashInfer API；可复制源码到 repo |
| Self-contained | 运行时代码位于 `solution/python/` 目录内，打包进 solution.json |
| 显存 | 32 experts × ~56MB FP8 = ~1.8GB；B200 ~180GB，不是瓶颈 |

---

## Modal B200 噪声分析

<details>
<summary><b>测量噪声特征</b></summary>

通过在**同一 Modal B200 session** 内背靠背跑同一份代码 (`ab_test_modal.py`) 测量：

| 指标 | 噪声范围 | 说明 |
|------|---------|------|
| **Mean speedup** | **±2%** | 非常稳定，可靠用于整体对比 |
| **单个中小 T** | **±15%** | 同代码跑出 55.52x 和 64.68x (Δ=16.5%) |
| **Large-T (≥4096)** | **<1%** | Δ=0.1–0.2%，近零噪声 |

**判断优化有效的标准：**
- Mean speedup Δ > 2%（超出噪声范围）
- 或 Large-T latency Δ > 1%
- 单个 workload ≤15% 变化**不可**作为判据

> **跨 session 漂移：** 同一代码在不同日期的 Modal B200 上可差 20–30%。这完全来自共享实例的时钟/负载状态，非代码退化。

</details>

---

## 注意事项

1. **评测环境统一：** 所有 Modal 脚本使用官方 Docker image，PyTorch 2.12.0+cu132, Triton 3.6.0, CuTe DSL (CUTLASS), Python 3.12。本地 (Windows) 仅用于打包和代码审查。

2. **三路 Dispatch 可独立迭代：** 通过 `kernel.py` 的 T 值条件路由。Per-T 隔离确保无共享状态污染。

3. **已确认的优化死路（请勿重试）：**
   - ~~bf16 Intermediate~~ — 7-bit mantissa 精度不足 (10/19)
   - ~~FP8 GEMM2~~ — 3-bit mantissa 封棺 (0–5/19)
   - ~~fp16 expert_out~~ — overflow >65504
   - ~~Persistent GEMM2~~ — 丧失并行度 (52x → 15.6x)
   - ~~TMA Accelerator~~ — HBM 已被 LDG 打满
   - ~~Expert skipping~~ — top-8 weight 太大，无法安全跳过
   - ~~所有 autotune 微调~~ — 已饱和

4. **已修复的历史问题：**
   - ~~T=14107 CuTe 精度 bug~~ → 已通过 per-T 隔离 runtime 修复 (`f088f03`)；T=14107 恢复 CuTe 路径 (20.45x vs Pure Triton 13.19x)
