# Fused MoE Kernel — Track A (MLSys 2026 FlashInfer Contest)

> **赛道:** Track A — Fused MoE  
> **硬件:** NVIDIA B200 (Blackwell, sm100)  
> **状态:** ✅ 19/19 PASSED  
> **当前可复现口径 (Modal B200):** peak ~55-65x, mean ~43-45x  
> **历史最佳 (Round 15, Modal 好运时段):** peak 106.65x, mean 55.77x

> ⚠️ **关于绝对 speedup 数值**：Modal B200 共享实例无锁频，同一代码不同时段的 speedup 可波动 20-30%。
> 官方评测在裸金属 B200 + 锁频 (`nvidia-smi -ac 3996,1965`) 下运行，结果更稳定。
> 本文中的 speedup 数值仅作相对趋势参考，不应作为绝对性能指标。

---

## 目录

1. [Kernel 架构](#kernel-架构)
2. [环境搭建](#环境搭建)
3. [运行与测试](#运行与测试)
4. [项目结构](#项目结构)
5. [Modal B200 噪声分析](#modal-b200-噪声分析)
6. [优化历程](#优化历程)
7. [全 Commit 审计](#全-commit-审计)
8. [已尝试但未生效的优化](#已尝试但未生效的优化)
9. [注意事项](#注意事项)

---

## Kernel 架构

```
kernel(routing_logits, routing_bias, hidden_states, hidden_states_scale,
       gemm1_weights, gemm1_weights_scale, gemm2_weights, gemm2_weights_scale,
       local_expert_offset, routed_scaling_factor, output)
```

### 计算流程（6 阶段）

1. **Routing (Pure Triton)** — `triton_ds_routing_kernel`  
   DeepSeek-V3 no-aux routing：sigmoid → group-filter → top-8 → normalized weights

2. **Token Sorting (Pure Triton)** — `parallel_sort_and_scatter`  
   Tile 级并行 histogram + offset 计算，按 expert 分组并 padding 到 BLOCK_M 对齐

3. **GEMM1 + SwiGLU (Fused Triton)**  
   按 batch 大小 dispatch 到 `_small_medium_*` / `_medium_*` / `_fused_moe_gemm1_swiglu_*` kernel  
   - FP8 Native Tensor Core Dot (`tl.dot(fp8, fp8)`) + post-dot scale，2x 吞吐 vs TF32  
   - 在同一 K-loop 中同时计算 W1 和 W3（共享 A 加载）  
   - Epilogue 融合 SwiGLU → 输出 fp32 `Intermediate [num_padded, 2048]`

4. **GEMM2 (Non-Atomic Triton)**  
   按 batch 大小 dispatch 到 `_small_medium_*` / `_medium_*` / `_fused_moe_gemm2_*` kernel  
   - Post-dot B-scale：`tl.dot(a, b.to(f32))` + `acc += partial * b_scale`  
   - 非原子写入 `expert_out [num_padded, 7168]`

5. **Token-Centric Reduce** — `_token_reduce_kernel`  
   每个 output token 启动一个程序，通过 `scatter_map` 读取 TOP_K=8 个 expert 贡献  
   fp32 求和后直接写入 bf16 output。**零原子、零清零、零 copy**

6. **T=1 专用路径** — `_kernel_t1`  
   单 token decode 特化：融合 routing+sort+GEMM 消除通用路径 overhead

### Runtime Dispatch 策略

| 条件 | BLOCK_M | GEMM Kernel |
|------|---------|-------------|
| T=1 | 16 | `_t1_*` 专用路径 |
| 32 ≤ T ≤ 64 | 32 | `_small_medium_*` |
| 65 ≤ T ≤ 128 | 32/64 | `_medium_*` |
| T > 128 | 64 | `_fused_moe_*` (generic) |
| T > 2048 | 128 | `_fused_moe_*` (generic) |
| T ≥ 4096 | dynamic | Exact dispatch (`total_blocks.item()`) |

### Profiling 瓶颈分布 (NCU, B200)

| 阶段 | T=64 占比 | T=4096 占比 |
|------|----------|------------|
| **GEMM1** | **54.5%** | — |
| GEMM2 | 30.6% | **31.7%** |
| Reduce | — | 18.2% |
| Sort | — | 12.4% |
| Routing | — | 11.6% |

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

## 运行与测试

### 1. 创建 Modal Volume 并上传数据（一次性）

```bash
modal volume create flashinfer-trace
modal volume put flashinfer-trace /path/to/mlsys26-contest /
```

> 上传后路径：volume 内 `/mlsys26-contest/`，挂载在 `/data`，TraceSet 路径为 `/data/mlsys26-contest`

### 2. 打包 & 运行

```bash
# 打包 solution
python scripts/pack_solution_simple.py

# 在 Modal B200 上跑 benchmark
python -m modal run scripts/test_modal.py
```

### 3. A/B 对比测试（推荐）

```bash
# 存 baseline
python scripts/pack_solution_simple.py
copy solution.json solution_a.json

# 改代码后存实验版
python scripts/pack_solution_simple.py
copy solution.json solution_b.json

# 同一 B200 session 内 A/B 对比
python -m modal run scripts/ab_test_modal.py
```

> **判断标准**：看 mean speedup 的 Δ，±2% 以内都是噪声。

### 4. 本地快速验证（可选）

```bash
python test_kernel.py  # 只测 T=7，验证正确性
```

### 5. 竞赛提交

```bash
git tag submission-vX
git push origin submission-vX
# flashinfer-bot 自动拉取，用 config.toml 确定赛道，在裸金属 B200 上评测
```

---

## 项目结构

```
mlsys_note/
├── solution/
│   ├── triton/kernel.py         # ← 主力 Triton kernel
│   └── cuda/
│       ├── binding.py           # CUDA baseline (PyTorch/cuBLAS, 19/19 PASSED)
│       └── kernel.cu            # CUDA kernel 模板 (已废弃, Modal 无 nvcc)
├── scripts/
│   ├── test_modal.py            # Modal B200 benchmark（推荐）
│   ├── ab_test_modal.py         # A/B 对比测试（同 session）
│   ├── run_modal.py             # Modal 完整 benchmark
│   ├── profile_modal.py         # Modal NCU profiling
│   ├── debug_modal.py           # 逐步对比 kernel vs reference
│   ├── pack_solution_simple.py  # 打包 solution.json
│   └── ncu_profile_modal.py     # per-kernel 时间分解
├── config.toml                  # 配置（队名、赛道）
├── profiling_notes.md           # Profiling 分析 & 优化记录
├── log.md                       # 详细实验日志
├── solution.json                # 打包后的提交文件
└── mlsys26-contest/             # 比赛数据集
```

---

## Modal B200 噪声分析

通过在**同一 Modal B200 session** 内背靠背跑同一份代码（`ab_test_modal.py` A/B self-comparison），测量到的噪声水平：

| 指标 | 噪声范围 | 说明 |
|------|---------|------|
| **Mean speedup** | **±2%** | 非常稳定，可用于判断整体优劣 |
| **单个中小 T workload** | **±15%** | 同代码两次跑出 55.52x 和 64.68x (Δ=16.5%) |
| **Large-T (≥4096)** | **<1%** | Δ=0.1-0.2%，几乎零噪声 |

**判断优化有效的标准**：
- Mean speedup Δ > 2%（超出噪声范围）
- 或 Large-T latency Δ > 1%
- 单个 workload 的 ≤15% 变化不可作为判据

> **Modal 环境漂移**：同一代码在不同日期的 Modal B200 上 speedup 可差 20-30%。
> 如 Round 15 跑出 peak 106.65x，当前 Modal 可复现口径仅 peak ~55-65x。
> 代码没有退化，差异完全来自 Modal 共享实例的时钟/负载状态。

---

## 优化历程

### Phase 1: 基础架构 (0x → 12.9x)

| 阶段 | 内容 | Peak |
|------|------|------|
| 初版 | PyTorch eager per-expert 循环 | ~1x |
| torch.compile | 融合 dequant+GEMM+SwiGLU | ~5x |
| Monolithic Triton | 消除 per-expert launch | ~12.9x |

### Phase 2: Triton 核心优化 (12.9x → 46x)

| Commit | 优化内容 | 效果 |
|--------|---------|------|
| `496da33` | Triton autotune + GEMM2 BLOCK_K=128 | ~5x→8x |
| `3d3aace` | Hybrid token sort + K-loop pointer hoist | 结构性提升 |
| `c1943ae` | Routing gather-first + pre-alloc buffers | 减少 CPU overhead |
| `99c1b13` | **FP8 Native Tensor Core** (`tl.dot(fp8,fp8)`) | **2x FLOP 吞吐** |
| `786ee59` | GEMM2 post-dot B-scale | 消除 [BK,BN] dequant |
| Token-Centric Reduce | 零原子、零清零、零 copy | 小 T +46-78% |
| Parallel Sort | GPU tile 级并行 histogram | 打破单线程瓶颈 |

### Phase 3: 分桶特化 (46x → 55x+)

| Commit | 优化内容 | 效果 |
|--------|---------|------|
| `0b865b3` | T=1 专用 kernel 路径 | 消除 decode overhead |
| `c59e09d` | 3-phase (T>4096) | Large-T latency **-20%** |
| `d2fdf14` | **Medium-T bucket** (4-level BLOCK_M + 特化 kernel) | 多 workload latency **-20~54%** |
| `c685b02` | Exact dispatch for large-T | Large-T latency **-5%** |
| `9447f19` | Bugfix: exact-dispatch 对 medium-T 的影响 | 正确性修复 |

### Phase 4: 精度/带宽优化探索 (2026-03-25/26)

| 实验 | 结果 | 结论 |
|------|------|------|
| bf16 Intermediate | ❌ 9/19 精度失败 | SwiGLU 动态范围太大，abs_err 2048-4096 >> atol=1 |
| expert_out bf16 (在优化后的GEMM2复测) | ❌ Mean -2.1% | 显式强转向 bf16 会阻塞 HBM Store 破坏流水线，带宽收益无法弥补 Cast 延迟 |
| GEMM1 BLOCK_N=32 | ⚠️ 19/19 通过，autotuner 未选中 | 更小 tile 无优势 |
| 关闭 Triton LSR 优化 | ✅ Mean +4.9% | 缓解 SwiGLU 复杂寻址带来的寄存器溢出问题 |
| 编译器 L2 非对称 eviction | ✅ 叠加在上面 | 契合 MoE 专家权重高频重用，显著提升 L2 Hit Rate |
| 极深流水线 (num_stages=6-8) | ✅ Mean +2.1% | 掩盖 GEMM2 中 Intermediate 访存延迟，惠及 Medium-T |
| 2D Tiled Token Reduce (BLOCK_T=16/32) | ✅ Large-T +2.9% | 削减 10x Grid Launch 开销，靠 ILP 打通 HBM Reduce 延迟墙 |
| Column-Major 调度与并行 Dispatch | ✅ Medium-T 最高 +8.9% | 消除 GEMM1 计算依赖并提高权重 Cache 命中，精准打击中等 T 的延迟瓶颈 |

---

## 全 Commit 审计

根据噪声分析，所有 kernel.py 优化提交分为四级：

| 判定 | 含义 |
|------|------|
| ✅✅ **确定真实** | 多 workload latency 改善 >20%，或新增整个 kernel 路径 |
| ✅ **大概率真实** | Mean latency 改善 5-20%，或 large-T 改善 >5% |
| ⚠️ **不确定** | 改善 <5%，在噪声范围内 |
| ❌ **已确认退化/回退** | 实测退化或从主线回退 |

### 当前主线中的有效优化

| Commit | 描述 | 判定 | 理由 |
|--------|------|------|------|
| 早期架构 (`aa7c1b1`→`effc2f2`) | 从零到 12.9x | ✅✅ | 从无到有 |
| `496da33` | Triton autotune + BLOCK_K=128 | ✅✅ | peak 5x→8x |
| `3d3aace` | Hybrid sort + pointer hoist | ✅✅ | 算法结构变化 |
| `99c1b13` | FP8 native tensor core | ✅✅ | 2x FLOP 吞吐 |
| `c1943ae` | Routing gather-first | ✅ | CPU overhead 减少 |
| `786ee59` | GEMM2 post-dot B-scale | ✅ | 计算顺序变化 |
| `0b865b3` | T=1 专用 kernel (+481 行) | ✅✅ | 整个新路径 |
| `c59e09d` | 3-phase (T>4096) | ✅✅ | latency -20% |
| `d2fdf14` | Medium-T bucket | ✅✅ | latency -20~54% |
| `c685b02` | Exact dispatch (large-T) | ✅ | latency -5% |
| `9447f19` | Bugfix: exact-dispatch | ✅ | 正确性修复 |
| `9b341bc` | Buffer cache | ⚠️ | torch.compile 部分回退 |
| `0c6eeee` | FA4 GROUP_M=16 | ⚠️ | 仅加 config |
| `93e3a84` | 关闭 Triton LSR 优化 | ✅✅ | 峰值/均值显著提升 (+4.9%) |
| `93e3a84` | 非对称 eviction_policy | ✅✅ | 同上 |
| `93e3a84` | 极深流水线 GEMM2 | ✅ | 均值 +2.1%，有效隐藏访存延迟 |
| `f49c5ab` | 2D Tiled Token Reduce | ✅ | 结构性优化，大型 Workload (T>8000) 净提速 +2.5%~2.9% |
| (待提交) | Medium-T Column-Major | ✅ | `GROUP_M=32/64` 列排布与并行扫描，Medium-T 最高提速 +8.9% |

**结论：当前主线中所有生效改动都是 ✅ 或 ✅✅ 确定真实的。**

### 已回退实验

| 实验 | 判定 | 理由 |
|------|------|------|
| Direction 4 微调 (T≤16→medium GEMM2 等) | ❌ | full19 mean 退化 -11% |
| `52≤T≤62 → medium GEMM2` | ❌ | full19 退到 42.47x |
| bf16 intermediate | ❌ | 9/19 精度失败 |
| Tiny GEMM1-only dispatch | ❌ | full19 退到 42.55x |

---

## 已尝试但未生效的优化

### 精度硬限制（FP8/bf16 死路）

| 尝试 | 结果 | 原因 |
|------|------|------|
| bf16 Intermediate | 6/19 PASSED | SwiGLU fp32→bf16 截断，abs_err ~4096 |
| GEMM2 FP8 Online Quantize | 0/19 | fp8 (3-bit mantissa) 量化误差级联放大 |
| GEMM2 bf16×bf16 Dot | 3/19 | bf16 截断在 16 次 K-iter 中累积 |
| GEMM2 FP8 Per-128-Block-Scale | 0/19, abs=10K+ | fp8 物理精度极限，3 种变体全部失败 |
| MXFP8 `tl.dot_scaled` | matched_ratio 0.17-0.32 | e8m0 共享指数比 fp32 scale 更粗糙 |

**结论：GEMM2 A-side 必须保持 fp32，FP8 在严格容差下彻底封棺。**

### 架构级失败

| 尝试 | 结果 | 原因 |
|------|------|------|
| Persistent GEMM2 | 52x→15.6x | 丧失张量级并行度，内存延迟暴露 |
| TMA Accelerator | -1~2% | HBM 带宽已被 LDG 指令榨干 |
| 1D Atomic Scatter | 8.3x→5.37x | 标量原子风暴打瘫内存控制器 |
| Token Reduce 融合进 GEMM2 | TIMEOUT | Triton 2D tile 无法逐行 scatter |
| CUDA 自定义 C++ 扩展 | Modal 失败 | 评测沙盒无 nvcc |

### Microbatch 调度失败 (Direction 4)

| 尝试 | 结果 | 原因 |
|------|------|------|
| Split-K + Atomic Accumulate | 退化 | fixed cost + atomic 开销 > 收益 |
| Tiny GEMM1-only Dispatch | 42.55x mean | 算力利用率下降 |
| Direct GEMM2+Reduce | 46-51x | 破坏 GEMM2 tile reuse |
| Bucket/Autotune Follow-up Sweeps | 全回退 | 边际极限 |

---

## 关键约束

| 约束 | 说明 |
|------|------|
| Destination-passing style | `kernel(*inputs, *outputs)`，output 是最后一个参数 |
| 正确性标准 | 95% 元素满足 `abs_err ≤ 0.01` OR `rel_err ≤ 0.01` |
| 官方评测容差 | `atol=1, rtol=0.3, matched_ratio=0.9` |
| Scalar tensor | `local_expert_offset` 和 `routed_scaling_factor` 是 tensor 非 Python scalar |
| 显存 | 32 experts 权重 ~1.8GB FP8；B200 ~180GB 显存，不是瓶颈 |

---

## 注意事项

1. **Modal 环境 vs 本地环境：**
   - Modal: PyTorch 2.10.0+cu128, flashinfer-bench 0.1.2, Python 3.12
   - 本地 (Windows): 无法安装 triton，仅用于打包和代码审查
   - 本地 GPU 显存 < 24GB 跑不了大 workloads（reference 也会 OOM）

2. **flashinfer-bench API 差异：**
   - `Solution.sources` 在 Modal 上是 `list` 类型（本地是 `dict`）
   - 用 `pack_solution_simple.py` 打包更稳定

3. **CUDA Baseline (已废弃)：**
   - `solution/cuda/binding.py` 是 PyTorch/cuBLAS 参考实现（19/19 PASSED）
   - `solution/cuda/kernel.cu` 包含 fused kernel 模板 + FA4 fast_sigmoid
   - 被 Modal 评测环境封死（无 nvcc），仅作参考
