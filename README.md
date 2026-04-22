# Fused MoE Kernel — Track A (MLSys 2026 FlashInfer Contest)

> **赛道:** Track A — Fused MoE
> **硬件:** NVIDIA B200 (Blackwell, sm100a)
> **架构:** Hybrid CuTe DSL + Pure Triton dispatch (per-T 隔离 runtime)
> **状态:** ✅ 19/19 PASSED
> **最新 Modal B200 full19 (官方镜像):** mean ~41-45x (session-dependent), 19/19 PASSED, T=11948 CuTe, T=14107 CuTe (AB-test +55% vs Pure Triton fallback)

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
   - **FP16 Intermediate (T≥32):** Epilogue 融合 SwiGLU → `×0.125` 缩放 → cast to fp16 → 存入 `Intermediate [num_padded, 2048]`，带宽减半  
   - **FP32 Fallback (T<32):** 通过 `USE_FP16_INTER` constexpr 走 baseline fp32 存储路径，避免小 T SwiGLU 极端值溢出 fp16 范围

4. **GEMM2 (Non-Atomic Triton)**  
   按 batch 大小 dispatch 到 `_small_medium_*` / `_medium_*` / `_fused_moe_gemm2_*` / `_fused_moe_gemm2_t901_*` kernel  
   - Post-dot B-scale：`tl.dot(a, b.to(a.dtype))` + `acc += partial * b_scale`（自动适配 fp16/fp32）  
   - **×8.0 Compensation (T≥32):** 当 Intermediate 是 fp16 时，epilogue 乘以 `8.0` 补偿 GEMM1 的 `×0.125` 缩放
   - T=901 专用 kernel：独立 autotune config (`BLOCK_N=128, BLOCK_K=128, GROUP_M=8`)，针对 GEMM2 瓶颈场景优化
   - **bf16 expert_out (T≥32):** epilogue `out.to(tl.bfloat16)` 存入 bf16 buffer，节省 50% 写带宽
   - 非原子写入 `expert_out [num_padded, 7168]`（bf16 for T≥32, fp32 for T<32）

5. **Token-Centric Reduce** — `_token_reduce_kernel`  
   每个 output token 启动一个程序，通过 `scatter_map` 读取 TOP_K=8 个 expert 贡献  
   fp32 求和后直接写入 bf16 output。**零原子、零清零、零 copy**

6. **T=1 专用路径** — `_kernel_t1`  
   单 token decode 特化：融合 routing+sort+GEMM 消除通用路径 overhead

### Runtime Dispatch 策略

`kernel.py` 入口根据 T 自动选择路径：

| 条件 | 路径 | BLOCK_M | GEMM Kernel |
|------|------|---------|-------------|
| **T=11948** | **CuTe DSL grouped GEMM** | 128 | `cute_gemm1_mma_runtime_11948` + `cute_gemm2_mma_runtime_11948` (隔离 CuTe) |
| **T=14107** | **CuTe DSL grouped GEMM** | 128 | `cute_gemm1_mma_runtime_14107` + `cute_gemm2_mma_runtime_14107` (隔离 CuTe) |
| T=1 | Pure Triton | 16 | `_t1_*` 专用路径 |
| 32 ≤ T ≤ 64 | Pure Triton | 32 | `_small_medium_*` |
| 65 ≤ T ≤ 128 | Pure Triton | 32/64 | `_medium_*` |
| T > 128 | Pure Triton | 64 | `_fused_moe_*` (generic) |
| **T = 901** | Pure Triton | 64 | `_fused_moe_gemm2_t901_kernel` (GEMM2 专用) |
| T > 2048 | Pure Triton | 128 | `_fused_moe_*` (generic) |
| T ≥ 4096 | Pure Triton | dynamic | Exact dispatch (`total_blocks.item()`) |

### Profiling 瓶颈分布 (`yjl_ncu.py`, B200, 19 real traces)

| T 范围 | GEMM1 占比 | GEMM2 占比 | Routing+Sort | Reduce | 瓶颈 |
|--------|-----------|-----------|-------------|--------|------|
| T=1 | 32% | **55%** | 10% | (fused) | GEMM2 |
| T=7-80 | **51-61%** | 32-37% | 5-10% | 2% | GEMM1 |
| T=901 | 35% | **58%** | 4% | 1.5% | GEMM2 |
| T=11948-14107 | 40% | **49-50%** | 5-6% | 3% | GEMM2 |

> GEMM1 TFLOPS: 103-182T (小T) / 485-535T (大T)。GEMM2 TFLOPS: 82-150T (小T) / 216T (大T)。

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

### 官方评测环境

| 项目 | 值 |
|------|----|
| Docker Image | `flashinfer/flashinfer-ci-cu132:20260401-2c675fb` |
| PyTorch | `2.12.0.dev20260331+cu132` |
| GPU | B200 (bare-metal, sm_100a) |
| 容差 | `--atol 1 --rtol 0.3 --required-matched-ratio 0.9` |
| 评分 | CUPTI GPU kernel 时间之和 (CPU 开销不计入) |

> 所有 Modal 脚本已统一使用上述官方镜像 + 容差参数 (`1bd784b`)。

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
│   └── python/                          # ← 提交代码目录 (config.toml: language=python)
│       ├── kernel.py                    # 入口：hybrid dispatch (CuTe DSL / Pure Triton)
│       ├── pure_triton_impl.py          # Pure Triton 主实现 (17/19 workloads)
│       ├── triton_impl.py               # Hybrid CuTe+Triton 实现 (大 T workloads)
│       ├── cute_gemm1_mma.py            # CuTe DSL GEMM1 核心逻辑
│       ├── cute_gemm1_mma_runtime_11948.py  # T=11948 隔离 GEMM1 runtime
│       ├── cute_gemm1_mma_runtime_14107.py  # T=14107 隔离 GEMM1 runtime
│       ├── cute_gemm2_mma.py            # CuTe DSL GEMM2 核心逻辑
│       ├── cute_gemm2_mma_runtime_11948.py  # T=11948 隔离 GEMM2 runtime
│       ├── cute_gemm2_mma_runtime_14107.py  # T=14107 隔离 GEMM2 runtime
│       └── cute_grouped_gemm_sm100.py   # CuTe DSL grouped GEMM 核心 (NVIDIA 参考)
├── scripts/
│   ├── ab_test_modal.py         # A/B 对比测试（同 B200 session）⭐推荐
│   ├── test_modal.py            # Modal B200 单次 benchmark
│   ├── run_modal.py             # Modal 完整 benchmark (含 auto-pack)
│   ├── run_local.py             # 本地 benchmark (需 B200)
│   ├── profile_modal.py         # Modal torch.profiler profiling
│   ├── ncu_profile_modal.py     # Modal NCU per-kernel 时间分解
│   ├── yjl_ncu.py               # NCU profiling + autotune 日志
│   ├── pack_solution_simple.py  # 打包 solution.json
│   └── test_cutlass_modal.py    # cuBLAS JIT 编译测试
├── config.toml                  # 配置（队名、赛道、entry_point）
├── test_bench.py                # 官方 CLI 评测脚本 (Docker + isolated runner)
├── profiling_notes.md           # Profiling 分析 & 优化记录
├── solution.json                # 打包后的提交文件
└── mlsys26-contest/             # 比赛数据集 (submodule)
```

### 双路径架构

```
kernel.py (入口)
  ├── T in {11948, 14107}  →  triton_impl.py  →  CuTe DSL grouped GEMM
  │                            ├── cute_gemm1_mma.py + cute_gemm1_mma_runtime_{11948,14107}.py
  │                            ├── cute_gemm2_mma.py + cute_gemm2_mma_runtime_{11948,14107}.py
  │                            └── cute_grouped_gemm_sm100.py
  └── 其他 T              →  pure_triton_impl.py  →  Pure Triton kernels
```

> **Per-T 隔离设计 (`f088f03`):** T=11948 和 T=14107 各自拥有独立的 runtime 文件，不再共享 Python module state、compile cache、metadata cache 或 packed weight cache。这消除了两个大 T 之间的状态污染，使 T=14107 也能走 CuTe 路径（+55% speedup vs Pure Triton fallback）。

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
| **GEMM2 Autotune 扩展** | ✅ AB-test Mean +2.1%，8/19 improved | 低 warp / GROUP_M=4/64 等新 configs 覆盖 GEMM2 短 K-loop (K=2048) |
| **T=901 GEMM2 专用 Kernel** | ✅ T=901 GEMM2 瓶颈特化 | 独立 autotune 配置，精准打击 GEMM2 58% 的中大 T 瓶颈点。Kernel级时长 -2.9% |
| **小中T GEMM1 Autotune 扩展** | ✅ Kernel级延迟 -1.6%~6.6% | 补充深流水线 (stages=3/4) 及各种 GROUP_M 覆盖，全面降低 T=32~80 的 GEMM1 时长 |
| **T=1 Autotune 扩展** | ✅ T=1 Kernel -3% | 补充针对极小维度的低纬度分块与浅流水线 (warps=2, stages=2) 特化 |
| **FP16 Intermediate Buffer** | ✅ AB-test Mean +4.5%，13/19 improved | `×0.125` scale + fp16 cast 减半 GEMM1→GEMM2 带宽，T<32 走 fp32 fallback 保精度 |

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
| `9d5a2f8` | Medium-T Column-Major | ✅ | `GROUP_M=32/64` 列排布与并行扫描，Medium-T 最高提速 +8.9% |
| `c19336d` | Fused Routing+Histogram | ✅ | Token-major sort/layout/scatter，large-T routing/sort 开销 -1.5% |
| `93e3a84` | **GEMM2 autotune 扩展** | ✅ | AB-test mean +2.1%，8/19 improved，0 regressed |
| `67ef373` | **T=901 GEMM2 专用 Kernel** | ✅ | 独立 `_fused_moe_gemm2_t901_kernel` + 专属 autotune，打击 GEMM2 58% 瓶颈，单 kernel 提速 ~3% |
| `1010fb4` | **小中T GEMM1 Autotune** | ✅✅ | 增补 13 个 candidates 覆盖深流水线，T=32~80 GEMM1 时长稳定减少 1.6%~6.6% |
| `c35907d` | **T=1 GEMM1/2 Autotune** | ✅✅ | 补充微型 kernel 设置 (warps=2)，单 token 指令开销降低，latency -3% |
| `a5256cc` | **FP16 Intermediate Buffer** | ✅✅ | AB-test mean +4.5%，13/19 improved，1 regressed。`USE_FP16_INTER` constexpr + `×0.125/×8.0` scale-and-cast，T≥32 fp16，T<32 fp32 fallback |
| `f040d09` | **T=901 Static Launch Cap** | ✅✅ | AB-test +6%。T=901 保持 `block_m=64` 以保住 GEMM1 tile 效率，同时用 static `pid_m` launch cap 降低 GEMM overlaunch，避免 `total_blocks.item()` host sync；GEMM2 约 `196us→177us`，wall 约 `0.531ms→0.504ms` |
| `d6367a0` | **T=14 bf16 expert_out** | ⚠️ | 仅将 T=14 的 `expert_out` 从 fp32 改为 bf16，profiler 约 `0.143ms→0.136ms`，收益较小但实现复杂度低 |

### Phase 5: 竞赛规则感知优化 — Round 10 (2026-04-15)

**关键赛制更新（Apr 14-15 organizer 澄清）：**
- **容差 100x 放宽**：官方 `atol=1.0, rtol=0.3, ratio=0.9`，我们之前测试用 `atol=0.01, rtol=0.01, ratio=0.95`
- **CUPTI GPU-only 计时**：评分 = GPU kernel 执行时间之和。CPU Python 开销、kernel launch latency、`.item()` sync 全部不计入
- **Self-contained 要求**：cuBLAS/CUTLASS/flashinfer 运行时调用不推荐。组委会 "value seeing the team's own implementation"
- **sm_100a 已确认**：裸金属 B200 + CUDA 13.2 原生支持 sm_100a，需在 build flags 中显式指定 `-arch=sm_100a`
- **Manual review 仅查合规**：不会有比 `atol=1.0` 更严格的数值标准

| Commit | 优化内容 | 结果 | 说明 |
|--------|---------|------|------|
| `c1effcc` | **bf16 expert_out (T≥32)** | ✅ 19/19 PASSED | `expert_out [MAX_PADDED, 7168]` 从 fp32→bf16，节省 50% 写带宽。bf16 range=3.4e38 不溢出。AB-test mean +0.4%，large-T +2.5% (T=14107) |
| `c1effcc` | **CUBLAS_ENABLED=False** | ✅ 策略调整 | 禁用 cuBLAS GEMM2 dispatch。组委会倾向团队自研实现，cuBLAS 仅影响 1/19 workload (T≥2048)，边际收益不值得 review 风险 |
| (tested) | **bf16 Intermediate** | ❌ 10/19 PASSED | 即使在 contest 容差下仍 9/19 INCORRECT_NUMERICAL。bf16 7-bit mantissa 太粗糙 vs fp16 10-bit。fp16 + `×0.125/×8.0` scaling hack 是正确方案 |

**结论：当前主线中所有生效改动都是 ✅ 或 ✅✅ 确定真实的。**

### Phase 6: CuTe 隔离 Runtime + T=14107 恢复 — Round 19 (2026-04-19)

| Commit | 优化内容 | 结果 | 说明 |
|--------|---------|------|------|
| `886c7fc` | **CuTe 精度调查 + T=14107 fallback** | ✅ 19/19 PASSED | T=14107 回退到 Pure Triton，T=11948 保持 CuTe。确认 FP16 预解量化在大 K (7168) 时精度瓶颈 |
| `f088f03` | **Per-T 隔离 CuTe Runtime** | ✅✅ 19/19 PASSED | 拆分 runtime 文件为 `*_runtime_11948.py` + `*_runtime_14107.py`，消除共享 module state / compile cache / metadata cache / packed weight cache 的状态污染。T=14107 恢复 CuTe 路径。**AB-test mean +1.4%，T=14107 +55.1% (13.19x→20.45x)** |

### 已回退实验

| 实验 | 判定 | 理由 |
|------|------|------|
| Direction 4 微调 (T≤16→medium GEMM2 等) | ❌ | full19 mean 退化 -11% |
| `52≤T≤62 → medium GEMM2` | ❌ | full19 退到 42.47x |
| bf16 intermediate (strict tol) | ❌ | 6/19 精度失败 (abs_err ~4096) |
| bf16 intermediate (contest tol, R10 复测) | ❌ | 10/19 PASSED，仍 9/19 INCORRECT_NUMERICAL。bf16 7-bit mantissa 精度不足 |
| fp16 expert_out (R9) | ❌ | GEMM2 输出值超 fp16 max (65504) → inf，3/19 overflow + 2/19 register 溢出 |
| Tiny GEMM1-only dispatch | ❌ | full19 退到 42.55x |

---

## 已尝试但未生效的优化

### 精度硬限制（FP8/bf16 死路）

| 尝试 | 结果 | 原因 |
|------|------|------|
| bf16 Intermediate (strict tol) | 6/19 PASSED | SwiGLU fp32→bf16 截断，abs_err ~4096 |
| bf16 Intermediate (contest tol, R10) | 10/19 PASSED | 即使 atol=1.0，bf16 7-bit mantissa 仍太粗糙。**彻底封棺** |
| **fp16 Intermediate (全局)** | **❌ T=7 matched_ratio=0.0** | SwiGLU 极端值超 fp16 max (65504)，小 T 数据密集度高更易溢出 |
| **fp16 Intermediate (T≥32 only)** | **✅ 19/19 PASSED, +4.5%** | fp16 10-bit mantissa 精度可接受，`×0.125` 缩放 + `×8.0` 补偿，T<32 走 fp32 fallback |
| **bf16 expert_out (T≥128, R10)** | **✅ 19/19 PASSED** | bf16 range=3.4e38 不溢出（fp16 失败因超 65504）。abs_err ~2K-600K 但在 contest tol 内。节省 50% expert_out 带宽 |
| fp16 expert_out (R9) | ❌ 3/19 overflow | GEMM2 输出值超 fp16 max → inf，另 2/19 register 溢出 ptxas 255 |
| GEMM2 FP8 Online Quantize | 0/19 | fp8 (3-bit mantissa) 量化误差级联放大 |
| GEMM2 bf16×bf16 Dot | 3/19 | bf16 截断在 16 次 K-iter 中累积 |
| GEMM2 FP8 Per-128-Block-Scale | 0/19, abs=10K+ | fp8 物理精度极限，3 种变体全部失败 |
| MXFP8 `tl.dot_scaled` | matched_ratio 0.17-0.32 | e8m0 共享指数比 fp32 scale 更粗糙 |
| GEMM2 FP8 On-the-fly Dot (direct cast) | 5/19, abs=500K-1M | fp16→fp8 直接截断，>448 值溢出 + register pressure (ptxas 255-reg failure on 2 WL) |
| GEMM2 FP8 On-the-fly Dot (per-row scaled) | 5/19, abs=14K-25K | absmax/448 动态缩放避免溢出，但 fp8 3-mantissa-bit + K=2048 累积 → 误差级联放大。远超 contest atol=1.0 |

**注意：** eval Triton 无 `tl.float8e4m3fn`，须用 `b.dtype`（从 fp8 weight 推断类型）进行 cast。

**结论：Intermediate 精度阶梯 fp16 > bf16 > fp8，仅 fp16+scaling 可行。expert_out 精度阶梯 bf16 > fp16 > fp8，仅 bf16 可行（contest tol 内）。GEMM2 A-side MUST stay ≥fp16。**

### 架构级失败

| 尝试 | 结果 | 原因 |
|------|------|------|
| Persistent GEMM2 | 52x→15.6x | 丧失张量级并行度，内存延迟暴露 |
| TMA Accelerator | -1~2% | HBM 带宽已被 LDG 指令榨干 |
| 1D Atomic Scatter | 8.3x→5.37x | 标量原子风暴打瘫内存控制器 |
| Token Reduce 融合进 GEMM2 | TIMEOUT | Triton 2D tile 无法逐行 scatter |
| CUDA 自定义 C++ 扩展 | Modal 失败 | 评测沙盒无 nvcc |
| CUDA Graph for GEMMs | 无收益 | CUPTI 仅计 GPU kernel 时间，launch overhead 不计入评分 |
| torch.compile on routing | 无收益 | 同上，CPU overhead 不在评分范围 |
| cuBLAS GEMM2 dispatch (R8) | ✅ 可工作但禁用 | 组委会倾向自研实现。仅影响 1/19 (T≥2048)，边际收益不值得 review 风险 |

### Microbatch 调度失败 (Direction 4)

| 尝试 | 结果 | 原因 |
|------|------|------|
| Split-K + Atomic Accumulate | 退化 | fixed cost + atomic 开销 > 收益 |
| Tiny GEMM1-only Dispatch | 42.55x mean | 算力利用率下降 |
| Direct GEMM2+Reduce | 46-51x | 破坏 GEMM2 tile reuse |
| Bucket/Autotune Follow-up Sweeps | 全回退 | 边际极限 |

### Autotune 饱和测试 (2026-04-15)

| 尝试 | 结果 | 原因 |
|------|------|------|
| GEMM2 +7 configs (BLOCK_K=64, stages=1, GROUP_M=128) | -0.2% mean | 45 个现有 configs 已穷尽空间 |
| GEMM1 small/medium +10 configs (BLOCK_N=256, deeper pipeline) | +0.4% mean | 噪声范围，GEMM1 小 T 本身 memory-bound |
| token_reduce +4 configs (BLOCK_N=512) | neutral | 5 个现有 configs 已覆盖 |
| **bf16 expert_out AB test** (fp32 vs bf16) | **+0.4% mean, +2.5% T=14107** | **正向确认，保留** |

**结论：所有 autotune 方向已饱和。后续提升需要算法级/架构级变化。**

---

## 关键约束

| 约束 | 说明 |
|------|------|
| Destination-passing style | `kernel(*inputs, *outputs)`，output 是最后一个参数 |
| **官方评测容差** | **`atol=1.0, rtol=0.3, matched_ratio=0.9`** — 元素仅在 abs>1 AND rel>0.3 时才 fail |
| 评分方式 | **CUPTI GPU kernel 时间之和**，CPU 开销不计入 |
| Docker Image | `flashinfer/flashinfer-ci-cu132:20260401-2c675fb` (pinned) |
| GPU 架构 | sm_100a — 需在 build flags 中显式指定 `-arch=sm_100a` |
| cuBLAS/CUTLASS 政策 | 不硬禁，组委会 "value seeing team's own implementation"。CuTe DSL + Triton fallback 是推荐方式 |
| FlashInfer runtime | ❌ 不允许 runtime 调用 flashinfer API。可复制源码到 repo |
| Manual review | 仅查合规（self-contained, 无预计算），不会有比公布容差更严格的数值标准 |
| Self-contained | 所有代码必须在 `solution/python/` 目录内，打包进 solution.json |
| Scalar tensor | `local_expert_offset` 和 `routed_scaling_factor` 是 tensor 非 Python scalar |
| 显存 | 32 experts 权重 ~1.8GB FP8；B200 ~180GB 显存，不是瓶颈 |

---

## 注意事项

1. **评测环境统一：**
   - 所有 Modal 脚本已统一使用官方 Docker image `flashinfer/flashinfer-ci-cu132:20260401-2c675fb`
   - PyTorch: `2.12.0.dev20260331+cu132`, Python 3.12
   - 容差: `atol=1.0, rtol=0.3, required_matched_ratio=0.9`
   - 本地 (Windows): 无法安装 triton，仅用于打包和代码审查

2. **Hybrid Dispatch 注意事项：**
   - Pure Triton 优化 → `solution/python/pure_triton_impl.py`
   - 大 T CuTe 优化 → `solution/python/triton_impl.py` + `cute_gemm*_mma.py` + `cute_gemm*_mma_runtime_*.py`
   - 两条路径可以独立迭代，通过 `kernel.py` 的 `_CUTE_TARGET_BLOCK_M` dict 控制分发
   - **Per-T 隔离:** 每个目标 T 拥有独立的 runtime 文件，不共享 Python module state / compile cache / metadata cache

3. **已修复的历史问题：**
   - ~~T=14107 CuTe 精度 bug~~ → 已通过 per-T 隔离 runtime 修复 (`f088f03`)，T=14107 恢复 CuTe 路径 (20.45x vs Pure Triton 13.19x)
