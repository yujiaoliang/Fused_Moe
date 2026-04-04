# CuTe Fused SwiGLU MoE Kernel — 开发笔记

> **硬件:** NVIDIA B200 (sm_100a, Blackwell)
> **框架:** CuTe DSL (CUTLASS 4.x) + Triton 3.6 + PyTorch 2.11.0+cu130
> **目标:** 替换 Triton GEMM1+SwiGLU kernel，利用 Blackwell tcgen05 MMA 指令集实现更高吞吐

---

## 1. 架构概览

### 1.1 文件结构

| 文件 | 职责 |
|------|------|
| `cute_kernel.py` | CuTe DSL kernel 类 `Sm100GroupedSwiGLUBlockscaledKernel`，包含 MMA mainloop、epilogue、tensormap 管理 |
| `cute_fused_gemm.py` | Host-side dispatch：gather buffer 管理、scale factor 转换、per-expert 指针/stride 构建、JIT 编译与 launch |
| `kernel.py` | 顶层 MoE dispatch，决定 CuTe vs Triton 路径 |

### 1.2 数据流

```
hidden_states (T, H) fp8e4m3
  │
  ├─[tma_gather_a_explicit]──► _GATHER_A_BUF (T_PAD, H) fp8e4m3    (按 sorted_token_ids 重排)
  │                            _GATHER_A_SCALE_BUF (T_PAD, H//32) e8m0 (A-side scale factors)
  │
  ├─[scale conversion]────────► sfb_w1, sfb_w3 (E, N//32, K_blks) e8m0 (B-side scale factors)
  │
  └─[CuTe Grouped GEMM]──────► intermediate (T_PAD, N_DIM) fp32
                                  = silu(GATHER_A @ W1.T) * (GATHER_A @ W3.T)
```

### 1.3 Kernel 配置

| 参数 | 值 | 说明 |
|------|------|------|
| CTA Tile | `(128, 128)` | MMA tiler (M × N per CTA) |
| Cluster | `(1, 1)` | 无 CTA cluster |
| SF Vec Size | `32` | 每 32 个 FP8 元素共享 1 个 e8m0 scale factor |
| Acc Dtype | `Float32` | Accumulator 精度 |
| Output Dtype | `Float32` | 与 `intermediate` buffer 匹配 |
| AB Pipeline | `Async` | TMA + 异步 pipeline (multi-stage) |
| Tensormap Update | `SMEM` | 运行时通过 SMEM 更新 TMA descriptor |

### 1.4 Kernel 工作流 (per CTA)

```
MMA Warp (warp 4):                    TMA Warp (warp 5):
  │                                     │
  │ ┌─for each tile (persistent)─┐      │ ┌─for each tile─┐
  │ │ if group_changed:          │      │ │ update_tensormap(A, B_w1, B_w3, SFA, SFB_w1, SFB_w3)
  │ │   wait tensormap update    │      │ │ for k_tile in K:
  │ │ for k_tile in K:           │      │ │   TMA load A, B_w1, B_w3 → SMEM
  │ │   gemm(acc1, A, B_w1)     │      │ │   TMA load SFA, SFB_w1, SFB_w3 → SMEM
  │ │   gemm(acc2, A, B_w3)     │      │ └──────────────────┘
  │ │ signal acc ready           │
  │ └────────────────────────────┘

Epilogue Warps (warp 0-3):
  │ ┌─for each tile─┐
  │ │ wait acc ready
  │ │ TMEM → REG: load acc1, acc2
  │ │ SwiGLU: silu(acc2) * acc1
  │ │ REG → SMEM → TMA store → GMEM (intermediate)
  │ └───────────────┘
```

---

## 2. 已完成的改动 (Bug Fix 记录)

### 2.1 Fix 1: Module-level CUTLASS 初始化
**问题:** `cutlass`, `cute`, `cuda_driver` 在函数内部 import 导致 PTX context 与 benchmark 多进程框架冲突，segfault。
**修复:** 所有 import 和 `Sm100GroupedSwiGLUBlockscaledKernel` 实例化移到模块级别。

### 2.2 Fix 2: 恢复动态 `_compute_grid` 参数
**问题:** `_compute_grid` 中 `total_num_clusters` 和 `max_active_clusters` 被错误移除，JIT trace 时 grid 计算失败。
**修复:** 恢复为 `__call__` 的动态参数，在 JIT trace 中正确传播。

### 2.3 Fix 3: 移除 `cutlass.Int32()` wrapper
**问题:** `cutlass.Int32(total_num_clusters)` 对已被 JIT tracer 追踪的 MLIR value 做了二次包装，触发 "integer conversion not supported"。
**修复:** 直接传递 Python `int`，DSL tracer 自动处理类型推导。

### 2.4 Fix 4: `np.int32` → Python `int` 类型转换
**问题:** `total_tile_clusters` 和 `_CUTE_MAX_CLUSTERS` 是 `np.int32` 类型，CuTe DSL 的 `IntTuple` 只接受原生 Python `int`。
**修复:**
```python
total_tile_clusters = int(sum(...))
_CUTE_MAX_CLUSTERS = int(utils.HardwareInfo().get_max_active_clusters(1))
```

### 2.5 Fix 5: 3D Tensor Layout for TMA
**问题:** `init_a`, `init_b_w1`, `init_b_w3`, `init_c` 是 2D `(M, K)` tensor，CuTe TMA atom 和 `local_tile` 要求 3D `(M, K, L)` layout。错误信息: `Operation creation failed` in `local_tile`。
**修复:** 所有 init tensor 添加 `.unsqueeze(-1)` → `(M, K, 1)`。

### 2.6 Fix 6: 动态 `block_m` 传播
**问题:** `T_PAD = total_blks * 64`（硬编码 64）和 `expert_ms = n_blks * 128`（硬编码 128）与实际 `block_m` 不匹配。Sorting kernel 的 block_offsets 以 `block_m` 为单位，而 `block_m` 可以是 16/32/64/128。
**修复:** `kernel.py` 传递 `block_m` 参数到 `run_cute_path()`，统一使用：
```python
T_PAD = total_blks_host * block_m
expert_ms.append(n_blks * block_m)
offset_m = e_starts[e_idx] * block_m
```

### 2.7 Fix 7: CuTe 路径 gating (`block_m >= 128`)
**问题:** 小 batch 时 `block_m = 16/32/64`，每个 expert 的 M 维度可能小于 CuTe CTA tile (128)，TMA 越界读取，产生 `±inf` 输出。
**修复:** `kernel.py` 添加条件: `if T > 1 and block_m >= 128:`

### 2.8 Fix 8: Expert 数量 guard (`num_g <= 16`)
**问题:** 当所有 32 个 expert 都有 token 时 (`num_g = 32`)，persistent tile scheduler 的 tensormap update 触发 `CUDA_ERROR_ILLEGAL_ADDRESS`。
**修复:** `run_cute_path()` 中添加 guard:
```python
if num_g > 16:
    return None  # 信号 caller 使用 Triton fallback
```
**阶段二深入调查 (5 个实验全部失败):**
- 去掉 `block_m >= 128` → tcgen05 128-row MMA 是硬件硬约束
- 强制 `block_m = 128` → 暴露 `num_g > 16` 是独立 bug
- 切 `TensorMapUpdateMode.GMEM` → kernel 绑定 SMEM 指令，`ILLEGAL_INSTRUCTION`
- Batch Split (16+16) → Batch 1 ✅ Batch 2 ❌ → 不是 group count 问题
- Batch Split + `torch.cuda.synchronize()` → 依然崩 → 不是 timing race
- **结论:** Experts 16-31 存在结构性 ILLEGAL_ADDRESS，疑似 CuTe DSL/PTS high group_idx 内部 bug

### 2.9 Fix 9: 移除 `torch.cuda.synchronize()` (阶段一)
**问题:** `cute_fused_gemm.py` 中有 4 处显式 `torch.cuda.synchronize()`。由于所有 kernel 都在同一个 CUDA stream 上排队，这些 sync 纯属不必要的 CPU 阻塞，且每次引入 ~20-50μs 延迟。
**修复:** 全部移除。CUDA stream 的自动排队保证了拓扑依赖。

### 2.10 Fix 10: Scale Factor 缓存 (阶段一)
**问题:** `gemm1_weights_scale` 的 e8m0 转换 (`repeat_interleave` × 2) 每次前向都重新计算，产生大量临时内存分配。
**修复:** 使用 `id(gemm1_weights_scale)` 作为 key 缓存 `_GATHER_W13_SCALE_CACHE`。首次计算后直接复用，避免重复的 `clamp → log2 → round → repeat_interleave` 链路。

### 2.11 Fix 11: `num_g > 16` 提前退出 (Early Exit)
**问题:** `num_g > 16` guard 位于 `run_cute_path()` 的 L200（太晚），在返回 `None` 之前已经执行了：`total_blocks.item()` (CPU-GPU sync)、`_GATHER_A_BUF` 分配、完整的 Triton gather kernel、`block_offsets.cpu()` (第二次 sync)。这些无用的 GPU 工作导致那 2 个 32-expert 的 Triton fallback workload 性能从 ~43x 暴跌到 ~10x。
**修复:** 将 `block_offsets.cpu()` 和 `num_g` 检查移到 gather kernel 之前，确保 early exit 时只有一次不可避免的 `total_blocks.item()` sync。

### 2.12 Fix 12: Cherry-pick 合并丢失 `grid1` 定义
**问题:** 将 master 的 `c7bf4ad` (large-T remapped GEMM1) cherry-pick 到 CuTe-dev 时，合并冲突解决遗漏了 `grid1 = lambda META: ...` 的定义。导致 Triton GEMM1 fallback 路径全部 `NameError: name 'grid1' is not defined`。
**修复:** 在 `if not _used_cute_gemm1:` 块内恢复 `grid1` 定义。

---

## 3. 结论：CuTe 路径完全无效 — Post-Mortem

### 3.1 最终状态

**CuTe 在所有 19 个 benchmark workload 上完全没有执行过。全部性能 100% 来自 Triton kernels。**

经过 12+ 个 bug fix、5 个失败实验、和完整的 A/B 测试后，我们发现 CuTe 集成被两个**互相矛盾的硬约束**彻底锁死。

### 3.2 根因分析：两个硬约束的死锁

```
                    ┌─────────────────────────────────────────────┐
                    │         CuTe 可执行区域 (理论上)              │
                    │    需要同时满足:                              │
                    │    ① block_m >= 128  (硬件约束)              │
                    │    ② num_g <= 16     (DSL bug 约束)          │
                    └─────────────────────────────────────────────┘

                            ① block_m >= 128
                                需要 T * TOP_K > 16384
                                即 T > 2048

                            ② num_g <= 16
                                但 T > 2048 时, T*8/32 ≈ 500+ tokens/expert
                                → 几乎所有 32 个 expert 都有 token
                                → num_g ≈ 32 > 16

                            ① ∧ ② = ∅  (空集)
```

**约束 ①: `block_m >= 128` (tcgen05 MMA 硬件约束)**

CuTe DSL 的 Blackwell tcgen05 MMA 指令要求每个 CTA 的 M 维度 tilé 固定为 128 行。这是硬件架构的刚性约束，不可通过软件绕过。当 `block_m < 128` 时（小/中 batch），每个 expert 分到的 token 行数不足 128，MMA 会越界读取 → `ILLEGAL_ADDRESS`。

MoE dispatch 的 `_select_block_m()` 根据 `T * TOP_K` 选择 block_m:

| T 范围 | T * TOP_K | block_m | 满足 ①? |
|--------|----------|---------|---------|
| 1-128 | ≤ 1024 | 16 (TINY) | ❌ |
| 129-512 | ≤ 4096 | 32 (SMALL) | ❌ |
| 513-2048 | ≤ 16384 | 64 (LARGE) | ❌ |
| **2049+** | **> 16384** | **128 (XLARGE)** | **✅** |

17 个 workload 的 T 值分布：1, 7, 14, 15, 16, 32, 52-62, 80, 901 — **全部 T ≤ 901，block_m 最大 64。CuTe 无法进入。**

**约束 ②: `num_g <= 16` (CuTe DSL Persistent Tile Scheduler bug)**

当 active expert 数 > 16，CuTe DSL 的 Persistent Tile Scheduler 触发 `CUDA_ERROR_ILLEGAL_ADDRESS`。通过 5 个系统性实验（batch split、GMEM mode、sync barrier 等），确认这是 CuTe DSL/CUTLASS 的内部 bug（疑似 tensormap 管理在 high group_idx 下的索引溢出），不可通过用户代码绕过。

仅有 2 个 workload 满足约束 ①（T=11948, T=14107, block_m=128），但：
- T=11948: 11948 × 8 / 32 ≈ **2990 tokens/expert** → 所有 32 expert 都有 token → `num_g = 32 > 16` ❌
- T=14107: 14107 × 8 / 32 ≈ **3527 tokens/expert** → 所有 32 expert 都有 token → `num_g = 32 > 16` ❌

### 3.3 全 19 workload 诊断表

| UUID | T (seq_len) | block_m | CuTe 入口 | num_g 检查 | CuTe 执行? |
|------|------------|---------|-----------|-----------|-----------|
| b8f4f012 | 7 | 16 | `block_m < 128` ❌ | — | ❌ 跳过 |
| e05c6c03 | 1 | — | `T == 1` 走 T1 路径 | — | ❌ 跳过 |
| 6230e838 | 32 | 16 | `block_m < 128` ❌ | — | ❌ 跳过 |
| 8f1ff9f1 | 80 | 16 | `block_m < 128` ❌ | — | ❌ 跳过 |
| 1a4c6ba1 | 901 | 64 | `block_m < 128` ❌ | — | ❌ 跳过 |
| a7c2bcfd | 16 | 16 | `block_m < 128` ❌ | — | ❌ 跳过 |
| 2e69caee | 15 | 16 | `block_m < 128` ❌ | — | ❌ 跳过 |
| 8cba5890 | 14 | 16 | `block_m < 128` ❌ | — | ❌ 跳过 |
| **5e8dc11c** | **14107** | **128** | ✅ 进入 | `num_g=32 > 16` ❌ | ❌ **fallback** |
| **58a34f27** | **11948** | **128** | ✅ 进入 | `num_g=32 > 16` ❌ | ❌ **fallback** |
| 5eadab1e | 62 | 16 | `block_m < 128` ❌ | — | ❌ 跳过 |
| eedc63b2 | 59 | 16 | `block_m < 128` ❌ | — | ❌ 跳过 |
| e626d3e6 | 58 | 16 | `block_m < 128` ❌ | — | ❌ 跳过 |
| 74d7ff04 | 57 | 16 | `block_m < 128` ❌ | — | ❌ 跳过 |
| 4822167c | 56 | 16 | `block_m < 128` ❌ | — | ❌ 跳过 |
| 81955b1e | 55 | 16 | `block_m < 128` ❌ | — | ❌ 跳过 |
| 76010cb4 | 54 | 16 | `block_m < 128` ❌ | — | ❌ 跳过 |
| fc378037 | 53 | 16 | `block_m < 128` ❌ | — | ❌ 跳过 |
| f7d6ac7c | 52 | 16 | `block_m < 128` ❌ | — | ❌ 跳过 |

**结论: 0/19 workload 执行了 CuTe 路径。**

### 3.4 为什么之前的 A/B Test 显示 "+5.5%"？

A/B test 对比的是 **main 分支 vs CuTe-dev 分支**的整体性能。CuTe-dev 分支从 main cherry-pick 了多个 Triton 优化（remapped GEMM1 等），这些优化提供了 +5.5% 的提升。**这个提升与 CuTe 无关。**

### 3.5 CuTe 代码仍保留的原因

虽然 CuTe 对当前 workload 无效，我们选择保留代码（`cute_fused_gemm.py`, `cute_kernel.py`）作为：
- **技术存档**：12 个 bug fix 记录了 Blackwell tcgen05 + CuTe DSL 的实战经验
- **未来参考**：如果 CUTLASS 修复了 `num_g > 16` PTS bug，或者 benchmark 增加 T ∈ [2048, ∞) 且 E_LOCAL ≤ 16 的 workload，CuTe 路径可以立即激活
- **零性能代价**：CuTe 路径被 `block_m >= 128 and not use_exact_dispatch` 完全门控，不产生任何运行时开销

---

## 4. 性能优化记录

### 4.1 实际生效的优化 (全部是 Triton)

| 优化 | 来源 | 效果 |
|------|------|------|
| Remapped GEMM1 (cherry-pick from master) | `kernel.py` L844-927 | 大 T workload +5% |
| Exact workload dispatch | `kernel.py` L2035-2040 | 减少 overlaunch |
| 禁用 remapped GEMM1 for T>4096 | `kernel.py` L1709 | 修复 2 个 INCORRECT_NUMERICAL → 19/19 PASSED |
| 跳过大 T 的 CuTe 调用 | `kernel.py` L2079 | 消除 2 个大 T workload 的无用 D2H sync |

### 4.2 CuTe 侧优化 (已实现但无效果)

| 优化 | 文件 | 说明 |
|------|------|------|
| 缓存 dispatch buffers (ptrs/strides/shapes) | `cute_fused_gemm.py` L68-100 | 按 num_g 缓存，避免 per-call `torch.empty()` |
| 预填充常量 strides | `cute_fused_gemm.py` L84-96 | H, KBLKS_32, N_DIM 只填一次 |
| 指针算术替代 slice+data_ptr() | `cute_fused_gemm.py` L271-298 | 消除 7×num_g 次 Python 对象创建 |
| 缓存 CuTe metadata wrappers | `cute_fused_gemm.py` L104-119 | `convert_cute_tensor` 按 tensor id 缓存 |
| 缓存 init tensors | `cute_fused_gemm.py` L322-323 | JIT 编译后复用 init tensors |
| GPU-side expert counting | `cute_fused_gemm.py` L125-142 | Triton kernel 替代 `block_offsets.cpu()` |
| 传递 total_blks_host | `cute_fused_gemm.py` L192 | 消除 redundant `total_blocks.item()` |

**以上全部优化对性能零影响**，因为 CuTe 路径从未被执行。

---

## 5. 教训与反思

### 5.1 架构陷阱：没有提前验证 workload coverage

**最大的错误**：在投入大量 CuTe 集成工作之前，没有先检查 19 个 workload 的 T 值分布。如果在第一天就运行诊断脚本打印出 T 值（1, 7, 14-16, 32, 52-62, 80, 901, 11948, 14107），就会立即发现 17/19 的 T 值远低于 CuTe 需要的 `T > 2048` 阈值。

### 5.2 硬件约束 vs 软件灵活性

tcgen05 MMA 的 128-row tile 是芯片级硬约束，不像 Triton 可以任意选择 BLOCK_M。这意味着 CuTe DSL 天然不适合 小 batch MoE workload（T < 2048），而比赛的 17/19 workload 恰好都在这个范围。

### 5.3 Guard 的累积效应

每个 guard（`block_m >= 128`, `num_g <= 16`, `T > 1`）单独看都合理且必要，但它们的 AND 组合产生了空集。这种"约束累积导致可行解空间坍缩为零"的模式是一个通用的工程教训。

### 5.4 CuTe DSL 的实战评价

- **优点**：接近硬件的 MMA 控制、TMA 异步 pipeline 自动管理、Persistent Tile Scheduler
- **限制**：
  - `num_g > 16` 的结构性 crash 无法通过用户代码绕过
  - 128-row MMA tile 是硬约束，无法 sub-tile
  - JIT 编译 30-60 秒，不适合 benchmark 的冷启动场景
  - Python DSL → PTX → cubin 的工具链，调试难度极高（错误信息几乎都是 ILLEGAL_ADDRESS）

---

## 6. 已验证的失败方向 (勿重复)

| 方向 | 失败原因 |
|------|--------|
| `cutlass.Int32()` 包装 JIT 参数 | MLIR 阶段 "integer conversion not supported" |
| 2D init tensor (无 unsqueeze) | `local_tile` rank mismatch (`3 vs 2`) |
| 硬编码 `T_PAD = total_blks * 64` | Block size mismatch → ILLEGAL_ADDRESS |
| try/except 捕获 CUDA crash | CUDA context 损坏后所有后续操作都会失败，无法恢复 |
| 32-slot SMEM tensormap buffer | 28KB SMEM 膨胀导致 ~10% 性能回退 |
| 去掉 `block_m >= 128` guard | tcgen05 128-row MMA 是硬件硬约束，ILLEGAL_ADDRESS |
| 强制 `block_m = 128` for all | 暴露 `num_g > 16` 独立 bug |
| `TensorMapUpdateMode.GMEM` | kernel 代码绑定 SMEM tensormap 管理，切 GMEM → ILLEGAL_INSTRUCTION (715) |
| Batch Split (16+16) | Batch 1 成功，Batch 2 crash → 不是 group count 问题 |
| Batch Split + `torch.cuda.synchronize()` | 加 sync 依然结构性崩溃 → 不是 timing/race |
| `num_g` guard 放在 gather 之后 | gather + alloc 白费 → fallback 从 43x 暴跌到 10x |
| CuTe 侧 Python 循环优化 | CuTe 路径从未被执行，优化无效 |
| CuTe 侧 D2H sync 消除 | CuTe 路径从未被执行，优化无效 |

---

## 7. 关键代码位置速查

| 功能 | 文件 | 行号 |
|------|------|------|
| CuTe 路径入口 guard | `kernel.py` | L2076-2079 |
| `run_cute_path()` 主函数 | `cute_fused_gemm.py` | L188-338 |
| `num_g > 16` early exit | `cute_fused_gemm.py` | L219-223 |
| GPU-side expert counting | `cute_fused_gemm.py` | L125-142 |
| Cached dispatch buffers | `cute_fused_gemm.py` | L68-100 |
| Cached CuTe metadata wrappers | `cute_fused_gemm.py` | L104-119 |
| Gather kernel (Triton) | `cute_fused_gemm.py` | L144-187 |
| Scale factor e8m0 转换 | `cute_fused_gemm.py` | L239-254 |
| JIT 编译 & cache | `cute_fused_gemm.py` | L316-332 |
| Triton GEMM1 fallback (`grid1`) | `kernel.py` | L2092-2150 |
| Remapped GEMM1 guard (T>4096 fix) | `kernel.py` | L1704-1712 |
| Kernel class 定义 | `cute_kernel.py` | L15-45 |
| MMA mainloop | `cute_kernel.py` | L580-640 |
| SwiGLU epilogue | `cute_kernel.py` | L738-765 |

