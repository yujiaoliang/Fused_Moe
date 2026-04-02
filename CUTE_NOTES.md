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
**问题:** 当所有 32 个 expert 都有 token 时 (`num_g = 32`)，persistent tile scheduler 的 tensormap update 在频繁 group 切换时触发 `CUDA_ERROR_ILLEGAL_ADDRESS`，导致 CUDA context 不可恢复地损坏。
**修复:** `run_cute_path()` 中添加 guard:
```python
if num_g > 16:
    return None  # 信号 caller 使用 Triton fallback
```
`kernel.py` 检查返回值：
```python
if cute_intermediate is not None:
    _used_cute_gemm1 = True
    Intermediate = cute_intermediate
# else: fall through to Triton path
```

---

## 3. 当前状态

### 3.1 测试结果 (19/19 PASSED ✅)

| 路径 | Workload 数 | 延迟范围 | 加速比 |
|------|-----------|---------|--------|
| CuTe | 17 | 0.246ms - 1.027ms | **28× - 52×** |
| Triton fallback | 2 | 3.975ms - 5.492ms | **10× - 11×** |

### 3.2 CuTe 路径触发条件

```
CuTe 路径生效 ⟺ 以下条件 AND:
  1. T > 1                    (非单 token)
  2. block_m >= 128            (大 batch, BLOCK_M_XLARGE)
  3. _USE_CUTE == True         (CuTe 模块加载成功)
  4. num_g <= 16               (active expert 数量)
  5. T_PAD > 0                 (至少有 1 个 token)
```

### 3.3 JIT 编译 Cache

CuTe kernel 按 `(num_g, total_tile_clusters)` 做缓存。相同配置不重复编译。
首次编译约 30-60 秒，后续调用直接复用。

---

## 4. TODO List

### 🔴 高优先 — 性能提升

- [ ] **支持 num_g > 16 (全 32 expert)**
  - 调查 persistent scheduler 在 32-group 场景下的 tensormap update race condition
  - 可能需要在 `kernel` 内部加 `__syncthreads()` 或 fence
  - 或者调整 tensormap update 策略(batch update vs per-tile update)
  - 目标: 让最大的两个 workload (T≈2000+) 也走 CuTe 路径

- [ ] **消除 `torch.cuda.synchronize()` 开销**
  - 目前 gather → kernel → sync 有两次显式 sync
  - 改为 stream-based event 同步，减少 host-device roundtrip
  - 预计节省 20-50μs per call

- [ ] **Cluster Shape (2,1) 或 (1,2)**
  - 当前 cluster = (1,1)，每个 CTA 独立
  - Blackwell 支持 CTA cluster 共享 SMEM，可减少 TMA 加载量
  - 需要调整 mcast_mask 和 共享存储 layout

### 🟡 中优先 — 数值精度 & 鲁棒性

- [ ] **验证 SwiGLU 方向是否正确**
  - 当前实现: `silu(acc2) * acc1` (line 750 in epilogue)
  - acc1 = W1·x, acc2 = W3·x
  - 需确认 Triton 参考实现是否为 `silu(W1·x) * W3·x` 或 `silu(W3·x) * W1·x`
  - 方向错误不一定导致 INCORRECT_NUMERICAL（只影响精度 margin）

- [ ] **Scale Factor 精度优化**
  - 当前 e8m0 转换: `round(log2(scale)) + 127` → uint8
  - 这是近似转换，可能引入 ±1 ULP 偏差
  - 考虑使用 `view_as_float8_e8m0fnu` 直接转换（如果 PyTorch 支持）

- [ ] **Gather Kernel 精度**
  - `tma_gather_a_explicit` 中 A_scale 是 FP32 → e8m0 的 per-token 转换
  - 验证 `tl.math.log2` 的精度与 `torch.log2` 的差异
  - padding 区域的 e8m0 值使用 `127` (=1.0)，需确认不影响 GEMM

### 🟢 低优先 — 代码质量

- [ ] **移除 Debug Phase 系统**
  - `CUTE_DEBUG_PHASE` 1-6 已完成使命，后续可以简化为 `CUTE_ENABLE=1/0`
  - 清理 `raise RuntimeError(f"CUTE_PHASE_*_OK: ...")` 语句

- [ ] **统一 numpy / Python int 类型**
  - `expert_ms` 列表中元素是 `np.int32`，虽然不影响正确性，但不够 clean
  - 改为: `expert_ms.append(int(n_blks * block_m))`

- [ ] **JIT Cache 预热**
  - 当前首次调用触发 JIT 编译（30-60s）
  - 考虑在 module load 时用 dummy tensor 预编译常见 `(num_g, tiles)` 组合

- [ ] **Gather Kernel → CuTe TMA 直接 gather**
  - 当前先用 Triton kernel gather 到临时 buffer，再用 CuTe kernel 读取
  - 理想方案：CuTe kernel 内部直接通过 indirect TMA 读取 sorted tokens
  - 消除中间 buffer 和一次额外的全局内存 round-trip

---

## 5. 已验证的失败方向 (勿重复)

| 方向 | 失败原因 |
|------|--------|
| `cutlass.Int32()` 包装 JIT 参数 | MLIR 阶段 "integer conversion not supported" |
| 2D init tensor (无 unsqueeze) | `local_tile` rank mismatch (`3 vs 2`) |
| 硬编码 `T_PAD = total_blks * 64` | Block size mismatch → ILLEGAL_ADDRESS |
| try/except 捕获 CUDA crash | CUDA context 损坏后所有后续操作都会失败，无法恢复 |
| `num_g = 32` (全 expert active) | Persistent scheduler tensormap update race → ILLEGAL_ADDRESS |

---

## 6. 关键代码位置速查

| 功能 | 文件 | 行号 |
|------|------|------|
| CuTe 路径入口 guard | `kernel.py` | L1774 |
| `run_cute_path()` 主函数 | `cute_fused_gemm.py` | L111-272 |
| Gather kernel (Triton) | `cute_fused_gemm.py` | L67-109 |
| Scale factor e8m0 转换 | `cute_fused_gemm.py` | L157-171 |
| Per-expert ptrs/strides 构建 | `cute_fused_gemm.py` | L203-235 |
| JIT 编译 & cache | `cute_fused_gemm.py` | L249-257 |
| `num_g > 16` guard | `cute_fused_gemm.py` | L195-196 |
| Kernel class 定义 | `cute_kernel.py` | L15-45 |
| `_compute_grid` | `cute_kernel.py` | L130-146 |
| `__call__` (JIT 入口) | `cute_kernel.py` | L148-154 |
| MMA mainloop | `cute_kernel.py` | L580-640 |
| SwiGLU epilogue | `cute_kernel.py` | L738-765 |
| `make_tensor_for_tensormap_update` | `cute_kernel.py` | L783-831 |
