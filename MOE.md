# Phase 10: Practical Optimization Plan (Post-65× Baseline)

> **起点:** 65.71× peak, T=8192 延迟 4.96ms, 19/19 PASSED
> **环境:** Modal B200, Triton 3.6, PyTorch 2.10.0+cu128, 无 nvcc
> **截止:** 2026-04-24 (26 天)

> ⚠️ **红线规则**: 以下方向全部已验证失败，严禁重复尝试：
> - Split-K GEMM1 (SwiGLU 解融合带宽开销 > SM 利用率收益)
> - Persistent Kernel / Static Stride (TMA 描述符在动态循环中损坏)
> - warp_specialize=True (Modal Triton 3.6 MLIR crash)
> - TMA make_tensor_descriptor (Modal 环境不支持)
> - bf16 intermediate / FP8 GEMM2 A-side (精度失败)
> - num_stages=1 (编译器挂起)
> - Branchless predicate elimination (带宽退化 -5.9%)

---

## Direction 1: Fused Routing + Histogram 单 Kernel
**优先级: 🔴 高 | 预期收益: T=4096 全局 +3~5% | 风险: 低**

### 现状问题
Routing (11.6%) + Sort (12.4%) 合计占 T=4096 时间的 ~24%。当前流程：
```
routing_kernel → parallel_histogram → prefix_sum → scatter
```
4 次 kernel launch，中间经过 global memory 传递 topk_ids, topk_weights, histogram。

### 方案
创建 `_fused_routing_histogram_kernel`，每个 program 处理一个 token：

```python
@triton.jit
def _fused_routing_histogram_kernel(
    logits_ptr, bias_ptr,       # 输入
    topk_ids_ptr, topk_weights_ptr,  # 输出: routing 结果
    expert_counts_ptr,          # 输出: 32-element histogram (atomic)
    T, E: tl.constexpr, TOP_K: tl.constexpr, NUM_GROUPS: tl.constexpr,
    ...
):
    pid = tl.program_id(0)  # 每个 program = 1 token
    
    # 1. Load 全部 256 logits 到 registers (256 × fp32 = 1KB)
    offs_e = tl.arange(0, E)  # E=256, constexpr
    logits = tl.load(logits_ptr + pid * E + offs_e)
    bias = tl.load(bias_ptr + offs_e)
    
    # 2. Sigmoid + bias (in-register)
    scores = tl.sigmoid(logits) + bias
    
    # 3. Group-level top-2 selection (8 groups × 32 experts)
    #    reshape to [8, 32], per-group top-2
    #    → 16 candidate experts
    
    # 4. Global top-8 from 16 candidates
    #    register-level comparison network
    
    # 5. Normalize weights
    #    sum selected weights, divide
    
    # 6. Store topk_ids, topk_weights
    tl.store(topk_ids_ptr + pid * TOP_K + tl.arange(0, TOP_K), selected_ids)
    tl.store(topk_weights_ptr + pid * TOP_K + tl.arange(0, TOP_K), norm_weights)
    
    # 7. Atomic histogram (FREE byproduct!)
    for k in range(TOP_K):
        tl.atomic_add(expert_counts_ptr + selected_ids[k], 1)
```

### 收益分析
- 消除 routing→histogram 之间的 global memory round-trip (~T×8 int32 writes + reads)
- 减少 2 次 kernel launch (~10-20μs each)
- histogram 是 routing 的"副产品"，zero extra cost
- Sort 的后半段 (prefix_sum → scatter) 仍用现有 kernel

### 验证
- 正确性: 对比 reference routing output
- A/B test: 替换 routing + histogram 两个 kernel

---

## Direction 2: GEMM1 BLOCK_M=128 + Masking (Tensor Core 满载)
**优先级: 🔴 高 | 预期收益: Medium-T +5~15% | 风险: 中**

### 现状问题
T=64 时每个 expert 平均 ~8 tokens。当前 BLOCK_M=32/64，但 B200 的 tcgen05.mma
是 128×128 systolic array。BLOCK_M=64 浪费 ~75% tensor core throughput。

### 方案
在 `_small_medium` 和 `_medium` kernel 的 autotune config 中增加 BLOCK_M=128 选项：

```python
# 新增 config (不删除现有的！Exp-C1 证明了删 config 会退化)
triton.Config({
    'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128,
    'GROUP_M': 32, 'num_stages': 3, 'num_warps': 4
}),
triton.Config({
    'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128,
    'GROUP_M': 64, 'num_stages': 4, 'num_warps': 4
}),
```

Kernel 内部已有 `m_mask` 逻辑处理不足 BLOCK_M 的情况，所以代码改动极小——
只需要确保 grid launch 正确计算 `ceil(num_tokens / BLOCK_M)`。

### 关键点
- **不需要任何代码变更**，只加 autotune config
- Autotuner 会自动判断 BLOCK_M=128 是否比 64/32 快
- 对 T=64 场景：expert 有 8 tokens → 1 个 128-tile (120 rows masked)
  - 看起来浪费，但 tcgen05.mma 处理 masked rows 无额外延迟
  - 相比 BLOCK_M=32 需要 1 个 tile 但 tensor core 仅 25% 利用率

### 验证
- 正确性: 现有 m_mask 逻辑应自动处理
- A/B test: 只加 config，autotuner 决定是否选中

---

## Direction 3: BLOCK_K=256 减半 K-loop 迭代
**优先级: 🟡 中 | 预期收益: GEMM1 +2~5% | 风险: 低**

### 现状问题
GEMM1 K=7168, 当前 BLOCK_K=128 → 56 次 K-loop 迭代。
每次迭代有 TMA load、barrier sync、scale multiply 的固定开销。

### 方案
新增 BLOCK_K=256 autotune config：

```python
triton.Config({
    'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 256,
    'GROUP_M': 32, 'num_stages': 2, 'num_warps': 4
}),
```

### SMEM 预算分析
- BLOCK_K=256, FP8: A tile = 128×256 = 32KB, B tile = 256×128 = 32KB
- GEMM1 需要 W1 和 W3: 2 × 32KB B tiles = 64KB
- A tile: 32KB
- 每 stage: 32 + 64 = 96KB
- num_stages=2: 192KB ← 刚好在 B200 228KB SMEM 预算内
- num_stages=3: 288KB ← 超出！所以 BLOCK_K=256 限制 num_stages≤2

### 收益
- K-loop 从 56 次降到 28 次
- 循环开销 (branch, pointer update, barrier) 减半
- 代价: pipeline depth 从 3→2，但每次加载量翻倍，可能更好隐藏延迟

### 验证
- A/B test: 让 autotuner 在 BLOCK_K=128 和 256 之间选择

---

## Direction 4: GEMM2 Atomic Contention 优化 (Tile 排序)
**优先级: 🟡 中 | 预期收益: Large-T +1~3% | 风险: 低**

### 现状问题
Exp-H1 成功融合了 GEMM2 + FP32 atomic reduce。但 TOP_K=8 意味着
每个 output token 最多有 8 个 CTA 并发 atomic_add 到同一行。
在 T=8192 时，某些 hot tokens 可能遇到 L2 atomic 排队。

### 方案
在 dispatch 时，对 tile 按 `(output_token_row % NUM_WAVES, expert_id)` 排序，
使得写入同一 output row 的 tile 被分散到不同的 execution wave：

```python
# Python-side, before kernel launch
tile_keys = output_token_row[tile_ids] % NUM_WAVES  # NUM_WAVES = 4
sorted_idx = torch.argsort(tile_keys, stable=True)
# 重排 tile dispatch order
```

### 收益
- 降低 L2 atomic bank conflict
- 无 kernel 代码改动，只改 host-side dispatch order

### 验证
- A/B test on Large-T workloads (T≥4096)

---

## Direction 5: Autotuner Cache 持久化
**优先级: 🟢 低 (但零风险) | 预期收益: benchmark 速度 +30~50% | 风险: 零**

### 现状问题
每次 Modal run 都重新 autotune 所有 kernel × 所有 config。
对 19 个 workload × ~20 configs × 4 kernels = ~1520 次 trial。

### 方案
```python
# test_modal.py 中
import os
os.environ["TRITON_CACHE_DIR"] = "/data/triton-cache"
# Modal volume 已挂载在 /data
```

或者更精确地，使用 `triton.autotune` 的 `restore_value` 和手动 cache：

```python
# 第一次 run 后导出
import json
cache = {key: best_config for key, best_config in autotuner.cache.items()}
json.dump(cache, open("/data/autotune_cache.json", "w"))

# 后续 run 预加载
if os.path.exists("/data/autotune_cache.json"):
    preloaded = json.load(open("/data/autotune_cache.json"))
    # inject into autotuner
```

### 收益
- A/B test 从 ~10min 降到 ~3min
- 更多实验迭代时间
- 零性能风险（不影响 kernel 本身）

---

## 执行顺序

```
Week 1 (Mar 30 - Apr 5):
  ├─ Day 1-2: Direction 5 (Autotuner cache) — 立即落地，加速后续实验
  ├─ Day 3-4: Direction 2 (BLOCK_M=128 configs) — 只加 config，快速 A/B
  └─ Day 5-7: Direction 3 (BLOCK_K=256 configs) — 同上

Week 2 (Apr 6 - Apr 12):
  └─ Full week: Direction 1 (Fused Routing+Histogram) — 需要写新 kernel

Week 3 (Apr 13 - Apr 19):
  ├─ Direction 4 (Tile 排序) — 如果 Exp-H1 在 Large-T 有 atomic 瓶颈
  └─ Buffer: 任何方向的 bugfix / follow-up tuning

Week 4 (Apr 20 - Apr 24):
  └─ 冻结代码，最终提交，写 writeup
```

## 预期总收益

| Direction | 目标 T 范围 | 预期 Δ | 风险 |
|-----------|-----------|--------|------|
| D1: Fused Routing+Histogram | T≥256 | +3~5% | 低 |
| D2: BLOCK_M=128 | T≤128 | +5~15% | 中 |
| D3: BLOCK_K=256 | All T | +2~5% | 低 |
| D4: Tile 排序 | T≥4096 | +1~3% | 低 |
| D5: Autotuner cache | N/A | 迭代加速 | 零 |

**乐观估计:** 叠加后 mean speedup +8~15%，从 ~45× mean 推到 ~50× mean
**保守估计:** +3~5% mean，peak 从 65× 推到 ~68-70×