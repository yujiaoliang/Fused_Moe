# 笔记
## 1、环境准备
        
    * https://huggingface.co/datasets/flashinfer-ai/mlsys26-contest 似乎没有release



## 2、kernel
#### 2.1 moe逻辑梳理
* 入参

```
  "inputs": {
    "routing_logits": {
      "shape": [
        "seq_len",
        "num_experts"
      ],
      "dtype": "float32",
      "description": "Tensor of routing logits for expert selection"
    },
    "routing_bias": {
      "shape": [
        "num_experts"
      ],
      "dtype": "bfloat16",
      "description": "Bias tensor for routing. Pass all zeros for no bias."
    },
    "hidden_states": {
      "shape": [
        "seq_len",
        "hidden_size"
      ],
      "dtype": "float8_e4m3fn",
      "description": "Input hidden states tensor (FP8 quantized)"
    },
    "hidden_states_scale": {
      "shape": [
        "num_hidden_blocks",
        "seq_len"
      ],
      "dtype": "float32",
      "description": "Block-wise scaling factors for hidden states."
    },
    "gemm1_weights": {
      "shape": [
        "num_local_experts",
        "gemm1_out_size",
        "hidden_size"
      ],
      "dtype": "float8_e4m3fn",
      "description": "First GEMM weights for all local experts (gate and up projections)."
    },
    "gemm1_weights_scale": {
      "shape": [
        "num_local_experts",
        "num_gemm1_out_blocks",
        "num_hidden_blocks"
      ],
      "dtype": "float32",
      "description": "Block-wise scaling factors for first GEMM weights."
    },
    "gemm2_weights": {
      "shape": [
        "num_local_experts",
        "hidden_size",
        "intermediate_size"
      ],
      "dtype": "float8_e4m3fn",
      "description": "Second GEMM weights for all local experts (down projection)."
    },
    "gemm2_weights_scale": {
      "shape": [
        "num_local_experts",
        "num_hidden_blocks",
        "num_intermediate_blocks"
      ],
      "dtype": "float32",
      "description": "Block-wise scaling factors for second GEMM weights."
    },
    "local_expert_offset": {
      "shape": null,
      "dtype": "int32",
      "description": "Offset of local experts in global expert space."
    },
    "routed_scaling_factor": {
      "shape": null,
      "dtype": "float32",
      "description": "Scaling factor for routing weights."
    }
  },
  "outputs": {
    "output": {
      "shape": [
        "seq_len",
        "hidden_size"
      ],
      "dtype": "bfloat16",
      "description": "Final MoE output tensor"
    }
  },
  


```
* 其中seq len是一个变量，bench中的case看起来是会换不同的seq len测试这个kernel的性能

* 计算逻辑分几个主要的step：
    
    * 把hidden state、routing_logits等恢复为fp32
    * 对routing_logits sigmoid 和 +bias
    * 把256个expert分组，8组，每组32个专家
    * 组内选top2的计算分数和，按照这个分数选出top4的组。在选出的组中，每个token选全局top8的专家，也就是在4*32的专家中选top8
    * 将选中专家输出进行加权并归一化
    * 每个rank的本地专家做两次gemm和以此SwiGLU


### 0218_log
* 构造了一个脚本但是没有print出trace信息。打印如下：
  ```
  Loading solution...
Loaded: my-team-solution-v1 (moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048)

Running benchmark on Modal B200...
⠼ Loading images (1 containers initializing)... View app at https://modal.com/apps/jiaoliangyu968/main/ap-18ebzVjg2AzpsDShutting down worker
Shutting down worker
Shutting down worker
Shutting down worker

moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048:
Stopping app - local entrypoint completed.
✓ App completed. View run at https://modal.com/apps/jiaoliangyu968/main/ap-18ebzVjg2AzpsDJekZi9oV
```
没有trace信息，定位中。。。


### 0308_log

1. sort_and_scatter 只初始化有效 num_padded 区间
由“清空整个 MAX_PADDED”改为“仅清空 total_blocks * BLOCK_M”。
两个 GEMM kernel 增加基于 total_blocks 的早退。
autotune key 修正为实际参数
key=['num_padded', ...] 改为 key=['MAX_PID_M', ...]。—— 无明显优化

2. topk_idx_ptr 改为int32，因为最大值只有255。—— 无明显优化

3. 消除两个d2h。 —— 省掉了两个d2h大约14us的耗时


4. - Motivation: `num_padded` is highly sensitive to expert-wise token fragmentation. Fixed `BLOCK_M=64` can over-pad small batches.
- Change: introduce runtime selection for token block size:
  - `BLOCK_M_SMALL = 32`
  - `BLOCK_M_LARGE = 64`
  - `SMALL_BATCH_TOPK_TOKENS = 512`
  - rule: if `T * TOP_K <= 512`, use `BLOCK_M=32`; else use `BLOCK_M=64`.
- Implementation details:
  - Added `_select_block_m(num_topk_tokens)`.
  - Sort path now uses selected `block_m` (instead of fixed `BLOCK_M`).
  - `MAX_PADDED` / `MAX_PID_M` are computed from selected `block_m`.
  - Workspace cache key changed from `bkey=T` to `bkey=(T, block_m)`.
  - GEMM1/GEMM2 launches now pass `BLOCK_M=block_m` explicitly.
  - GEMM autotune configs no longer hardcode `BLOCK_M`; key now includes `BLOCK_M`.
- Expected effect:
  - Small batches: lower padding waste, reduce useless FLOPs.
  - Medium/large batches: keep existing 64-tile behavior for throughput.

### 2026-03-10 cuda graph experiment

- Added a CUDA Graph wrapper on top of kernel execution path in `solution/triton/kernel.py`.
- Refactor:
  - Original execution path moved to `_kernel_impl(...)`.
  - Public `kernel(...)` now tries graph replay when:
    - destination-passing mode is used (`output` is provided), and
    - all major tensor inputs are CUDA tensors.
- Graph cache key includes device, T, selected `block_m`, all major tensor pointers, output pointer, and scalar identity/value.
- On graph miss:
  - run one eager pass to allocate workspaces,
  - capture with `torch.cuda.CUDAGraph`,
  - cache and replay subsequently.
- Fallback behavior:
  - if graph preconditions are not met, run eager `_kernel_impl(...)` unchanged.

  moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048:
  Workload b8f4f012...: PASSED | 0.167 ms | 69.57x speedup | abs_err=2.05e+03, rel_err=6.34e+01
  Workload e05c6c03...: PASSED | 0.165 ms | 67.00x speedup | abs_err=2.05e+03, rel_err=8.84e+01
  Workload 6230e838...: PASSED | 0.311 ms | 44.23x speedup | abs_err=4.10e+03, rel_err=4.02e+01
  Workload 8f1ff9f1...: PASSED | 0.659 ms | 23.75x speedup | abs_err=4.10e+03, rel_err=3.61e+02
  Workload 1a4c6ba1...: PASSED | 0.855 ms | 24.21x speedup | abs_err=4.69e+05, rel_err=4.69e+13
  Workload a7c2bcfd...: PASSED | 0.198 ms | 62.73x speedup | abs_err=4.10e+03, rel_err=1.99e+01
  Workload 2e69caee...: PASSED | 0.162 ms | 69.62x speedup | abs_err=2.05e+03, rel_err=3.63e+01
  Workload 8cba5890...: PASSED | 0.209 ms | 58.28x speedup | abs_err=2.05e+03, rel_err=3.47e+02
  Workload 5e8dc11c...: PASSED | 6.631 ms | 6.76x speedup | abs_err=5.49e+05, rel_err=4.51e+13
  Workload 58a34f27...: PASSED | 4.728 ms | 7.53x speedup | abs_err=5.28e+05, rel_err=5.28e+13
  Workload 5eadab1e...: PASSED | 0.264 ms | 51.23x speedup | abs_err=4.10e+03, rel_err=9.76e+01
  Workload eedc63b2...: PASSED | 0.293 ms | 45.64x speedup | abs_err=4.10e+03, rel_err=3.83e+01
  Workload e626d3e6...: PASSED | 0.349 ms | 43.21x speedup | abs_err=4.10e+03, rel_err=2.84e+02
  Workload 74d7ff04...: PASSED | 0.348 ms | 42.01x speedup | abs_err=4.10e+03, rel_err=4.59e+02
  Workload 4822167c...: PASSED | 0.347 ms | 45.80x speedup | abs_err=4.10e+03, rel_err=4.99e+02
  Workload 81955b1e...: PASSED | 0.343 ms | 41.60x speedup | abs_err=2.05e+03, rel_err=1.18e+09
  Workload 76010cb4...: PASSED | 0.327 ms | 43.20x speedup | abs_err=4.10e+03, rel_err=4.56e+02
  Workload fc378037...: PASSED | 0.344 ms | 41.50x speedup | abs_err=4.10e+03, rel_err=4.71e+02
  Workload f7d6ac7c...: PASSED | 0.276 ms | 46.89x speedup | abs_err=4.10e+03, rel_err=5.15e+01



### 对T>4096进行3-phase优化，
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048:
  Workload b8f4f012...: PASSED | 0.172 ms | 68.13x speedup | abs_err=2.05e+03, rel_err=5.43e+00
  Workload e05c6c03...: PASSED | 0.128 ms | 86.69x speedup | abs_err=5.12e+02, rel_err=8.06e-03
  Workload 6230e838...: PASSED | 0.306 ms | 45.30x speedup | abs_err=4.10e+03, rel_err=4.85e+01
  Workload 8f1ff9f1...: PASSED | 0.661 ms | 23.93x speedup | abs_err=4.10e+03, rel_err=2.55e+02
  Workload 1a4c6ba1...: PASSED | 0.857 ms | 24.50x speedup | abs_err=3.36e+05, rel_err=3.36e+13
  Workload a7c2bcfd...: PASSED | 0.194 ms | 65.00x speedup | abs_err=2.05e+03, rel_err=2.28e+02
  Workload 2e69caee...: PASSED | 0.178 ms | 64.51x speedup | abs_err=4.10e+03, rel_err=3.92e+01
  Workload 8cba5890...: PASSED | 0.224 ms | 55.37x speedup | abs_err=2.05e+03, rel_err=5.78e+00
  Workload 5e8dc11c...: PASSED | 5.278 ms | 8.55x speedup | abs_err=5.37e+05, rel_err=4.49e+13
  Workload 58a34f27...: PASSED | 3.766 ms | 9.52x speedup | abs_err=5.49e+05, rel_err=5.49e+13
  Workload 5eadab1e...: PASSED | 0.259 ms | 53.04x speedup | abs_err=4.10e+03, rel_err=4.90e+01
  Workload eedc63b2...: PASSED | 0.290 ms | 46.78x speedup | abs_err=2.05e+03, rel_err=1.97e+02
  Workload e626d3e6...: PASSED | 0.345 ms | 44.34x speedup | abs_err=4.10e+03, rel_err=2.91e+02
  Workload 74d7ff04...: PASSED | 0.346 ms | 42.88x speedup | abs_err=4.10e+03, rel_err=6.60e+01
  Workload 4822167c...: PASSED | 0.347 ms | 43.05x speedup | abs_err=4.10e+03, rel_err=2.64e+02
  Workload 81955b1e...: PASSED | 0.343 ms | 42.18x speedup | abs_err=4.10e+03, rel_err=8.23e+02
  Workload 76010cb4...: PASSED | 0.316 ms | 44.91x speedup | abs_err=4.10e+03, rel_err=1.39e+02
  Workload fc378037...: PASSED | 0.343 ms | 42.30x speedup | abs_err=4.10e+03, rel_err=7.97e+01
  Workload f7d6ac7c...: PASSED | 0.270 ms | 49.18x speedup | abs_err=2.05e+03, rel_err=9.33e+02


### 改为不直接硬判断T，而是对num tile进行判断

  Workload b8f4f012...: PASSED | 0.175 ms | 66.21x speedup | abs_err=2.05e+03, rel_err=1.43e+01
  Workload e05c6c03...: PASSED | 0.127 ms | 86.69x speedup | abs_err=1.02e+03, rel_err=1.19e-02
  Workload 6230e838...: PASSED | 0.306 ms | 44.97x speedup | abs_err=2.05e+03, rel_err=5.95e+01
  Workload 8f1ff9f1...: PASSED | 0.661 ms | 23.91x speedup | abs_err=4.10e+03, rel_err=2.06e+02
  Workload 1a4c6ba1...: PASSED | 0.857 ms | 24.48x speedup | abs_err=3.32e+05, rel_err=3.09e+13
  Workload a7c2bcfd...: PASSED | 0.196 ms | 64.44x speedup | abs_err=4.10e+03, rel_err=2.45e+01
  Workload 2e69caee...: PASSED | 0.179 ms | 64.23x speedup | abs_err=4.10e+03, rel_err=2.26e+01
  Workload 8cba5890...: PASSED | 0.224 ms | 55.04x speedup | abs_err=4.10e+03, rel_err=2.59e+01
  Workload 5e8dc11c...: PASSED | 5.278 ms | 8.54x speedup | abs_err=5.28e+05, rel_err=4.92e+13
  Workload 58a34f27...: PASSED | 3.766 ms | 9.51x speedup | abs_err=4.94e+05, rel_err=4.51e+13
  Workload 5eadab1e...: PASSED | 0.259 ms | 52.98x speedup | abs_err=4.10e+03, rel_err=3.78e+02
  Workload eedc63b2...: PASSED | 0.290 ms | 46.68x speedup | abs_err=4.10e+03, rel_err=3.24e+02
  Workload e626d3e6...: PASSED | 0.345 ms | 44.27x speedup | abs_err=4.10e+03, rel_err=3.92e+02
  Workload 74d7ff04...: PASSED | 0.346 ms | 42.75x speedup | abs_err=4.10e+03, rel_err=1.18e+02
  Workload 4822167c...: PASSED | 0.347 ms | 42.93x speedup | abs_err=4.10e+03, rel_err=2.35e+03
  Workload 81955b1e...: PASSED | 0.343 ms | 42.06x speedup | abs_err=4.10e+03, rel_err=5.47e+02
  Workload 76010cb4...: PASSED | 0.316 ms | 44.83x speedup | abs_err=4.10e+03, rel_err=6.66e+01
  Workload fc378037...: PASSED | 0.343 ms | 42.30x speedup | abs_err=4.10e+03, rel_err=1.08e+02
  Workload f7d6ac7c...: PASSED | 0.270 ms | 49.08x speedup | abs_err=2.05e+03, rel_err=1.96e+01


### 2026-03-16 Round 15（commit `d2fdf14`）：Medium-T Bucket Specialization

- 背景：Round 14 之后 routing / sort / reduce 的固定开销已经压得比较低，主要剩下的是 `T≈32-128` 上 generic GEMM 的固定成本；上一版“按 num tile 判断”的方案把 mean 大致推到 `45.05x`，但中型 T 仍然没有真正吃满。
- 这次改动：
  - `BLOCK_M` 扩成四档：`16 / 32 / 64 / 128`
  - `32 <= T <= 64` 走 `_small_medium_*` kernel
  - `65 <= T <= 128` 走 `_medium_*` kernel
  - small-medium GEMM1 中 expert lookup 改成基于 `block_offsets_ptr` 的二分查找，避免每个 tile 做完整 32-way scan
- 结果摘要：
  - **Peak:** **106.65x**（`2e69caee`）
  - **Mean:** **55.77x**
  - **GMean:** **47.54x**
  - **50x+ Workloads:** **14 / 19**
  - **代表性提升:** `b8f4f012 66.21x -> 95.85x`，`8f1ff9f1 23.91x -> 48.09x`，`a7c2bcfd 64.44x -> 72.44x`，`2e69caee 64.23x -> 106.65x`
  - **Large-T 基本持平:** `58a34f27 9.51x -> 9.25x`，`5e8dc11c 8.54x -> 8.34x`
- 对比上一版（按 num tile 判断）：
  - `b8f4f012`: `66.21x -> 95.85x`
  - `8f1ff9f1`: `23.91x -> 48.09x`
  - `a7c2bcfd`: `64.44x -> 72.44x`
  - `2e69caee`: `64.23x -> 106.65x`

moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048:
  Workload b8f4f012...: PASSED | 0.112 ms | 95.85x speedup | abs_err=2.05e+03, rel_err=3.54e+00
  Workload e05c6c03...: PASSED | 0.106 ms | 96.83x speedup | abs_err=2.05e+03, rel_err=1.71e-02
  Workload 6230e838...: PASSED | 0.248 ms | 51.59x speedup | abs_err=4.10e+03, rel_err=1.03e+02
  Workload 8f1ff9f1...: PASSED | 0.307 ms | 48.09x speedup | abs_err=4.10e+03, rel_err=6.65e+02
  Workload 1a4c6ba1...: PASSED | 0.860 ms | 23.04x speedup | abs_err=3.46e+05, rel_err=3.13e+13
  Workload a7c2bcfd...: PASSED | 0.161 ms | 72.44x speedup | abs_err=4.10e+03, rel_err=5.94e+01
  Workload 2e69caee...: PASSED | 0.100 ms | 106.65x speedup | abs_err=4.10e+03, rel_err=1.12e+02
  Workload 8cba5890...: PASSED | 0.168 ms | 67.82x speedup | abs_err=2.05e+03, rel_err=8.77e+01
  Workload 5e8dc11c...: PASSED | 5.239 ms | 8.34x speedup | abs_err=4.96e+05, rel_err=4.34e+13
  Workload 58a34f27...: PASSED | 3.735 ms | 9.25x speedup | abs_err=5.53e+05, rel_err=5.53e+13
  Workload 5eadab1e...: PASSED | 0.206 ms | 61.82x speedup | abs_err=4.10e+03, rel_err=1.13e+02
  Workload eedc63b2...: PASSED | 0.228 ms | 55.24x speedup | abs_err=4.10e+03, rel_err=7.89e+02
  Workload e626d3e6...: PASSED | 0.271 ms | 52.39x speedup | abs_err=4.10e+03, rel_err=5.97e+01
  Workload 74d7ff04...: PASSED | 0.271 ms | 50.66x speedup | abs_err=4.10e+03, rel_err=1.97e+02
  Workload 4822167c...: PASSED | 0.275 ms | 49.96x speedup | abs_err=2.05e+03, rel_err=3.38e+08
  Workload 81955b1e...: PASSED | 0.264 ms | 50.93x speedup | abs_err=4.10e+03, rel_err=8.35e+01
  Workload 76010cb4...: PASSED | 0.256 ms | 51.48x speedup | abs_err=4.10e+03, rel_err=8.33e+02
  Workload fc378037...: PASSED | 0.261 ms | 51.64x speedup | abs_err=4.10e+03, rel_err=1.08e+02
  Workload f7d6ac7c...: PASSED | 0.220 ms | 55.52x speedup | abs_err=2.05e+03, rel_err=4.73e+01


### 2026-03-23 保留优化：large-T exact dispatch

#### 背景

base 版本里，sort 之后虽然已经能得到真实的 `total_blocks`，但后续 GEMM1/GEMM2 的 launch 仍然按 `MAX_PADDED // BLOCK_M` 这个上界发射。  
当 `T` 很大且 local expert 分布比较稀疏时，`MAX_PID_M` 会明显大于真实需要计算的 block 数，导致两个问题：

* GEMM1/GEMM2 存在明显 overlaunch，很多 program 只是在 kernel 内早退
* kernel bucket 仍然按 `T` 分桶，而不是按真实 workload 分桶，large-T 下容易错配

#### 修改内容

在 `solution/triton/kernel.py` 中只保留了一条最终有效的优化：

* 新增 `EXACT_WORKLOAD_DISPATCH_T_MIN = 4096`
* 新增 `_use_exact_workload_dispatch(t)`，仅在 `T >= 4096` 时启用 exact dispatch
* 新增 `_select_gemm_buckets_from_workload(t, block_m, total_blocks)`，对于 large-T 不再只按 `T` 选择 small/medium/general GEMM bucket，而是按真实 `total_blocks * block_m` 判断
* 在 `parallel_sort_and_scatter(...)` 之后，读取一次 `total_blocks.item()` 得到真实 `exact_pid_m`
* GEMM1/GEMM2 的 grid 从：

```python
MAX_PID_M * ceil_div(N, BLOCK_N)
```

改为：

```python
exact_pid_m * ceil_div(N, BLOCK_N)
```

* GEMM1/GEMM2 launch 时传入的 `MAX_PID_M` 也同步改成 `exact_pid_m`
* 如果 `exact_pid_m <= 0`，则直接 `output.zero_()` 返回

#### 为什么只对 large-T 开启

这条优化需要一次 `total_blocks.item()`，也就是一次 host 读标量。  
在 `T=901` 这类 workload 上，这个同步成本会吃掉 GPU 侧减少 overlaunch 的收益；但在 `T=11948/14107` 这类真正的大 workload 上，这个固定成本可以被摊薄，所以净收益为正。

因此最终策略是：

* `T < 4096`：保持 base 行为
* `T >= 4096`：启用 exact dispatch + workload-aware bucket

#### 实测效果

相对 `0321 base`，当前保留版本在大 T 上有稳定收益：

* `T=11948`：`3.790 ms -> 3.583 ms`，约 `-5.5%`
* `T=14107`：`5.308 ms -> 5.066 ms`，约 `-4.6%`
* `T=901`：基本持平，因此不对这类 case 开启

进一步看 kernel 级别，主要收益来自 GEMM2：

* `T=11948`：`GEMM2 1832.89 us -> 1606.74 us`
* `T=14107`：`GEMM2 2629.37 us -> 2374.67 us`

#### 结论

这是目前相对 base 唯一稳定、可复现、且不牺牲正确性的通用路径优化。  
后续优化应继续围绕 large-T 的真实 workload 调度与 GEMM2 主体展开，而不是再回到 small-T 特化或大而全的 fuse 路线。
moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048:
  Workload b8f4f012...: PASSED | 0.171 ms | 66.95x speedup | abs_err=2.05e+03, rel_err=8.19e+00
  Workload e05c6c03...: PASSED | 0.124 ms | 87.55x speedup | abs_err=2.56e+02, rel_err=7.19e-03
  Workload 6230e838...: PASSED | 0.250 ms | 54.41x speedup | abs_err=4.10e+03, rel_err=7.22e+01
  Workload 8f1ff9f1...: PASSED | 0.305 ms | 51.01x speedup | abs_err=4.10e+03, rel_err=5.83e+02
  Workload 1a4c6ba1...: PASSED | 0.855 ms | 24.19x speedup | abs_err=4.32e+05, rel_err=2.45e+13
  Workload a7c2bcfd...: PASSED | 0.177 ms | 69.96x speedup | abs_err=2.05e+03, rel_err=3.34e+03
  Workload 2e69caee...: PASSED | 0.173 ms | 65.08x speedup | abs_err=2.05e+03, rel_err=6.91e+01
  Workload 8cba5890...: PASSED | 0.175 ms | 69.17x speedup | abs_err=2.05e+03, rel_err=7.87e+01
  Workload 5e8dc11c...: PASSED | 4.983 ms | 9.00x speedup | abs_err=5.49e+05, rel_err=5.49e+13
  Workload 58a34f27...: PASSED | 3.516 ms | 10.10x speedup | abs_err=5.18e+05, rel_err=5.18e+13
  Workload 5eadab1e...: PASSED | 0.209 ms | 64.62x speedup | abs_err=4.10e+03, rel_err=2.04e+02
  Workload eedc63b2...: PASSED | 0.228 ms | 58.28x speedup | abs_err=4.10e+03, rel_err=3.01e+02
  Workload e626d3e6...: PASSED | 0.271 ms | 55.33x speedup | abs_err=4.10e+03, rel_err=2.34e+02
  Workload 74d7ff04...: PASSED | 0.269 ms | 53.89x speedup | abs_err=2.05e+03, rel_err=2.40e+02
  Workload 4822167c...: PASSED | 0.275 ms | 53.05x speedup | abs_err=4.10e+03, rel_err=2.98e+02
  Workload 81955b1e...: PASSED | 0.266 ms | 53.30x speedup | abs_err=4.10e+03, rel_err=1.21e+03
  Workload 76010cb4...: PASSED | 0.255 ms | 54.44x speedup | abs_err=4.10e+03, rel_err=8.51e+01
  Workload fc378037...: PASSED | 0.261 ms | 54.65x speedup | abs_err=4.10e+03, rel_err=4.72e+01
  Workload f7d6ac7c...: PASSED | 0.222 ms | 58.14x speedup | abs_err=2.05e+03, rel_err=1.17e+02
