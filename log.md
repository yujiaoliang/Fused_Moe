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


### 0326 尝试对gemm1_swiglu的kernel做进一步的优化

* 实验：把w1 和 w3 分别计算，避免双路acc的压力。
* 结论: 拿掉w3后，耗时降低接近一半，说明并没有严重的register bound

* 实验：把swiglu拿掉，看看swiglu是不是引入了太多开销
* 结论：实际上也没有，看起来swiglu不是瓶颈

* 实验：拆分full block和tail block
* 结论：对于类似T=80这样的case，基本没有full block，所以会变成全是碎的tail block，所以不是个优化方案

### 0328 继续对gemm相关的实验

* 实验：把32路scan expert id的方式换成 预计算和look up
* 结论：us级别的优化，几乎很像噪声

### 2026-04-02 保留优化：large-T single remapped GEMM1

#### 背景

`0328base` 之后，large-T 路径里最有价值的剩余空间不在 lookup，也不在 cluster，而在 `GEMM1` 的 padding / tile 形态。  
canonical large-T 路径里 `BLOCK_M=128`，对 `T=11948/14107` 这类 trace，`GEMM1` 仍有明显 padding 浪费；但如果把整条路径一起改小 tile，又会把 `GEMM2` 和整体调度一起打乱。

因此最终保留下来的方案是：**只对 large-T exact-dispatch 路径的 GEMM1 做 remap，GEMM2 和最终 reduce 保持 canonical 路径不变。**

#### 修改内容

在 `solution/triton/kernel.py` 中最终保留了下面这组改动：

* 新增 `_use_remapped_gemm1_large_t_override(t, block_m, use_exact_dispatch)`，仅在 `use_exact_dispatch && block_m == 128 && T > 128` 时启用
* 保留现有 routing 和 canonical sort / scatter
* 在 sort layout 阶段同时产出两套 offsets：
  * canonical `block_offsets / total_blocks`
  * remapped `remapped_block_offsets / remapped_total_blocks`
* 新增 `triton_sort_layout_with_remap_kernel(...)`，把 canonical / remapped layout 合并到一次 layout kernel 中
* `GEMM1` 改走 `_mapped_bucket_fused_moe_gemm1_swiglu_kernel`
  * remapped `BLOCK_M = 64`
  * token 仍然从 canonical `sorted_token_ids` 读取
  * 输出直接写回 canonical `Intermediate`
* 去掉显式 `row_map` buffer，改成在 kernel 内基于
  * `canonical_block_offsets_ptr`
  * `remapped_block_offsets_ptr`
  做算术映射
* `GEMM2` 继续走 canonical `_fused_moe_gemm2_kernel`
* 最终 `token_reduce` 保持原样

#### 实测效果

相对 `0328base`，这条优化在 large-T 上有稳定正收益：

* `T=11948`
  * `Wall: 3.447 ms -> 3.370 ms`，约 `-2.2%`
  * `GEMM1: 1.355 ms -> 1.268 ms`，约 `-6.4%`
  * `GEMM2: 1.603 ms -> 1.599 ms`，基本持平
* `T=14107`
  * `Wall: 4.937 ms -> 4.769 ms`，约 `-3.4%`
  * `GEMM1: 1.973 ms -> 1.819 ms`，约 `-7.8%`
  * `GEMM2: 2.340 ms -> 2.362 ms`，约 `+0.9%`

从 profiler 看，large-T 的 top GEMM1 kernel 已经切到 `_mapped_bucket_fused_moe_gemm1_swiglu_kernel`，说明收益来源比较干净，确实来自 remapped GEMM1 本体。

#### 结论

这条优化真正有效的点，不是 dual-bucket，也不是改 `GEMM2`，而是：

* **只重排 large-T GEMM1**
* **把 GEMM1 的 tile 从 canonical `BLOCK_M=128` 改成 remapped `BLOCK_M=64`**
* **GEMM2 / reduce 保持 canonical，不把收益再吃回去**

后续如果继续围绕这条线优化，优先级应该放在：

* 继续压 remapped GEMM1 辅助路径的固定成本
* 再评估 remapped GEMM1 的 autotune shortlist

而不是重新回到 dual-bucket、cluster、lookup-only 这些已经被证伪或收益很弱的方向。

### 2026-04-05 负优化：small/medium GEMM1-only `BLOCK_M=128` 实验（已回退）

#### 背景

曾考虑利用 B200 Tensor Core 更偏好大 tile 的特点，在 `32 <= T <= 128` 的 small/medium 区间尝试
`BLOCK_M=128`。但这条思路不能像普通 autotune 一样只加 config，因为当前实现里的 `BLOCK_M`
同时参与了 sort/layout 的数据组织，不能直接让 canonical `BLOCK_M=16/32/64` 的 layout 被
`GEMM1` 按 `128` 重新解释。

因此这次实际做的是一个 **最小可实现版本**：

* 保留 canonical sort/layout 和 canonical `GEMM2`
* 只给 `GEMM1` 单独构建一套 `BLOCK_M=128` 的 token layout
* 用 `row_map` 把 `GEMM1` 输出写回 canonical `Intermediate`

#### 修改内容

围绕 `solution/triton/kernel.py` 当时新增过下面这组实验性改动，后续已全部回退：

* `triton_count_tokens_from_canonical_layout_kernel(...)`
* `triton_scatter_canonical_to_remapped_gemm1_kernel(...)`
* `_row_mapped_fused_moe_gemm1_swiglu_kernel`
* `_use_small_medium_gemm1_block_m128_experiment(...)`
* `_kernel_small_medium_gemm1_block_m128_experiment(...)`

这组改动的目的，是在不动 `GEMM2` 的前提下，单独验证 `GEMM1 BLOCK_M=128` 是否能从更大的
Tensor Core tile 中获益。

#### 实测效果

结果非常差，而且不是小幅回退，而是明显负优化。相对 `0402base`：

* `T=32: 0.226 -> 1.112 ms`，约 `+392%`
* `T=52: 0.208 -> 0.774 ms`，约 `+272%`
* `T=59: 0.214 -> 1.106 ms`，约 `+417%`
* `T=80: 0.299 -> 1.461 ms`，约 `+389%`

命中实验的 `T=32/52-62/80` 共 11 个点，平均 `wall` 从约 `0.236 ms` 变成 `1.090 ms`，
约 `+361.5%`。

从 profiler 看，问题非常集中：

* 新增的辅助 kernel 本身很小
  * `triton_count_tokens_from_canonical_layout_kernel`: 约 `1.7~2.0 us`
  * `triton_scatter_canonical_to_remapped_gemm1_kernel`: 约 `2.1~2.3 us`
* 真正炸掉的是 `_row_mapped_fused_moe_gemm1_swiglu_kernel`
  * `T=32: 116 us -> 992 us`
### 2026-04-05 保留优化：fused routing + token-major sort/layout/scatter

#### 背景

`0402base` 之后，large-T 路径里 `GEMM1/GEMM2` 仍然是主要大头，但 `routing + sort/layout/scatter` 这段固定成本还存在一笔可继续压缩的开销。此前并行 sort 路径仍然依赖：

* routing 先产出 `topk_idx / topk_weights`
* 再单独生成 histogram / layout
* 最后走 tile-based scatter

这条链路本身没有错，但和 `1 program = 1 token` 的 routing 主体并不完全匹配，而且 sort/scatter 侧还有额外的 launch、重复读写和逐 item atomic 成本。

#### 修改内容

围绕 `solution/triton/kernel.py` 最终保留了下面这组修改：

* 新增 `_fused_routing_histogram_kernel(...)`
  * 保持 `1 program = 1 token`
  * routing 时顺手产出 `local_topk_idx`
  * 只对 local expert 保留 `0..31`，非本地记 `-1`
* 新增 `ds_routing_with_histogram(...)` wrapper，给 large-T 并行 sort 场景提供 fused routing 入口
* 新增 token-major sort/layout/scatter 路径：
  * `triton_token_layout_kernel(...)`
  * `triton_token_layout_with_remap_kernel(...)`
  * `triton_token_scatter_kernel(...)`
  * `token_sort_and_scatter(...)`
* `token_scatter` 从“逐 item atomic 分配”改成“program 内局部计数 + 按 expert 成批预留区间 + 回填”
  * 当前 `BLOCK_TOKENS = 8`
* host 侧新增 `_routing_hist_cache` 和 `_token_sort_cache`
* large-T 命中条件下：
  * routing 走 fused 版本
  * sort/layout/scatter 走 token-major 版本
  * `large-T single remapped GEMM1` 路径继续保留，不改 `GEMM2 / reduce`

#### 实测效果

相对 `0402base`，这组修改的端到端收益不算大，但已经是稳定正收益：

* 平均 `wall: 0.661211 ms -> 0.656579 ms`，约 `-0.70%`
* `T=11948`
  * `Wall: 3.370 ms -> 3.319 ms`，约 `-1.5%`
* `T=14107`
  * `Wall: 4.769 ms -> 4.730 ms`，约 `-0.8%`

如果只看新 feature 命中的 `routing / sort / scatter` 这一段局部固定成本，改善幅度会更明显：

* `triton_token_scatter_kernel`
  * `T=11948: 16.82 us -> 16.58 us`
  * `T=14107: 19.39 us -> 17.78 us`
* large-T 下的 routing + sort/layout/scatter 子路径，整体属于约 `2%-5%` 量级的 feature 改善
* 但端到端总收益会被 `GEMM1/GEMM2` 主体占比稀释，最终只体现为约 `0.7%` mean 改善

#### 结论

这条优化是有效的，而且已经可以保留进主线，但它的性质更接近：

* **压缩 large-T 下 routing / sort / scatter 的固定成本**
* **让 sort/layout/scatter 的组织方式真正和 `1 program = 1 token` 的 routing 对齐**
* **在不破坏 `GEMM1/GEMM2` 主体的前提下拿一笔稳定小收益**

后续如果继续围绕这条线优化，优先级应该放在：

* 继续压 `token_scatter` 的局部原子和回写开销
* 评估是否还有必要进一步精简 token-major layout 的辅助 metadata

而不应该再回到 tile-serial routing、global atomic histogram，或者把整条 routing 主体改成低并行度实现。

### 2026-04-05 负优化：small/medium + medium GEMM1 `BLOCK_K=256` 实验（已回退）

#### 背景

希望在 `GEMM1` 的 `K=7168` 上用更大的 `BLOCK_K=256` 减少 K-loop 次数。直观上，`BLOCK_K=128` 需要 `56` 轮，若能稳定切到 `256`，外层循环可以降到 `28` 轮，从而减少 barrier、指针跃迁和 loop 壳子的开销。

但当前 FP8 量化仍以 `QBLOCK=128` 为基本单位，因此 `BLOCK_K=256` 不能简单视为“一次 dot 覆盖 256 并只做一次 scale”。为了保证量化语义正确，`256` 的每一步实际上仍然需要拆成两个 `128` 子块分别做 `dot + scale` 再累加。

#### 修改内容

围绕 `small_medium` 和 `medium` 两条 `GEMM1` 路径做了几轮实验：

* 在 autotune config 中加入 `BLOCK_K=256` 候选，并将 `num_stages` 压到 `2` 以满足 shared memory 预算
* 第一版实现采用“外层 `BLOCK_K=256` + 内层 `128` 子块循环”的写法，保证 scale 仍按 `128` 对齐
* 第二版将 nested loop 改成“单层外循环 + 两段显式 `128` 展开”，目的是去掉额外的 `static_range` 壳子成本
* 后续又为部分 `256K` 配置补了 `num_stages=3`
* 最后又做了一个 `medium-only fixed probe`，只验证 upper-medium `GEMM1` 上 `BLOCK_K=256` kernel 本体是否能打赢

#### 实测效果

这条线没有在目标区间形成稳定收益。

* broad `BLOCK_K=256` autotune 版本在 `T=32..80` 这段仍然整体偏负
* 把 nested loop 改成单层循环 + 两段显式展开后，表现比最初版本略有恢复，但依然没有打赢 `0402base`
* 给 `256K` 配置补 `num_stages=3` 后，整体结果更差
  * 平均 `wall: 0.661 ms -> 0.679 ms`
  * 聚焦 `T=32,52-59,62,80` 的平均 `wall: 0.236 ms -> 0.253 ms`
  * 许多 case 的 `GEMM1` 几乎没变，但 `CPU overhead` 上升，更像 autotune / host 侧成本先把潜在收益吃掉了
* `medium-only fixed probe` 也明确失败
  * `T=80: wall 0.299 -> 0.317 ms`
  * `GEMM1 0.167 -> 0.186 ms`
  * top kernel 已切到 `_medium_block_k256_probe_fused_moe_gemm1_swiglu_kernel`，说明 probe 确实命中，但 kernel 本体仍然更慢

#### 结论

这轮实验说明，当前 FP8 `QBLOCK=128` 的量化/scale 语义下，`BLOCK_K=256` 并没有真正减少 `128` 粒度上的 `dot + scale` 工作量；它减少的主要只是外层循环壳子的那部分成本。

与之交换的是更高的资源占用、更浅的流水线选择空间，以及在 autotune / host 侧更高的额外成本，因此整体上没有形成正收益。`BLOCK_K=256` 这条 `GEMM1` 优化线已判断为负优化，并已从 `kernel.py` 中干净回退。

### 2026-04-05 Profiling：yjl_ncu.py GEMM1 vs GEMM2 时间分解

#### 工具

使用 `scripts/yjl_ncu.py` 在 Modal B200 上对所有 19 个 trace workload 做 per-kernel 时间分解（15 warmup + 80 iters）。

#### Summary Table (ms/iter)

| T | Wall | GEMM1 | GEMM2 | Reduce | Route | Sort | GEMM1% | GEMM2% | G1 TF | G2 TF |
|---|------|-------|-------|--------|-------|------|--------|--------|-------|-------|
| 1 | 0.105 | 0.032 | 0.055 | (fused) | 0.010 | 0.000 | 32% | **55%** | 88 | 26 |
| 7 | 0.159 | 0.048 | 0.030 | 0.003 | 0.007 | 0.003 | **51%** | 32% | 137 | 110 |
| 14 | 0.158 | 0.068 | 0.046 | 0.004 | 0.007 | 0.003 | **52%** | 35% | 165 | 121 |
| 15 | 0.160 | 0.046 | 0.029 | 0.003 | 0.007 | 0.002 | **51%** | 32% | 103 | 82 |
| 16 | 0.156 | 0.072 | 0.047 | 0.004 | 0.006 | 0.002 | **54%** | 35% | 169 | 130 |
| 32 | 0.228 | 0.121 | 0.077 | 0.004 | 0.007 | 0.002 | **56%** | 36% | 170 | 134 |
| 52 | 0.188 | 0.096 | 0.063 | 0.004 | 0.008 | 0.003 | **54%** | 35% | 166 | 128 |
| 56 | 0.252 | 0.134 | 0.087 | 0.005 | 0.008 | 0.003 | **56%** | 36% | 182 | 141 |
| 62 | 0.195 | 0.100 | 0.063 | 0.005 | 0.008 | 0.003 | **55%** | 35% | 169 | 134 |
| 80 | 0.290 | 0.166 | 0.088 | 0.005 | 0.008 | 0.003 | **61%** | 32% | 158 | 150 |
| 901 | 0.808 | 0.279 | 0.453 | 0.012 | 0.016 | 0.018 | 35% | **58%** | 486 | 149 |
| 11948 | 3.478 | 1.350 | 1.660 | 0.102 | 0.126 | 0.074 | 40% | **49%** | 535 | 217 |
| 14107 | 4.995 | 2.017 | 2.484 | 0.134 | 0.149 | 0.082 | 40% | **50%** | 533 | 216 |

#### 核心发现

* **小/中 T (7-80)**: GEMM1 是瓶颈 (51-61%)
* **T=901 和大 T**: GEMM2 是瓶颈 (49-58%)
* **Padding 浪费严重**: T=52 only 26 local rows but padded to 272 (10.5× waste)
* **Reduce kernel 很轻**: 2-5μs，不值得单独优化
* **GEMM1 TFLOPS 小 T 效率低**: 103-182T vs 峰值 1200T（只有 8-15%）
* **GEMM2 输入是 fp32**: 不能用 FP8 tensor core，是 GEMM2 TFLOPS 的根本限制

---

### 2026-04-05 保留优化：GEMM2 autotune 扩展

#### 背景

profiling 发现 GEMM2 在 T=901 占 58%、大 T 占 49-50% 时间，但 GEMM2 的 autotune configs 相对少，TFLOPS 只有 82-150T（小/中T）。扩展 autotune search space 是零风险的提升方式。

#### 修改内容

在 `solution/triton/kernel.py` 中，对三个 GEMM2 kernel 的 `@triton.autotune` configs 都做了扩展：

* `_fused_moe_gemm2_kernel`：从 24 个 configs 扩展到 38 个
* `_small_medium_fused_moe_gemm2_kernel`：从 16 个扩展到 27 个
* `_medium_fused_moe_gemm2_kernel`：从 12 个扩展到 23 个

新增 configs 分三类：

1. **低 warp 数 / 低延迟**：`num_warps=2/4, num_stages=2`，适合 K=2048 只有 16 次 K-loop 的短循环
2. **低 GROUP_M (1/2/4)**：适合 few-block workloads（T=901 只有 36 blocks）
3. **高 GROUP_M (16/64)**：增强 L2 cache weight tile 复用

#### AB-Test 实测效果（同 GPU 同环境对比）

```
Mean speedup: A=42.51x  B=43.40x  Delta=+2.1%
Summary: 8 improved, 0 regressed, 11 neutral (±2% threshold)
```

代表性提升：
* `b8f4f012 (T=7)`: `56.57x -> 59.72x (+5.6%)`
* `2e69caee (T=15)`: `55.20x -> 57.44x (+4.1%)`
* `a7c2bcfd (T=16)`: `51.79x -> 53.29x (+2.9%)`
* `74d7ff04 (T=57)`: `43.71x -> 44.89x (+2.7%)`

#### 结论

这是一个纯 autotune search space 扩展，没有改任何 kernel 逻辑。零退化、8 workload 显著提升。后续可以结合 profiling 继续针对 GEMM2 的 memory-bound 特性做更深层优化（如 Intermediate fp32→bf16）。

---

### 2026-04-18 CuTe DSL large-T hybrid path: 已验证有效修改

#### 背景

Pure Triton 路线在大 T 上主要瓶颈集中在 GEMM1/GEMM2。Triton 主体已经比较成熟，继续在 Triton kernel 内部小修收益有限，因此尝试用 `solution/python` 做 hybrid entry：非大 T 保持 pure Triton，大 T 使用 CuTe DSL grouped GEMM 做专用路径。

当前只把 CuTe 分支打开给：

* `T = 11948`
* `T = 14107`
* `block_m = 128`

非大 T 要继续走 pure Triton 路径，避免小 T/中 T 因 Python wrapper 或 CuTe lazy import 产生额外开销。

#### 已验证有效的实现点

1. `solution/python/kernel.py` 作为主入口，主体保持 pure Triton 代码，只在 `T in {11948, 14107} && block_m == 128` 时早跳到 `triton_impl.py` 的 hybrid/CuTe 路径。

2. `solution/python/triton_impl.py` 中的大 T 路径接入 CuTe GEMM2：`Intermediate` 为 fp16，GEMM2 weight 预先 dequant/cache 成 fp16，CuTe grouped GEMM2 输出 `expert_out` 使用 bf16，后续 reduce 从 bf16/fp32 expert_out load 后转 fp32 累加。

3. `solution/python/triton_impl.py` 中的大 T 路径接入 CuTe GEMM1：先用 `_dequantize_gemm1_sorted_a_kernel` 将 sorted hidden states 从 fp8 dequant 到 fp16，GEMM1 weight 预先 dequant/cache 成 fp16，CuTe grouped GEMM1 输出 raw W1/W3，再用 `_cute_gemm1_swiglu_epilogue_kernel` 做 `silu(W3) * W1 * 0.125`，写 fp16 `Intermediate`。

4. GEMM1 raw W1/W3 输出从 fp32 改为 bf16：`RAW_OUT_DTYPE = torch.bfloat16`，`c_dtype = cutlass.BFloat16`，epilogue 中 `tl.load(...).to(tl.float32)` 后再做 SiLU 和乘法。已确认精度通过当前 tolerance，不影响 correctness。

#### 最新验证数据

Profiling 文件：`ncu_profiler_yjl.txt`

环境：

* GPU: NVIDIA B200
* PyTorch: `2.11.0+cu130`
* Triton: `3.6.0`
* Solution spec: `language=python`, `entry=kernel.py::kernel`, `source_count=7`

Runtime status 确认命中 CuTe：

* `last_active_impl = hybrid_cute_large_t`
* `triton_gemm1_backend = cute_gemm1_mma_predequant_t11948 / t14107`
* `triton_gemm2_backend = cute_gemm2_mma_t11948 / t14107`
* `cute_gemm1_mma_viability = grouped_fp16_predequant_w13_bf16_raw_large_t`
* `cute_gemm2_mma_viability = grouped_fp16_prepacked_b_bf16_out_large_t`

当前结果：

| T | Wall | CUDA | GEMM1 | GEMM2 | Route | G1 TFLOPS | G2 TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11948 | 1.586 ms | 1.513 ms | 0.746 ms | 0.349 ms | 0.135 ms | 966.6T | 1034.3T |
| 14107 | 2.202 ms | 2.160 ms | 1.140 ms | 0.505 ms | 0.171 ms | 942.6T | 1064.0T |

对比 `ncu_profiler_yjl_0415base.txt` pure Triton：

| T | 0415 pure Triton Wall | CuTe bf16 raw Wall | Wall 改善 |
|---:|---:|---:|---:|
| 11948 | 2.451 ms | 1.586 ms | 约 35.3% |
| 14107 | 3.501 ms | 2.202 ms | 约 37.1% |

对比 CuTe GEMM1 fp32 raw：

| T | fp32 raw Wall | bf16 raw Wall | Wall 改善 |
|---:|---:|---:|---:|
| 11948 | 1.864 ms | 1.586 ms | 约 14.9% |
| 14107 | 2.317 ms | 2.202 ms | 约 5.0% |

#### 结论

CuTe DSL 对大 T 是明确有效的。GEMM1 从 pure Triton 的约 540T 提升到约 940-970T，GEMM2 也通过 fp16 prepacked B + bf16 expert_out 稳定达到约 1000T。当前最有价值的 hybrid 结构是：

`pure Triton routing/sort/reduce scaffold + CuTe GEMM1 + CuTe GEMM2 + Triton epilogue/reduce`

这条路径已经在 `T=11948` 和 `T=14107` 上同时获得明显 wall time 收益，并且 bf16 raw 的精度已确认可接受。

#### 已验证有效：metadata host sync 合并

最新 profiler 已确认这条修改有效。修改内容：

* 原来 CuTe large-T 路径中存在 3 次 host 回读/同步：`total_blocks.item()`、GEMM1 metadata 的 `block_offsets.detach().cpu().tolist()`、GEMM2 metadata 的 `block_offsets.detach().cpu().tolist()`。
* 现在在 `triton_impl.py` 中统一执行一次 `block_offsets.detach().cpu().tolist()`，然后把同一份 `block_offsets_host` 传给 CuTe GEMM1/GEMM2 metadata。
* `exact_pid_m` 改为使用 `block_offsets_host[-1]`，等价于 `total_blocks`，计算逻辑保持不变。

验证结果：

| T | bf16 raw baseline Wall | metadata sync 合并后 Wall | Wall 改善 |
|---:|---:|---:|---:|
| 11948 | 1.586 ms | 1.548 ms | 约 2.4% |
| 14107 | 2.202 ms | 2.168 ms | 约 1.5% |

CPU/API 侧证据：

| T | 优化前 `cudaMemcpyAsync` | 优化后 `cudaMemcpyAsync` |
|---:|---:|---:|
| 11948 | x3 | x1 |
| 14107 | x3 | x1 |

最新结果：

| T | Wall | CUDA | CPU overhead | GEMM1 | GEMM2 | G1 TFLOPS | G2 TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11948 | 1.548 ms | 1.514 ms | 0.034 ms | 0.748 ms | 0.357 ms | 964.8T | 1011.8T |
| 14107 | 2.168 ms | 2.199 ms | 0.000 ms | 1.175 ms | 0.514 ms | 915.1T | 1044.7T |

对比 `ncu_profiler_yjl_0415base.txt` pure Triton：

| T | 0415 pure Triton Wall | 当前 CuTe large-T Wall | 总改善 |
|---:|---:|---:|---:|
| 11948 | 2.451 ms | 1.548 ms | 约 36.8% |
| 14107 | 3.501 ms | 2.168 ms | 约 38.1% |

结论：metadata host-sync 合并是小幅但稳定的正收益，主要体现在 CPU/API 同步开销下降。大 T CuTe 路径当前已验证有效的组合为：CuTe GEMM1 + bf16 raw + CuTe GEMM2 bf16 expert_out + metadata host sync 合并。

---

### 2026-04-18 T=901 CuTe / fused reduce 实验：已验证不保留

#### 背景

`T=901` 是一个典型的中等 T / 碎 block workload：

* `block_m = 64`
* `total_blocks = 36`
* `num_padded = 2304`
* `num_local_rows = 1331`
* `tail_rows = 973`
* `tail_experts = 32`

0415 pure Triton baseline 中，`T=901` 已经有专门的 `_fused_moe_gemm2_t901_kernel`，整体表现为：

| T | Wall | CUDA | GEMM1 | GEMM2 | Route | Sort |
|---:|---:|---:|---:|---:|---:|---:|
| 901 | 0.531 ms | 0.524 ms | 0.273 ms | 0.196 ms | 0.016 ms | 0.019 ms |

目标是验证：能否用 CuTe DSL 或 fused reduce 把 901 的碎 block GEMM2 进一步拉高。

#### 实验 1：Grouped CuTe GEMM2 for T=901

实现方式：

* 将 `T=901, block_m=64` 加入 CuTe GEMM2 target。
* GEMM1 仍走 Triton。
* GEMM2 走 CuTe grouped GEMM2，输出 bf16 `expert_out`。
* 后续使用 `_token_reduce_weighted_kernel` 做 weighted reduce。

结果：

| T | Wall | GEMM1 | GEMM2 | Route / Reduce | 结论 |
|---:|---:|---:|---:|---:|---|
| 901 | 0.536 ms | 0.274 ms | 0.168 ms | 0.027 ms | GEMM2 单 kernel 更快，但端到端小亏 |

分析：

* CuTe GEMM2 本体从 `0.196 ms -> 0.168 ms`，单看 GEMM2 快约 14%。
* 但 CuTe grouped path 引入 metadata / host-device copy / sync 固定成本。
* CuTe GEMM2 输出是 unweighted，额外需要 `_token_reduce_weighted_kernel`。
* 对 901 这种规模，固定成本和额外 reduce 吃掉了 GEMM2 本体收益。

结论：**Grouped CuTe GEMM2 不适合直接推广到 T=901。**

#### 实验 2：T=901 atomic direct reduce

实现方式：

* GEMM2 仍按 sorted expert rows 计算。
* 不再写 `expert_out`。
* 计算后直接 `atomic_add` 到 fp32 token accumulator。
* 最后 cast fp32 accumulator 到 bf16 output。

结果：

| T | Wall | 主要 kernel | 结论 |
|---:|---:|---|---|
| 901 | 0.558 ms | `_fused_moe_gemm2_t901_atomic_reduce_kernel`: 226 us | 比 pure Triton 更慢 |

分析：

* 虽然省掉了 `_token_reduce_kernel`，但 atomic reduce 让 GEMM2 kernel 从 `196 us` 变成约 `226 us`。
* 还额外需要 fp32 accumulator 清零和 bf16 cast。
* 对 901 来说，atomic 冲突和额外 memops 不划算。

结论：**Atomic direct reduce 路线不保留。**

#### 实验 3：T=901 token-centric non-atomic direct reduce

实现方式：

* 每个 program 负责一个 `token_id + N tile`。
* program 内循环 `TOP_K=8` 个 routed slot。
* 每个 slot 从 `scatter_map` 找到 sorted position，再做 GEMM2 dot，乘 routing weight 后累加。
* 直接写最终 bf16 output，无 `expert_out`、无 token_reduce、无 atomic。

结果：

| T | Wall | 主要 kernel | 结论 |
|---:|---:|---|---|
| 901 | 16.397 ms | `_fused_moe_gemm2_t901_token_reduce_kernel`: 16116 us | 完全不可用 |

分析：

* 该方案避免了 atomic 和 reduce kernel，但把 GEMM2 的矩阵结构彻底打散。
* 每个 token/slot 都在做细粒度 vector dot，几乎无法利用 tensor core / grouped GEMM 的 tile 复用。
* B 权重访问也变成 per-token/per-slot 的动态 expert gather，访存和调度都很差。

结论：**Token-centric non-atomic fused reduce 路线不保留。**

#### 总结

`T=901` 的三条 CuTe / fused reduce 路线都不如现有 pure Triton 专用路径：

| 路线 | Wall | 相对 0415 pure Triton |
|---|---:|---:|
| 0415 pure Triton `_fused_moe_gemm2_t901_kernel` | 0.531 ms | baseline |
| grouped CuTe GEMM2 + weighted reduce | 0.536 ms | 慢约 0.9% |
| atomic direct reduce | 0.558 ms | 慢约 5.1% |
| token-centric non-atomic direct reduce | 16.397 ms | 完全失败 |

当前结论：

* `T=901` 应回退并保持 pure Triton T901 专用 GEMM2。
* CuTe grouped GEMM 的固定 metadata 成本只有在 `T=11948/14107` 这类大 T 上才能摊薄。
* 中小 T 的优化不应照搬 large-T grouped CuTe 方案。
* 若未来继续尝试中小 T，必须保持 GEMM tile 结构，不能把 GEMM2 改成 token-centric scalar/vector dot。

#### 实验 4：T=901 CuTe GEMM1-only

实现方式：

* 新增独立 `cute_gemm1_t901_*` 实验通路，只在 `T=901 && block_m=64` 时启用。
* GEMM1 拆成三段：sorted hidden states fp8 dequant 到 fp16、CuTe grouped GEMM1 输出 bf16 raw W1/W3、Triton epilogue 做 SwiGLU 并写 fp16 Intermediate。
* GEMM2、token reduce、T901 Triton GEMM2 专用 kernel 保持原逻辑不变。

结果：

| T | Wall | CuTe GEMM1 main | dequant | epilogue | GEMM2 | 结论 |
|---:|---:|---:|---:|---:|---:|---|
| 901 | 0.653 ms | 308.84 us | 9.84 us | 6.18 us | 205.18 us | 比 pure Triton baseline 明显更慢 |

对比 baseline：

| 路线 | Wall | 相对 0415 pure Triton |
|---|---:|---:|
| 0415 pure Triton `_fused_moe_gemm2_t901_kernel` | 0.531 ms | baseline |
| CuTe GEMM1-only + Triton GEMM2 | 0.653 ms | 慢约 23% |

分析：

* 该路径确实进入 `hybrid_cute_t901_gemm1_only`，`triton_gemm1_backend = cute_gemm1_t901_predequant_t901`。
* Profiler 中 CuTe GEMM1 main kernel 被归类到 `[gemm2]`，但 kernel 名为 `GroupedGemm1T901`，实际是 GEMM1 主体。
* GEMM1-only 拆分后带来了额外 kernel launch、metadata host read / `cudaMemcpyAsync`，且 CuTe grouped GEMM1 本体在 T=901 的碎 block 规模下没有打赢原 Triton GEMM1。
* 该实验不保留，回退到原来的 901 pure Triton 路径。

结论：**T=901 CuTe GEMM1-only 路线失败，不保留。**
