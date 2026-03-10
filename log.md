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
