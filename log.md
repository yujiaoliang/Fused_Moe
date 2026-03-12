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

Workload b8f4f012...: PASSED | 0.174 ms | 65.85x speedup | abs_err=4.10e+03, rel_err=1.98e+01
  Workload e05c6c03...: PASSED | 0.175 ms | 63.62x speedup | abs_err=2.05e+03, rel_err=3.81e+00
  Workload 6230e838...: PASSED | 0.308 ms | 45.99x speedup | abs_err=4.10e+03, rel_err=9.27e+01
  Workload 8f1ff9f1...: PASSED | 0.661 ms | 24.06x speedup | abs_err=4.10e+03, rel_err=2.67e+02
  Workload 1a4c6ba1...: PASSED | 0.857 ms | 24.66x speedup | abs_err=3.60e+05, rel_err=3.60e+13
  Workload a7c2bcfd...: PASSED | 0.196 ms | 65.70x speedup | abs_err=4.10e+03, rel_err=1.17e+02
  Workload 2e69caee...: PASSED | 0.173 ms | 66.82x speedup | abs_err=4.10e+03, rel_err=1.57e+01
  Workload 8cba5890...: PASSED | 0.241 ms | 51.75x speedup | abs_err=2.05e+03, rel_err=1.52e+02
  Workload 5e8dc11c...: PASSED | 6.653 ms | 6.82x speedup | abs_err=5.57e+05, rel_err=5.32e+13
  Workload 58a34f27...: PASSED | 4.735 ms | 7.61x speedup | abs_err=5.61e+05, rel_err=4.51e+13
  Workload 5eadab1e...: PASSED | 0.262 ms | 53.06x speedup | abs_err=4.10e+03, rel_err=1.24e+03
  Workload eedc63b2...: PASSED | 0.291 ms | 47.42x speedup | abs_err=4.10e+03, rel_err=4.18e+01
  Workload e626d3e6...: PASSED | 0.348 ms | 44.28x speedup | abs_err=4.10e+03, rel_err=1.24e+03
  Workload 74d7ff04...: PASSED | 0.347 ms | 43.22x speedup | abs_err=4.10e+03, rel_err=4.28e+02
  Workload 4822167c...: PASSED | 0.346 ms | 43.83x speedup | abs_err=2.05e+03, rel_err=8.03e+02
  Workload 81955b1e...: PASSED | 0.343 ms | 42.82x speedup | abs_err=4.10e+03, rel_err=4.10e+02
  Workload 76010cb4...: PASSED | 0.322 ms | 44.78x speedup | abs_err=4.10e+03, rel_err=4.40e+02
  Workload fc378037...: PASSED | 0.343 ms | 43.09x speedup | abs_err=4.10e+03, rel_err=8.95e+03
  Workload f7d6ac7c...: PASSED | 0.274 ms | 48.93x speedup | abs_err=2.05e+03, rel_err=5.26e+01



5. cuda graph experiment

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

  Workload b8f4f012...: PASSED | 0.204 ms | 58.55x speedup | abs_err=4.10e+03, rel_err=7.46e+00
  Workload e05c6c03...: PASSED | 0.210 ms | 52.72x speedup | abs_err=2.05e+03, rel_err=3.59e+00
  Workload 6230e838...: PASSED | 0.308 ms | 45.53x speedup | abs_err=2.05e+03, rel_err=2.33e+02
  Workload 8f1ff9f1...: PASSED | 0.663 ms | 23.85x speedup | abs_err=4.10e+03, rel_err=1.93e+02
  Workload 1a4c6ba1...: PASSED | 0.854 ms | 24.61x speedup | abs_err=3.77e+05, rel_err=3.40e+13
  Workload a7c2bcfd...: PASSED | 0.210 ms | 60.53x speedup | abs_err=2.05e+03, rel_err=3.40e+01
  Workload 2e69caee...: PASSED | 0.210 ms | 54.94x speedup | abs_err=4.10e+03, rel_err=2.39e+01
  Workload 8cba5890...: PASSED | 0.240 ms | 51.90x speedup | abs_err=2.05e+03, rel_err=1.06e+02
  Workload 5e8dc11c...: PASSED | 6.571 ms | 6.88x speedup | abs_err=4.59e+05, rel_err=4.40e+13
  Workload 58a34f27...: PASSED | 4.810 ms | 7.50x speedup | abs_err=5.41e+05, rel_err=4.28e+13
  Workload 5eadab1e...: PASSED | 0.264 ms | 52.18x speedup | abs_err=4.10e+03, rel_err=2.34e+02
  Workload eedc63b2...: PASSED | 0.291 ms | 47.12x speedup | abs_err=4.10e+03, rel_err=1.23e+02
  Workload e626d3e6...: PASSED | 0.344 ms | 44.33x speedup | abs_err=4.10e+03, rel_err=4.21e+02
  Workload 74d7ff04...: PASSED | 0.345 ms | 43.33x speedup | abs_err=4.10e+03, rel_err=4.92e+01
  Workload 4822167c...: PASSED | 0.347 ms | 43.39x speedup | abs_err=4.10e+03, rel_err=1.28e+02
  Workload 81955b1e...: PASSED | 0.343 ms | 42.59x speedup | abs_err=4.10e+03, rel_err=1.26e+03
  Workload 76010cb4...: PASSED | 0.323 ms | 44.39x speedup | abs_err=4.10e+03, rel_err=1.09e+02
  Workload fc378037...: PASSED | 0.345 ms | 42.53x speedup | abs_err=4.10e+03, rel_err=5.94e+02
  Workload f7d6ac7c...: PASSED | 0.274 ms | 48.79x speedup | abs_err=2.05e+03, rel_err=1.19e+02

好像并没有相比上一步有进一步优化

6. block_expert_ids optimization

- Motivation: GEMM1/GEMM2 previously scanned all 32 experts and used `argmax` per program to map `pid_m -> expert_id`, which adds repeated control overhead.
- Change:
  - Added `block_expert_ids` workspace (`int32`, length `MAX_PID_M`).
  - `triton_sort_and_scatter_kernel` now emits `block_expert_ids` after computing `block_offsets`.
  - GEMM1/GEMM2 now directly load `expert_id = block_expert_ids[pid_m]`.
  - Removed per-program expert boundary scan (`b_start/b_end/argmax`) from both GEMMs.
- Host path updates:
  - Extended `_sort_cache` tuple and sort launch args with `block_expert_ids`.
  - Updated profiling helper (`yjl_ncu.py`) infer path to allocate/pass `block_expert_ids`.
- Expected effect:
  - Lower per-tile overhead in GEMM kernels, especially when `total_blocks * num_pid_n` is large.
——没有优化

### 2026-03-11 GEMM2 batched writeback experiment

- Objective: reduce atomic contention in `_fused_moe_gemm2_scatter_kernel` writeback.
- Change:
  - Added constexpr switch `BATCHED_WRITEBACK` (currently enabled at launch).
  - In batched mode, for each `BLOCK_M` tile:
    - detect duplicate `token_idx` rows,
    - aggregate duplicate rows locally,
    - only leader row performs one `atomic_add` to `output_fp32`.
  - `token_weights` load now uses `valid_mask` masked load (`other=0.0`) to avoid padded-row noise.
- Notes:
  - This is an experimental, higher-risk change: it trades additional local control/reduction work for potentially fewer atomic collisions.
  - Must validate by profiling GEMM2 kernel time and end-to-end wall time.
- Fix: Triton compilation error `unsupported tensor index: constexpr[...]` in GEMM2 batched writeback.
  - Replaced unsupported direct indexing (`token_idx[i]`, `out[i, :]`) with pointer scalar load + mask-based vector aggregation (`tl.sum(tl.where(...), axis=0)`).
—— 劣化
### 2026-03-11 GEMM2 two-stage writeback (slot + reduce)

- Objective: test structural writeback optimization by decoupling GEMM compute from output scatter atomics.
- Changes:
  - `_fused_moe_gemm2_scatter_kernel` now writes weighted GEMM2 results to contiguous slot buffer `SlotOut[num_padded, N]` (no atomics).
  - Added `_reduce_scatter_slots_kernel` to reduce `SlotOut` into `output_fp32[T, N]` using `token_ids`.
  - Host path updated:
    - buffer cache now stores `SlotOut`.
    - launch sequence becomes: GEMM2-to-slot -> reduce-scatter.
  - Profiler classification updated (`_reduce_scatter_slots_kernel` counted into GEMM2 bucket).
- Note:
  - This introduces one extra kernel launch and extra global-memory traffic.
  - Expected to help only if atomic contention was the dominant bottleneck.
