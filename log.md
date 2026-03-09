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


1. 去掉热路径里的 GPU→CPU 标量同步  
代码里有 `int(local_expert_offset)` 和 `float(routed_scaling_factor)`（[kernel.py:565](d:/mlsys2026/flashinfer-bench-starter-kit/solution/triton/kernel.py:565), [kernel.py:569](d:/mlsys2026/flashinfer-bench-starter-kit/solution/triton/kernel.py:569)）。  
把它们改成 device scalar 传入 Triton（pointer load），可进一步压低小 batch 的 CPU overhead。

2. `sort` 目前是单 CTA，改成多 CTA 两阶段  
现在 `triton_sort_and_scatter_kernel` 只 launch `(1,)`（[kernel.py:597](d:/mlsys2026/flashinfer-bench-starter-kit/solution/triton/kernel.py:597)），并在 kernel 内串行扫 `N=T*TOP_K`（[kernel.py:198](d:/mlsys2026/flashinfer-bench-starter-kit/solution/triton/kernel.py:198), [kernel.py:230](d:/mlsys2026/flashinfer-bench-starter-kit/solution/triton/kernel.py:230)）。  
可改成 `histogram -> scan -> scatter` 的多 block 版本，能更好覆盖 T=512/4096。

3. 预计算 `block -> expert` 映射，避免 GEMM 内反复 `argmax`  
GEMM1/GEMM2 每个 program 都在扫 32 个 expert 边界找 `expert_id`（[kernel.py:337](d:/mlsys2026/flashinfer-bench-starter-kit/solution/triton/kernel.py:337), [kernel.py:489](d:/mlsys2026/flashinfer-bench-starter-kit/solution/triton/kernel.py:489)）。  
在 sorting 阶段直接产出 `block_expert_ids`，GEMM 里直接 load，一般是低风险且稳定收益。

4. 修正 autotune key，让 GEMM2 真正按 T 选配置  
autotune key 写了 `num_padded`（[kernel.py:444](d:/mlsys2026/flashinfer-bench-starter-kit/solution/triton/kernel.py:444)），但 kernel 参数里没有这个字段（[kernel.py:448](d:/mlsys2026/flashinfer-bench-starter-kit/solution/triton/kernel.py:448)）。  
建议改成 `MAX_PID_M` 或显式传 `num_padded`，对应 README 里提到的 `T=512` GEMM2 回归问题。

5. 索引类型从 `int64` 降到 `int32/int16`  
`topk_idx`、`sorted_token_ids` 目前是 `int64`（[kernel.py:142](d:/mlsys2026/flashinfer-bench-starter-kit/solution/triton/kernel.py:142), [kernel.py:589](d:/mlsys2026/flashinfer-bench-starter-kit/solution/triton/kernel.py:589)），但 token/expert 范围很小。  
降位宽可减少 sorting + GEMM 的访存带宽压力。

6. `exp` 密集路径可尝试 fast sigmoid 近似  
routing 和 SwiGLU 都在用 `tl.exp`（[kernel.py:68](d:/mlsys2026/flashinfer-bench-starter-kit/solution/triton/kernel.py:68), [kernel.py:405](d:/mlsys2026/flashinfer-bench-starter-kit/solution/triton/kernel.py:405)）。  
可按 README/CUDA baseline 思路试 Pade 近似，重点验证误差门限。