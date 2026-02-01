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

