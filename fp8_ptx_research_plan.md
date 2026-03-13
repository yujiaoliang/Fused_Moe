# Research Plan: Dynamic FP8 Quantization in Triton on Blackwell (B200)

## 1. 现象描述 (Symptom Snapshot)

### 目标
将 GEMM1 (SwiGLU) 的输出（Intermediate 激活值）从 FP32 动态计算 Scale 并量化为 `float8_e4m3fn` (FP8)，从而满足 Blackwell/Hopper 架构上的完全满血状态：在 GEMM2 彻底激活纯 FP8xFP8 TensorCore 运算指令，预期可以将 GEMM2 算力吞吐上限再翻一倍。

### 遇到了什么致命错误
我们尝试了两种 Triton 层的转换方案，均告失败，且均指向了底层编译器问题：
1. **Triton 原生 Cast** (`.to(tl.float8e4nv)`)
2. **内联 PTX 汇编** (`tl.inline_asm_elementwise("cvt.rn.satfinite.e4m3x2.f32 ...")`)配合类型劫持

这引发了：
1. **编译器崩溃 (ICE, Internal Compiler Error)**: 在 B200 节点生成中间态 PTX 汇编时，Triton 发生崩溃，报错堆栈甚至出现了 `please share the reproducer above with Triton project.` 提示。这是因为 Triton LLVM 发射器产生了非法的寄存器捆绑导致 CUDA `ptxas` 汇编器拒收。
2. **灾难性数值崩溃**: 在偶尔成功组装的少部分 Tile 配置中，计算结果输出的绝对误差去到了惊人的 `abs=1.46e+04, rel=1.90e+03`，直接破坏了模型权重。

---

## 2. 根本原因病理分析 (Root Cause Hypothesis)

这不是单纯的 "数值计算越界"，而是深层编译器缺陷引发的一连串骨牌效应：

### 第一层：PTX 寄存器打包缺陷 (Register Packing Defect)
`cvt.rn.satfinite.e4m3x2.f32` 这条 PTX 指令的作用是将 **两个 32 bit 的浮点数 (FP32)** 压缩打包，并放到 **一个 16 bit 寄存器（包含两个 FP8）** 里面。但是，Triton 3.x 后端在推导张量的 `layout`（尤其是 Block 尺寸并非完美二次幂或者经过 Reshape/Broadcast 操作）时，可能分配出了连续的单独 16-bit virtual registers，但送给 PTX 的却是分散的 32-bit registers，这导致汇编语法错位。

### 第二层：Tile Memory Layout 错乱 (Memory Alignment)
当我们使用 `tl.store` 写入 `fp8` 张量时，对于 NVIDIA Tensor Core 来说，FP8 数据在 Shared Memory / Global Memory 中有极其苛刻的物理连续性要求（16-byte alignment 等）。Triton 尝试做优化掩盖，但我们在手动算 Dynamic Scale 时的 `tl.reshape` 或 Broadcast 操作让元素的连续性逻辑被破坏了，写入到显存的数据格式变成了不可读的"乱码"，导致 GEMM2 取回 FP8 进行 `tl.dot` 时读出的完全是一坨随机数，继而造成了 14000 的恐怖绝对误差。

### 第三层：FP8 e4m3 极限精度的脆弱性 (Numeric Fragility)
`e4m3` 格式最多只能表示到 `448.0`（如果发生上溢就会导致硬件直接生成 NaN 或者最大饱和值），且只有极其有限的尾数位数。即便编译器不崩溃，SwiGLU （具有激活剧烈放大的特质）在同一 Block 里的极大值也会极大拉高 scale，导致同一个 block 下的其他小数点在转 FP8 时全部下溢成了 0 (Underflow)，这叫 **量化挤压 (Quantization Squeeze)**，同样会报出巨额误差。

---

## 3. 远期攻坚计划 (Long-Term Research / Resolution Plan)

如果我们决心想要彻底克服这个问题（可能是冲击国际赛事巅峰所必须的最终武器），不能再在 700 行的业务代码里进行碰运气式的尝试。我们需要开启一段专门的系统底层联调旅程。

### Phase 1: 制作极致纯净的复现器 (Minimal Reproducer)并提交给 OpenAI
- 剥离复杂的 MoE GEMM 环境。编写一个仅仅 30 行代码的独立 Triton Kernel。
- 逻辑：输入一个 FP32 矩阵 -> 计算 max 并 broadcast scale -> 试图写入 `tl.float8e4nv` 矩阵。
- 验证：在本地或者 Modal B200 跑通这段小逻辑。如果它同样触发了 `ptxas` 崩溃或乱码，那这就是确凿无疑的 **Triton 官方 Bug**，直接携复现脚本在 OpenAI Triton 代码库提 Issue，这是对硬件系统社区的重要贡献。

### Phase 2: CUDA C++ 探针逆向工程 (Reverse Engineering via CUDA)
- 对于急切想要跑通业务的我们，没办法等上游发新版。我们需要手写一份原生 CUDA Kernel（用原生的 `__nv_cvt_float_to_fp8` Intrinsics）。
- 使用 `nvcc -keep -ptx` 抽出它的标准答案 PTX 汇编，观察 CUDA 官方是如何正确捆绑这几个寄存器的。
- 与 Triton 挂掉时的 PTX 输出对比。找到是哪一个 `mov` 或者是 `cvt.rn` 传参写错了。
- 尝试通过更为精确的 `tl.inline_asm_elementwise` 寄存器绑定描述符（比如 `"=h,f,f"` 手动传入成对寄存器）来强塞正确的汇编语句，强制劫持 Triton 编译链。

### Phase 3: 内存与数据流整形 (Memory Coalescing & Bitcasting)
- 为了解决“乱码读写误差（1.4e+04）”，我们**绝对不能**用高层语义的 `tl.store` 去直接存不定 layout 的 FP8。
- **降维打击**：在 Kernel 内部，把量化好的 2 个 FP8，在寄存器里用 `<< 8` 以及按位或 (`|`) 操作，硬拼凑成一个标准的 `int16`（甚至 4 个拼成 `int32`）。
- 然后将指针 `bitcast` 成 `int32` `tl.store` 写回主存。GEMM2 去取的时候也是取 `int32`，读取后手动 `bitcast` 退解包成 FP8。强制硬件不要做任何偷偷摸摸的 layout 转置，将内存的控制力 100% 攥在自己手里。

### Phase 4: SwiGLU 放大的隔离量化算法 (Outlier-Aware Fine-grained Quantization)
- 一旦程序不会崩溃且内存不出错了，面对数值精度问题，我们就不能在每个 Block 简单粗暴求 `max`：
- 在算 Scale 的同时，设定一个极值阈值。针对极大异常激活值（**Outliers**，通常只有 < 1% 的权重），将它们的索引和原始值保留在额外开辟的 FP32 稀疏张量缓冲里。
- 其余 99% 的数值（已被安全压缩）参与 GEMM2 FP8 张量计算；
- GEMM2 末端，使用 FP32 缓冲区针对那 1% 的 Outlier 再进行一次稀疏向量点乘补偿加成。
- 这样能完美在保留精度的前提下解锁 FP8 的理论峰值。
