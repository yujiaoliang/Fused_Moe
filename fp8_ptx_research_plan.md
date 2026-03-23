# Research Plan: Dynamic FP8 Quantization in Triton on Blackwell (B200)

## 0. 2026-03-16 重新尝试：最新实测结论

这次没有再直接回到主线 `kernel.py` 里碰运气，而是先做了两个**最小复现**：

- `D:/Research/mlsys_note/scripts/fp8_ptx_repro.py`
  - Triton 高层 cast：`x.to(tl.float8e4nv)`
  - Triton inline PTX pack：`cvt.rn.satfinite.e4m3x2.f32`
  - Triton block-scale + inline PTX pack（更接近原始 dynamic quantization）
- `D:/Research/mlsys_note/scripts/fp8_ptx_nvcc_probe.cu`
  - CUDA 官方 `__nv_cvt_float2_to_fp8x2(..., __NV_E4M3)` 参考实现

### 这次已经确认的事实

1. **`nvcc` 官方 PTX 路径没问题。**
   用 `nvcc -ptx -arch=sm_100 scripts/fp8_ptx_nvcc_probe.cu` 能稳定得到：
   ```ptx
   cvt.rn.satfinite.e4m3x2.f32 %rs1, %f2, %f1;
   ```

2. **Triton 高层 cast 的最小版本，现在已经能离线编译到 `sm100`。**
   也就是说，`x.to(tl.float8e4nv)` 在最小复现里不再是“必然 ICE”。

3. **Triton inline PTX pack 这次是成功的。**
   下面这条最小写法可以稳定工作：
   ```python
   packed = tl.inline_asm_elementwise(
       'cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;',
       '=h,f,f',
       [x1, x0],
       dtype=tl.int16,
       is_pure=True,
       pack=1,
   )
   ```

4. **最关键：packed bytes 可以和 PyTorch reference 做到逐字节一致。**
   本次实测：
   - Triton inline pack bytes：`[56, 192, 70, 40, 126, 0, 176, 84]`
   - PyTorch reference bytes：`[56, 192, 70, 40, 126, 0, 176, 84]`

5. **block-scale dynamic quantization 的最小版也成功了。**
   在 Triton 内先算 `amax`、再算 `scale = amax / 448`、再 pack 成 `fp8x2`，本地实测：
   - `block_scale_match = True`
   - `block_scale = 1.1428571939468384`
   - `ref_scale = 1.1428571939468384`

### 这次结论和之前最大的不同

之前的判断更偏向“FP8 PTX 这条路整体不可行”；
这次重新尝试后，结论更精确了：

- **`cvt.rn.satfinite.e4m3x2.f32` 本身不是死路；**
- **真正更可疑的是高层 `fp8` tensor 的 store/layout 路径，而不是 `cvt` 指令本身。**

换句话说，这次恢复的是：

**FP32 -> packed FP8x2 (`int16`) 这条底层路径。**

但这还不等于主线 GEMM2 现在就能直接切到 FP8，因为：

- 还没把 packed `int16/int32` 中间表示接进主线 GEMM2；
- 也还没解决 SwiGLU 输出在端到端模型里的量化误差放大问题。

### 因此，后续计划应该更新为

优先级最高的下一步，不该再继续赌高层 `tl.store(... fp8 tensor ...)` 会不会碰巧正常，而是：

1. **GEMM1 末端显式 pack 成 `int16/int32`**；
2. **GEMM2 侧显式 unpack / load**；
3. 先做一个**独立的小 GEMM2 原型**，不要直接改 2000 行主 kernel；
4. 在这个原型上重新评估真实的端到端数值误差，再决定是否接回主线。

---


## 0.1 继续推进：packed-FP8 GEMM2 prototype

基于上面的最小复现，这次进一步验证了：

- GEMM1 末端把激活显式 pack 成 `int16(fp8x2)`；
- 下一段 kernel 再显式 unpack 成 `uint8 -> fp8e4m3`；
- 然后直接喂给 `tl.dot(fp8, fp8)`；

这条路径在一个**独立的小型 GEMM2 prototype** 里是可行的。

对应脚本：

- `D:/Research/mlsys_note/scripts/fp8_packed_gemm2_prototype.py`

### prototype 实测结果

这次还把 prototype 补成了可切换 `--a-block-k {32,64,128}`，并修掉了最初只 pack 首个 K-block 的问题。  
本地实测（RTX 4080）得到：

- `A_BLOCK_K=128`
  - `packed_byte_match = True`
  - `scale_match = True`
  - `direct_abs_err ≈ 9.33e-02`
  - `packed_abs_err ≈ 9.46e-02`
  - `packed_vs_direct ≈ 5.80e-02`
  - `quant_vs_fp32 ≈ 9.97e+00`
- `A_BLOCK_K=64`
  - `packed_byte_match = True`
  - `scale_match = True`
  - `direct_abs_err ≈ 4.97e-02`
  - `packed_abs_err ≈ 4.50e-02`
  - `packed_vs_direct ≈ 4.16e-02`
  - `quant_vs_fp32 ≈ 1.29e+01`
- `A_BLOCK_K=32`
  - `packed_byte_match = True`
  - `scale_match = True`
  - `direct_abs_err ≈ 2.23e-02`
  - `packed_abs_err ≈ 2.09e-02`
  - `packed_vs_direct ≈ 1.77e-02`
  - `quant_vs_fp32 ≈ 9.86e+00`

这几个数字说明：

1. **pack 路径本身没有引入额外字节级错误**；
2. packed 路径喂给下一段 `tl.dot(fp8, fp8)` 后，误差水平和“直接拿高层 fp8 tensor 做 dot”仍然在同一量级；
3. **更细粒度 scale 并不单调更好**：`64` 反而更差，`32` 只比 `128` 略好一点；
4. 真正更大的误差来源仍然是 **FP8 量化策略本身**，不是这次显式 pack / unpack 数据流。

### 离线 `sm100` 编译结果

同一个 prototype 还做了离线 Blackwell 目标编译：

- pack kernel：`.target sm_100a`，并且 `has_cvt_e4m3x2 = True`
- packed gemm kernel：`.target sm_100a`，并且 `has_mma = True`

这意味着：

- **显式 pack 的量化 kernel 可以编到 Blackwell**；
- **packed -> unpack -> fp8 dot 的 GEMM kernel 也可以编到 Blackwell**；
- 从“最小原型”的角度看，`explicit packed storage -> explicit unpack/load -> tl.dot(fp8, fp8)` 这条链路已经被打通了。

### 因此，下一步最值得做什么

现在最值得继续推进的，不再是重复验证 PTX pack 本身，而是：

1. 做一个 **更接近主线 GEMM2 的 K-loop prototype**；
2. 验证 packed A-side 在更大 `K/N` 上的精度与性能；
3. 如果结果继续稳定，再考虑把这条数据流嵌入 `solution/triton/kernel.py` 的独立实验分支。

---

## 0.2 接进真实 `kernel.py` 实验分支后的结果

这一步已经继续做了：在 `D:/Research/mlsys_note/solution/triton/kernel.py` 里加了一个**默认关闭**的实验分支，通过环境变量开启：

```powershell
$env:FUSED_MOE_EXPERIMENTAL_PACKED_GEMM2='1'
$env:FUSED_MOE_EXPERIMENTAL_PACKED_GEMM2_BLOCK_K='32'  # 可选 32 / 64 / 128
```

实验分支做的是：

1. GEMM1 正常输出 `Intermediate[num_padded, 2048]`（FP32）；
2. 新增 `_pack_intermediate_fp8x2_blockscaled_kernel`，把 `Intermediate` 按可配置 `BLOCK_K ∈ {32, 64, 128}` 做 block-scale quantize，并 pack 成 `int16(fp8x2)`；
3. GEMM2 改走 `_packed_fp8_fused_moe_gemm2_kernel`，显式 unpack 成 `uint8 -> fp8e4m3` 后再执行 `tl.dot(fp8, fp8)`；
4. 默认主路径完全不变，只有环境变量打开时才走这条实验路径。

### 本地真实 workload 结果（`test_kernel.py`, T=7）

开启实验分支后，最小 workload 的真实结果是：

- `BLOCK_K=128`
  - `matched_ratio = 0.789501`
  - `PASS = False`
  - `max_abs = 12288`
  - `mean_abs = 534.55`
- `BLOCK_K=64`
  - `matched_ratio = 0.795420`
  - `PASS = False`
  - `max_abs = 10240`
  - `mean_abs = 487.33`
- `BLOCK_K=32`
  - `matched_ratio = 0.802117`
  - `PASS = False`
  - `max_abs = 9728`
  - `mean_abs = 437.41`

目标要求仍然是：`matched_ratio >= 0.95`

而默认主路径同样在这个 workload 上仍然是：

- `matched_ratio = 0.994778`
- `PASS = True`

这说明两件事：

1. **这次显式 pack / unpack 的数据流已经能接进真实 kernel 并完整跑通；**
2. **即使把 A-side scale 粒度从 `128` 继续缩到 `64/32`，端到端数值误差仍然远大于 benchmark tolerance。**

### 目前最可信的判断

到这里基本可以把问题切成两半：

- **编译/布局链路：已经基本打通。**
  - 最小 PTX pack 可行
  - packed GEMM2 prototype 可行
  - 接进真实 kernel 也可行
- **数值精度链路：依然是主矛盾。**
  - 也就是说，现在阻止这条路线进入主线的，不再是“会不会编不过 / 会不会 layout 错”，而是 **SwiGLU 输出量化后误差太大**；
  - 且把 A-side `BLOCK_K` 从 `128 -> 64 -> 32` 并没有带来决定性改善，`32` 也只把 `matched_ratio` 提到 `0.802117`。

### 因此，后面最值得尝试的方向

接下来如果还要继续试，不应再把时间主要花在 PTX packing 上，而应该优先尝试这些方向：

1. **Outlier split / residual correction**
   - 把极端激活值保留在稀疏 FP32 路径；
   - 大多数值走 packed FP8 路径；
   - 最后做一次 residual 补偿。

2. **只对部分 token / block 开启 FP8**
   - 对动态范围温和的块走 packed FP8；
   - 对高风险块退回 FP32 A-side；
   - 做成混合路径，而不是全量一刀切。

3. **换 scale 策略，而不只是继续缩 block**
   - 例如改成 percentile / clipped amax / RMS-based scale；
   - 或先做 lightweight precondition，再量化；
   - 因为单纯 `128 -> 64 -> 32` 已经验证：收益有限，且并不稳定。

## 0.3 真实 `Intermediate` 上的新发现：residual correction 开始显著有效

在上面的基础上，这次继续往前做了两步：

1. 写了一个真实 workload 分析脚本：`D:/Research/mlsys_note/scripts/fp8_packed_gemm2_realdata_analysis.py`
2. 在 prototype 里补了一个 Triton 稀疏 residual GEMM kernel：`D:/Research/mlsys_note/scripts/fp8_packed_gemm2_prototype.py`

### 先看真实 `Intermediate` 的分布

对最小 workload（`b8f4f012`, `T=7`）直接抓 GEMM1 的真实 `Intermediate[num_valid_rows, 2048]` 后，看到它的动态范围极端重尾：

- `abs_p95 ≈ 9.23e3`
- `abs_p99 ≈ 1.75e4`
- `abs_p999 ≈ 3.09e4`
- `abs_max ≈ 6.54e4`

这基本解释了为什么“单一 block scale + 全量 FP8”会在 GEMM2 上崩得很厉害：少数极端值把 scale 顶得过高，导致大量中等值量化分辨率不够。

### 纯 scale 策略：依然不行

这次额外试了几类**不引入 residual** 的纯量化策略：

- quantile / clipped scale（如 `q=0.995 / 0.99 / 0.95`）
- RMS-based scale（不同 multiplier）

在真实 `Intermediate` 上都没有明显改善，`matched_ratio` 仍大约落在：

- `0.77 ~ 0.80`

也就是说，**只改 scale 公式本身，不足以把这条路线救回来。**

### 粗粒度 fallback：也不够

这次还试了两种更便宜的 fallback：

1. **row fallback**：只把少数高风险行退回 FP32
2. **32-wide block fallback**：只把少数高风险 block 退回 FP32

结果也都不理想：

- row fallback（最多退回 top-3 行）只能到 `matched_ratio ≈ 0.829`
- block fallback（最多保留 top-128 blocks，约 `28.6%` blocks）也只有 `matched_ratio ≈ 0.825`

说明问题不是“极少数完整行/完整块坏掉”，而是**每个 32-lane block 内部就存在少量主导误差的 outliers**。

### 真正开始有效的方向：per-block outlier residual

最有价值的新结果是：

- 把 `BLOCK_K=32`
- 对每个 `32` 元素 block，保留 top-k outliers 走 residual
- 其余元素再做 packed FP8 GEMM2

在真实 workload 上已经明显起效：

- `topk=8`
  - outlier fraction = `25.0%`
  - residual L1 fraction ≈ `86.6%`
  - `matched_ratio ≈ 0.93754`
- `topk=9`
  - outlier fraction = `28.125%`
  - residual L1 fraction ≈ `89.9%`
  - `matched_ratio ≈ 0.95047`
- `topk=10`
  - outlier fraction = `31.25%`
  - residual L1 fraction ≈ `92.7%`
  - `matched_ratio ≈ 0.95924`
- `topk=12`
  - outlier fraction = `37.5%`
  - `matched_ratio ≈ 0.97730`
- `topk=16`
  - outlier fraction = `50.0%`
  - `matched_ratio ≈ 0.99300`

这说明两件事：

1. **准确率角度**：这条 packed-FP8 GEMM2 路线并不是“天然不可能过线”；只要把每个 32-block 里最坏的一小部分值单独修正，最小 workload 已经能重新过 `0.95`。
2. **性能角度**：真实问题变成了另一个 tradeoff —— 为了过线，需要保留的 residual 比例并不低。  
   `topk=9/10` 虽然已经过线，但 residual 仍占 `28%~31%` 元素，而且承载了接近 `90%+` 的 L1 量级。

### 工程可行性：稀疏 residual kernel 原型已打通

这次还把随机矩阵 prototype 补成了：

- packed FP8 GEMM2 主路径（Triton）
- per-block top-k residual 路径（Triton 稀疏 GEMM）
- 最后两路相加

在小型原型上，`BLOCK_K=32, topk=10` 的结果已经说明：

- mixed kernel 可以稳定工作
- `mixed_kernel_vs_bquant ≈ 3.00`
- 明显优于纯 packed 量化路径（`aquant_vs_bquant ≈ 221`）

所以现在可以说：

- **纯 packed 路径**：工程已经验证，但精度不够
- **packed + sparse residual 路径**：工程上开始变得可实现，而且精度上已经第一次看到“真实 workload 可过线”的希望

### 更新：真实 `kernel.py` 里的 residual gather 已在 Modal B200 上跑通

后面继续把 real-kernel residual path 单独抽出来做最小复现后，已经确认：

1. 之前的 `~32x` 放大**不是公式错了**；
2. 真正 root cause 是 `_packed_fp8_fused_moe_gemm2_residual_kernel` 做了 autotune，但最初漏了
   `restore_value=['C_ptr']`；
3. 由于这个 kernel 是往 `expert_out` 上**累加** residual，autotune 首轮试配置时会把同一块输出重复累加多次，于是看起来像“residual gather 爆炸”。

修完以后，`solution/triton/kernel.py` 的实验入口已经重新可用，且在 Modal B200 上完成了三层确认：

- `scripts/fp8_residual_gather_modal.py`
  - all-one sanity：`real_over_ref ≈ 1.0`
  - random-weight compare：`max_abs(real-proto) ≈ 1.7e-05`
  - real runtime-buffer compare：`matched_ratio = 1.0`
- 说明：**真实 kernel 里的 residual gather / layout 实现本身现在已经是对的。**

### 2026-03-17 最新 Modal 结论：全局阈值不需要到 `topk=16`

在 gather bug 解掉之后，重新在 Modal B200 上做了 quick sweep：

- `BLOCK_K=32, residual_topk=12`
  - 之前曾出现 `17/19` 通过；
  - 这轮按失败 workload 重新精查时，只剩 `1a4c6ba1` 一例没过，`a7c2bcfd` 已经通过。
- 对 `1a4c6ba1-3cd2-4d7d-b716-84f2d52b69fc`（`T=901`）单独扫：
  - `topk=12`：`matched_ratio = 0.944369`，未过线
  - `topk=13`：`matched_ratio = 0.957404`，已过线
  - `topk=14`：`matched_ratio = 0.968221`
- **全 19 workload quick sweep**：
  - `BLOCK_K=32, residual_topk=13` 已经做到 `19/19 PASSED`

所以当前最准确的判断应该更新为：

- **gather correctness blocker：已解决**
- **最小全局静态阈值：目前看是 `topk=13`，不是 `16`**
- **真正剩下的主问题：性能，而不是 residual gather correctness**

### 因此，下一步最值得做什么

现在最值得推进的，不再是继续 debug gather，而是继续做**性能策略**：

先看两个已经很明确的对照：

#### 对大 workload，`topk=13` 的 experimental path 远慢于 mainline

- `1a4c6ba1`
  - experimental `topk=13`：`21.275 ms`
  - mainline：`0.923 ms`
- `5e8dc11c`
  - experimental `topk=13`：`173.939 ms`
  - mainline：`5.385 ms`
- `58a34f27`
  - experimental `topk=13`：`114.862 ms`
  - mainline：`3.851 ms`

#### 对已经看起来“比较快”的小/中 workload，mainline 也仍然更快

- `b8f4f012`
  - experimental `topk=13`：`1.330 ms`
  - mainline：`0.262 ms`
- `e05c6c03`
  - experimental `topk=13`：`0.351 ms`
  - mainline：`0.313 ms`
- `a7c2bcfd`
  - experimental `topk=13`：`2.215 ms`
  - mainline：`0.363 ms`
- `2e69caee`
  - experimental `topk=13`：`1.060 ms`
  - mainline：`0.228 ms`

这意味着结论还要再往前收紧一步：

- **hybrid fallback 当然有价值，但它本身还不够**
- 在目前这版实现里，packed+residual path 至少在已抽查的快/慢 workload 上都**没有跑赢 mainline**
- 所以如果下一步目标是 README 跑分提升，优先级应该是先把 experimental path 的额外开销大幅压下去，而不是急着把它接回默认主线

1. 重点评估 `topk=13` 在大 `T` workload 上为什么仍然明显偏慢（当前最差仍在 `5e8dc11c`、`58a34f27` 这类 case）
2. 优先做 micro-profile，拆开看：
   - pack kernel 本身花了多少
   - residual GEMM / gather 花了多少
   - 是否存在 `BLOCK_N` / autotune 配置明显不合适
3. 在确认 experimental path 至少能在一部分 workload 上跑赢 mainline 之后，再尝试 hybrid policy：
   - 对能明显受益的 workload / `T` 区间启用 packed+residual
   - 对大 `T` 慢 case 回退到主线 GEMM2 路径
4. 如果静态 `topk=13` 仍太贵，再考虑按 workload / `T` / residual-energy 做动态 top-k，而不是把全局阈值继续抬到 `16`

换句话说，**“显式 pack + 显式 unpack + residual gather” 这一层工程问题现在可以视为已打通；下一阶段应该转向“怎样把它跑得值得”。**

### 2026-03-17 Modal micro-profile：瓶颈已经非常明确

这轮又专门补了一个真实 workload 的 micro-profile：

- `D:/Research/mlsys_note/scripts/fp8_residual_microprofile_modal.py`
- 直接在 Modal B200 上对比：
  - mainline
  - `BLOCK_K=32, residual_topk=13` 的 experimental path

结论非常集中：**真正拖垮性能的不是 packed GEMM2，也不是 pack kernel，而是 residual kernel 本身。**

#### `b8f4f012`（`T=7`）

- mainline：`0.184 ms`
- experimental：`1.189 ms`
- 其中 experimental CUDA 分解：
  - `residual = 1.026 ms`
  - `packed_gemm2 = 0.042 ms`
  - `pack+residual = 0.035 ms`
  - `gemm1 = 0.052 ms`

也就是说，小 workload 上时间几乎全被 residual correction 吃掉了。

#### `1a4c6ba1`（`T=901`）

- mainline：`0.825 ms`
- experimental：`21.135 ms`
- experimental CUDA 分解：
  - `residual = 19.884 ms`
  - `pack+residual = 0.622 ms`
  - `packed_gemm2 = 0.319 ms`
  - `gemm1 = 0.286 ms`

这里 residual kernel 已经占 experimental 总 CUDA 时间的绝大部分，远大于 packed GEMM2 主路径本身。

#### `5e8dc11c`（`T=14107`）

- mainline：`5.410 ms`
- experimental：`172.492 ms`
- experimental CUDA 分解：
  - `residual = 159.911 ms`
  - `pack+residual = 6.123 ms`
  - `packed_gemm2 = 2.716 ms`
  - `gemm1 = 1.737 ms`

大 workload 上更明显：**packed GEMM2 主路径其实已经和 mainline GEMM2 同一个量级，真正爆炸的是 residual gather/GEMM。**

### 这轮 profile 的真正含义

它把问题边界收得更清楚了：

1. **packed GEMM2 本体不是主要问题**
   - `packed_gemm2` 和 mainline `gemm2` 已经在同一量级；
   - 所以“显式 pack + packed GEMM2”这条线本身没有完全失去希望。
2. **现在几乎全部性能债都在 residual path**
   - 当前 residual kernel 近似是在做一个非常昂贵的稀疏 gather-GEMM；
   - `topk=13` 虽然能过线，但它带来的 kernel cost 远远超过主路径节省。
3. **所以下一步优化优先级应该继续收敛**
   - 不是先动 packed GEMM2 tile；
   - 而是优先重构 residual correction 的实现形式，或者显著减少 residual 参与量。

### 因此，下一步最值得做什么（再次收敛版）

1. **先优化 residual path，而不是 mainline/packed GEMM2**
   - 当前 profile 已经说明 residual 是绝对主瓶颈。
2. 具体优先级建议：
   - 先尝试只对“最危险的一小部分 rows / blocks”做 residual，而不是全量 row 全量 block 全做 top-k
   - 或者把 residual correction 改写成更批量化、更少 gather 的形式
   - 再不然就必须把 `topk` 继续压低，并配合 selective fallback / dynamic policy
3. 在 residual path 没明显瘦身之前，不值得继续花时间微调 packed GEMM2 autotune。

### 2026-03-17 继续实验：两条“稀疏化 residual”路线的最新结论

在上面的 micro-profile 之后，又继续试了两条更激进的降成本思路：

1. **Selective residual blocks**
   - 对每一行先算每个 `32`-wide block 的 residual score（`abs(val).sum()`）
   - 只保留 top blocks 进入 residual kernel
2. **Dense fallback on selected rows**
   - 先跑 packed GEMM2 主路径
   - 再只对“高风险 rows”做 dense GEMM2 fallback，并直接覆盖这些 rows 的 `expert_out`

#### 先说 selective residual blocks：效果很差

在最难 workload `1a4c6ba1`（`T=901`）上，`BLOCK_K=32, residual_topk=13` 时：

- `residual_blocks=8`：`matched_ratio = 0.371270`
- `residual_blocks=16`：`matched_ratio = 0.372650`
- `residual_blocks=24`：`matched_ratio = 0.374186`
- `residual_blocks=32`：`matched_ratio = 0.375972`

这基本说明：**误差并不是只集中在少数 blocks 上。**
至少对这个 hardest case 来说，按 block 稀疏化 residual 几乎没有保住精度。

#### Dense fallback on selected rows：数值上比 sparse residual 更有希望，但仍不够

这次还补了一个新的 kernel：

- 对选中的 rows，直接跑 dense GEMM2 fallback，而不是做稀疏 residual gather
- 这样做的直接原因是：主线 `gemm2` 本身其实很快，慢的是 residual kernel

在 `1a4c6ba1` 上：

- `dense_fallback_row_pct=25`：`matched_ratio = 0.564294`
- `dense_fallback_row_pct=50`：`matched_ratio = 0.876499`
- `dense_fallback_row_pct=75`：`matched_ratio = 0.985424`，过线
- `dense_fallback_row_pct=100`：`matched_ratio = 0.985408`，过线

但在大 workload `5e8dc11c`（`T=14107`）上：

- `dense_fallback_row_pct=75`：`matched_ratio = 0.885369`，**仍然不过线**

也就是说，**row-level fallback 至少比 sparse residual 更像一个“可能的方向”，但需要 fallback 的 rows 比例仍然太高。**

### Dense fallback 的 micro-profile：比 residual kernel 好很多，但还不够快

对应的 Modal micro-profile（`dense_fallback_row_pct=75`）：

- `1a4c6ba1`
  - mainline：`0.824 ms`
  - experimental：`6.101 ms`
  - 其中：
    - `dense_fallback = 4.612 ms`
    - `packed_gemm2 = 0.325 ms`
    - `pack+residual = 0.622 ms`
- `5e8dc11c`
  - mainline：`5.407 ms`
  - experimental：`49.869 ms`
  - 其中：
    - `dense_fallback = 36.606 ms`
    - `packed_gemm2 = 2.659 ms`
    - `pack+residual = 7.685 ms`

和之前 `159.9 ms` 的 residual kernel 相比，dense fallback 已经明显更合理；
但它距离 mainline 仍然很远，尤其在大 `T` 上仍然不具竞争力。

### 这轮实验之后，判断可以再收紧一步

1. **按 block 稀疏 residual：基本可以判定不值得继续了**
   - hardest case 上几乎没有形成有效的 accuracy/perf tradeoff
2. **按 row 做 dense fallback：是更合理的替代物**
   - 它至少证明“dense correction”远比“sparse gather residual”高效
3. **但如果不把 selected rows 重新按 expert / block 组织成更高效的 GEMM，仍然不够快**
   - 当前 one-row fallback kernel 还是太贵
4. 因此，如果还要继续这条线，下一步最值得做的是：
   - **把 selected rows regroup by expert，再复用更接近主线的 GEMM2 block kernel**
   - 而不是继续优化 sparse residual gather

### 2026-03-17 再推进一步：grouped dense fallback 已经打通

上面那版 dense fallback 还是“一行一行”地做 fallback GEMM2；
这次继续把它改成了：

- 先选出 high-risk rows
- 再按 `expert` regroup
- 然后走一个 **block-batched dense fallback GEMM2 kernel**
- 最后把这些 rows 的结果 scatter 回原来的 `expert_out`

这个版本已经在 Modal B200 上通过了 sanity / correctness：

- `1a4c6ba1`
  - `dense_fallback_row_pct=100`：`matched_ratio = 0.985366`
  - `dense_fallback_row_pct=75`：`matched_ratio = 0.985358`
- `5e8dc11c`
  - `dense_fallback_row_pct=75`：`matched_ratio = 0.886144`，仍不过线

所以结论是：

- **grouped dense fallback 的数学实现已经是对的**
- 但 row selection policy 仍然不够强，尤其大 `T` case 还是需要更高的 fallback 覆盖率

### grouped dense fallback 的收益：性能改善非常明显

新的 Modal micro-profile（同样 `dense_fallback_row_pct=75`）结果：

- `1a4c6ba1`
  - 旧 one-row fallback：`6.101 ms`
  - 新 grouped fallback：`2.717 ms`
  - 其中新的 CUDA 分解：
    - `pack+residual = 0.623 ms`
    - `dense_fallback = 0.354 ms`
    - `packed_gemm2 = 0.333 ms`
- `5e8dc11c`
  - 旧 one-row fallback：`49.869 ms`
  - 新 grouped fallback：`17.194 ms`
  - 其中新的 CUDA 分解：
    - `pack+residual = 7.696 ms`
    - `packed_gemm2 = 2.669 ms`
    - `dense_fallback = 2.461 ms`

这说明一件很关键的事：

- **把 selected rows regroup by expert 是对的，收益是真实存在的**
- 现在新的主瓶颈已经不再是 dense fallback kernel 本身
- **新主瓶颈变成了 `pack+residual` 阶段 + row selection/scatter 这部分辅助开销**

### 因而下一步最值得做什么（最新收敛版）

如果还继续这条线，优先级应该进一步收敛成：

1. **不要再花时间在 sparse residual kernel 上**
2. 保留 grouped dense fallback 方向
3. 下一步重点改：
   - 为 dense-fallback 模式单独做一个更轻的 **score-only pack kernel**
   - 不再生成完整 `residual idx / val`
   - 只输出 packed FP8 + row risk score
4. 如果还能再进一步，最好做到：
   - 先根据 score 选 rows
   - **只对剩余 rows 跑 packed GEMM2**
   - 避免对最终会 dense fallback 的 rows 再白跑一遍 packed GEMM2

### 2026-03-17 继续推进：score-only 没有本质收益，simple-pack + scale-score 明显更好

在 Modal B200 上继续往下试了两条路：

1. **精确 score-only pack**
   - 保留 `topk=13` 的 block 内 outlier 剔除逻辑；
   - 只是不再写 `residual idx/val`，而是只输出 row score。
   - 结论：**收益非常有限**，瓶颈仍然是 block 内 `topk` 本身，而不是 `idx/val` 写回。

   实测：
   - `1a4c6ba1`：`PASSED | 2.830 ms`
   - `5e8dc11c`：`matched_ratio = 0.887332`，仍未过线
   - microprofile：
     - `1a4c6ba1`：`_pack_intermediate_fp8x2_blockscaled_score_kernel = 648.6 us`
     - `5e8dc11c`：`_pack_intermediate_fp8x2_blockscaled_score_kernel = 7454.7 us`

2. **simple-pack + scale-based row score**
   - dense-fallback 模式下，不再做 block 内 `topk` 剔除；
   - 直接走普通 `_pack_intermediate_fp8x2_blockscaled_kernel`；
   - 用 `packed_intermediate_scales.sum(dim=1)` 作为 row risk score；
   - 然后继续沿用 grouped dense fallback by expert。

   这条路的结果明显更好：

   - `dense_fallback_row_pct = 75`
     - `1a4c6ba1`：`PASSED | 2.441 ms`
     - `5e8dc11c`：`matched_ratio = 0.921356`，未过线，但比旧版 `0.887332` 明显提升
   - `dense_fallback_row_pct = 80`
     - `5e8dc11c`：`matched_ratio = 0.945908`，仍差一点
   - `dense_fallback_row_pct = 85`
     - `5e8dc11c`：`matched_ratio = 0.972675`，**通过**
     - quick benchmark：
       - `1a4c6ba1`：`PASSED | 2.605 ms | 8.07x`
       - `5e8dc11c`：`PASSED | 13.821 ms | 3.26x`
   - `dense_fallback_row_pct = 90`
     - `5e8dc11c`：`matched_ratio = 0.988616`，通过
   - `dense_fallback_row_pct = 100`
     - `5e8dc11c`：`matched_ratio = 0.988609`，通过

### 这一轮 microprofile 说明了什么

以 `dense_fallback_row_pct = 85` 为例：

- `1a4c6ba1`
  - mainline：`0.826 ms`
  - experimental：`2.559 ms`
  - 分解：
    - `pack = 0.308 ms`
    - `packed_gemm2 = 0.334 ms`
    - `dense_fallback = 0.387 ms`
- `5e8dc11c`
  - mainline：`5.425 ms`
  - experimental：`13.886 ms`
  - 分解：
    - `pack = 3.860 ms`
    - `packed_gemm2 = 2.732 ms`
    - `dense_fallback = 2.773 ms`

因此现在最准确的判断应该更新为：

- **score-only pack 不是对症下药**；真正贵的是 block 内 `topk` 本身
- **simple-pack + scale-score 是当前更值得继续的方向**
- `5e8dc11c` 已经证明：**这条近似选行策略在 `85%` fallback 时可以过线**
- 但性能仍然离 README 主线很远，因为：
  - 还在对所有 rows 做 pack
  - 还在对所有 rows 做 packed GEMM2
  - 而其中 `85%` rows 最终又会被 dense fallback 覆盖

### 因而下一步真正该做什么（再次收敛）

下一刀最值钱的，不再是继续调 score，而是：

1. **先做 cheap row score / row selection**
2. 然后把 rows 拆成两路：
   - **fallback rows**：直接走 grouped dense fallback
   - **non-fallback rows**：单独 regroup + pack + packed GEMM2
3. 目标是彻底避免：
   - 对 fallback rows 白跑 `pack`
   - 对 fallback rows 白跑 `packed GEMM2`

如果这个“两路分流”做出来，才第一次有机会把 large-T case 的 experimental 时间真正压近 mainline。

### 2026-03-17 晚些时候继续推进：two-path split 已落地，large-T 再次明显下降

在上面的结论基础上，继续把 dense-fallback 路径往前推了两步：

1. **two-path split**
   - 先选出 fallback rows；
   - fallback rows 继续走 grouped dense fallback；
   - **non-fallback rows 单独 regroup by expert，再单独 pack + packed GEMM2**；
   - 从而避免对 fallback rows 白跑 packed GEMM2。

2. **coarse score kernel**
   - 不再用 `BLOCK_K=32` 给 dense-fallback 做 row score；
   - 改成单独的 `_score_intermediate_blockscaled_kernel`；
   - 并继续把 score 粒度放粗：
     - 先试 `BLOCK_K=128`
     - 再试 `BLOCK_K=256`

### 阶段性结果 A：two-path split + score `BLOCK_K=128`

- `5e8dc11c`
  - `81%` fallback：`matched_ratio = 0.950619`，通过
  - quick benchmark：`9.728 ms | 4.64x`
- microprofile（`81%`）：
  - mainline：`5.407 ms`
  - experimental：`10.254 ms`
  - 分解：
    - `score = 0.968 ms`
    - `pack = 0.124 ms`
    - `packed_gemm2 = 0.765 ms`
    - `dense_fallback = 2.674 ms`

这说明 two-path split 已经把：

- `pack` 从 **`3.860 ms` 压到 `0.124 ms`**
- `packed_gemm2` 从 **`2.732 ms` 压到 `0.765 ms`**

新的主瓶颈只剩：

- `dense_fallback`
- `score kernel`
- 以及若干 GPU `index_select/index_copy` 辅助开销

### 阶段性结果 B：把 score 粒度进一步放粗到 `BLOCK_K=256`

这一刀继续有效：

- `5e8dc11c`
  - `81%` fallback：`matched_ratio = 0.949658`，**刚好不过**
  - `83%` fallback：`matched_ratio = 0.960270`，通过
  - quick benchmark：`9.211 ms | 4.91x`
- `1a4c6ba1`
  - `83%` fallback：`2.884 ms | 7.36x`
- `58a34f27`
  - `83%` fallback：`7.190 ms | 5.01x`

microprofile（`5e8dc11c`, `83%`）：

- mainline：`5.411 ms`
- experimental：`9.407 ms`
- 分解：
  - `score = 0.480 ms`
  - `pack = 0.111 ms`
  - `packed_gemm2 = 0.693 ms`
  - `dense_fallback = 2.734 ms`

对比 `BLOCK_K=128` 的 score 版本：

- `score`：`0.968 ms -> 0.480 ms`
- experimental wall：`10.254 ms -> 9.407 ms`
- quick benchmark：`9.728 ms -> 9.211 ms`

### 这轮之后，最准确的判断再更新一次

- **two-path split 是对的，而且收益很大**
- **coarse score 也是对的，而且 `BLOCK_K=256` 比 `128` 更好**
- 当前最优实验配置变成：
  - `packed_gemm2 block_k = 32`
  - `residual_topk = 13`
  - `dense_fallback_row_pct = 83`
  - `dense-fallback score block_k = 256`

但也必须承认一件事：

- **它仍然没有超过 README 主线**
  - `5e8dc11c`：当前 experimental `4.91x`，README 主线约 `8.34x`
  - `58a34f27`：当前 experimental `5.01x`，README 主线约 `9.25x`

也就是说，这条线已经从“完全不可用”推进到了“有明显进展”，但**仍然只是 research path，不是 README-ready path**。

### 如果还继续这条线，最值钱的下一步只剩两个

1. **继续减少 dense fallback 占比**
   - 关键是提升 row score 的排序质量，而不是继续堆 dense fallback 比例；
   - 可以尝试：
     - 更聪明的 coarse score（例如 block-mean + block-max 混合）
     - 多级 score（先极粗筛，再对边界 rows 精筛）

2. **把 regroup / gather / scatter 辅助开销继续压到 Triton 里**
   - 当前 `index_select/index_copy` 和其他 GPU-side Tensor ops 仍然有明显成本；
   - 如果把 row regroup 也下沉成 kernel，wall time 还能继续降一截。

### 2026-03-18 再推进：dense fallback 的真正下一刀是 grouped `BLOCK_M=128`

这一轮继续沿着 “row selection + grouped dense fallback” 往下打，先后做了三类实验：

1. `score_mode` sweep：`sum / sum_max / max / top2 / top4`
2. coarse-to-fine boundary refine：只对落在边界的一小段 rows 再用 `BLOCK_K=32` 精筛
3. dense fallback regroup 的 `BLOCK_M` 强制实验

#### A. `score_mode` sweep：`sum` 仍然最好

在 `5e8dc11c`、`80%` fallback 上，结果如下：

- `sum`：`matched_ratio = 0.946565`（`ap-46jGSB5buKo0azUqqLsMm2`）
- `sum_max`：`0.945476`（`ap-BGj9bbOfuV4vHdJYG9WadS`）
- `max`：`0.943205`（`ap-Hl8Mh3OM7ujGMOyLHGOJsB`）
- `top2`：`0.944438`（`ap-nbYafAkhU2XBrLQqOCUAyz`）
- `top4`：`0.945145`（`ap-e5Tt3eHIsdCu8kca4BdQ8S`）

结论很直接：

- **更花哨的 block aggregation 没有超过原始 `sum(scale) * abs(weight)`**
- row score 不是当前最一阶的瓶颈

#### B. boundary refine：方向没错，但还不足以把 `80%` 拉过线

在同一个 `5e8dc11c`、`80%` fallback case 上，加入边界精筛后：

- `1%` window：`0.946098`（`ap-T4gbbiO2lZkX5dOVExnc4X`）
- `2%` window：`0.947654`（`ap-aaJlRC2E0j6pIsMbvwMlNV`）← 这一轮最好
- `3%` window：`0.946947`（`ap-Vfjs0KpNrGHQOS8UytKroi`）
- `4%` window：`0.946393`（`ap-XNC9LbIBZ1LTG7mfc6pZKm`）
- `5%` window：`0.947504`（`ap-5Bri7WNrMuvp9ySHJa3dLC`）
- `6%` window：`0.947530`（`ap-SQIExerKXSIJww6nG7GXBS`）
- `8%` window：`0.946600`（`ap-H7xS0mAcRyWkpH8CCTSTgx`）
- `10%` window：`0.947231`（`ap-xAO9bF4YlANIRZmaBtj1oG`）

可以看出：

- 多级 score / boundary refine **确实能带来一点提升**
- 但这点提升仍然不够把 `80%` 从 `~0.946` 推到 `0.95+`
- 说明 coarse score 的排序质量已经接近这条路的局部上限

#### C. 真正的 win：dense fallback regroup 强制 `BLOCK_M=128`

继续看 microprofile 后，最重要的发现变成了：

- dense fallback kernel 仍然是最大核
- 现有按 padding 最优来选 `grouped_block_m`，并不等于 wall time 最优

于是直接对 dense fallback regroup 强制 `BLOCK_M`，在 `5e8dc11c`、`81%` fallback 上得到：

- `BLOCK_M=32`：`matched_ratio = 0.951319`（`ap-pLJ6vgPTX5GFyDQ7DAyoAB`）
- `BLOCK_M=64`：`0.951648`（`ap-dQLD7VQfbrIXG1IRYlLsrw`）
- `BLOCK_M=128`：`0.951035`（`ap-ED0mqn4mLJxa8oE8Lpbxub`）

也就是说：

- **三档都能过线**
- correctness 不是问题，性能才是问题

对应 microprofile（`5e8dc11c`）：

- `BLOCK_M=32`（`ap-TYq728d9j54IJRPkd6ERpl`）
  - experimental wall：`9.507 ms`
  - `dense_fallback = 2.558 ms`
- `BLOCK_M=64`（`ap-QMYzd9Jaztm0rEoqRBnXJJ`）
  - experimental wall：`9.056 ms`
  - `dense_fallback = 2.707 ms`
- `BLOCK_M=128`（`ap-iPrOwFegqHiHUYnY31uXKh`）
  - experimental wall：`8.696 ms`
  - `dense_fallback = 2.154 ms`
  - `packed_gemm2 = 0.783 ms`
  - `score = 0.486 ms`
  - `copy = 0.170 ms`

这是这一轮最关键的结论：

- **对于 regroup 之后的大 dense fallback batch，`BLOCK_M=128` 的 TensorCore 吞吐收益明显大于 padding 成本**
- 这和主线 large-T GEMM 的经验是一致的：一旦进入大批量区间，tile 变大往往更值钱

#### benchmark 结果：这次是真正把 research best 往前推了一步

先看 3 个重点 workload 的过滤 benchmark（`ap-NuC1CjHWTy3QVzyXbkoUib`）：

- `1a4c6ba1`：`2.851 ms | 7.31x`
- `5e8dc11c`：`8.281 ms | 5.42x`
- `58a34f27`：`6.314 ms | 5.66x`

再看完整 19 workload benchmark（`ap-wr457CR7ipW88wbgRor3Ll`）：

- **`19 / 19 PASSED`**
- `1a4c6ba1`：`3.434 ms | 6.28x`
- `5e8dc11c`：`8.578 ms | 5.31x`
- `58a34f27`：`6.783 ms | 5.40x`

同时也顺手确认了：

- `80%` fallback + `BLOCK_M=128` 仍然不过线：`matched_ratio = 0.946021`（`ap-inKu4OdFAtkNzFt4LHZDUZ`）

所以当前最优实验配置正式更新为：

- `packed_gemm2 block_k = 32`
- `residual_topk = 13`
- `dense_fallback_row_pct = 81`
- `dense-fallback score block_k = 256`
- `dense-fallback score mode = sum`
- `dense-fallback refine window pct = 0`
- `dense-fallback grouped BLOCK_M = 128`

#### 这轮之后，判断也要再更新一次

- **score formula 调优不是主矛盾**
- **boundary refine 有帮助，但收益太小**
- **dense fallback regroup 的 tile 选择才是当前 first-order lever**
- 这条线已经把 `5e8dc11c` 从之前的 `~9.0 ms / 5.00x` 再推进到 `8.578 ms / 5.31x`

但同样必须诚实地说：

- 它**仍然没有超过 README 主线**
  - `5e8dc11c`：当前 experimental `5.31x`，README 主线约 `8.34x`
  - `58a34f27`：当前 experimental `5.40x`，README 主线约 `9.25x`

所以这条线现在的定位仍然是：

- **research path，而且是在持续变好**
- 但还不是 README-ready path

如果还要继续往下打，下一步最值钱的方向已经进一步收敛成：

1. **把 dense fallback 的 regroup / gather / scatter 辅助逻辑继续下沉到 Triton**
   - 当前 `other` 里仍有稳定的 `gpu_index_kernel` 成本
   - 这部分是下一个最像“纯 overhead”的刀口
2. **把 `dense_fallback BLOCK_M` 做成动态性能策略，而不是固定按 padding 最优**
   - 现在 `BLOCK_M=128` 对大 fallback batch 更快
   - 但未来更理想的是按 `selected_row_count / grouped_total_blocks` 动态切 `32 / 64 / 128`

---
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
