/*
 * CUDA MoE Kernel Templates — Track A (FlashInfer AI Kernel Contest)
 *
 * Target: NVIDIA B200 (Blackwell, sm_100a)
 * Model: DeepSeek-V3 MoE layer
 *   - hidden_dim H = 7168, intermediate I = 2048, E_local = 32
 *   - FP8 (e4m3fn) weights with per-128-block scales
 *   - SwiGLU activation: silu(W3_out) * W1_out
 *
 * These kernels are REFERENCE implementations for correctness.
 * The current binding.py uses PyTorch matmul (cuBLAS) instead.
 * Replace binding.py's per-expert loop with these fused kernels to
 * eliminate weight materialization and reduce kernel launch overhead.
 *
 * ═══════════════════════════════════════════════════════════════
 * Optimization Roadmap (for teammates):
 *
 * Step 1: Wire up these kernels in binding.py via ctypes/pybind11
 *         - Replace per-expert dequant+matmul with fused kernel launches
 *         - Eliminates [4096, 7168] and [7168, 2048] fp32 weight materialization
 *
 * Step 2: In-tile online dequant
 *         - Load FP8 tiles, convert to float in registers, apply block scale
 *         - Never write full fp32 weights to global memory
 *
 * Step 3: CUTLASS 3.x sm100a templates
 *         - Use TMA (Tensor Memory Accelerator) for async global→shared loads
 *         - Use wgmma (Warp Group Matrix Multiply) for tensor core MMA
 *         - Use TMEM (Tensor Memory) for accumulator storage
 *
 * Step 4: Persistent kernel
 *         - Single kernel launch processes all experts
 *         - Eliminates per-expert launch overhead (~25 launches → 1)
 *         - Work-stealing scheduler across expert tiles
 *
 * Step 5: Larger tile sizes (BM=128, BN=256) + multi-stage pipeline
 *         - Deeper software pipeline (4-6 stages) to hide global memory latency
 *         - Double/triple buffering in shared memory
 * ═══════════════════════════════════════════════════════════════
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

/* ── FP8 Block-Scale Constants ── */
#define QBLOCK 128     /* Quantization block size */
#define H_DIM  7168    /* Hidden dimension */
#define I_DIM  2048    /* Intermediate dimension (after SwiGLU) */
#define W13_N  4096    /* W1 concat W3 output dimension */

/* ── Tile sizes for reference GEMM kernels ── */
#define BM 64          /* Tokens per M-tile */
#define BN 64          /* Output channels per N-tile */
#define BK 32          /* K-reduction per iteration */
#define TM 4           /* Thread tile M (each thread computes TM x TN) */
#define TN 4           /* Thread tile N */


/* ═══════════════════════════════════════════════════════════════
 * Kernel 1: FP8 Block-Scale Dequantization
 *
 * Converts FP8 tensor to FP32 with per-128-block scales.
 * Grid:  ((N * K + 255) / 256, 1, 1)
 * Block: (256, 1, 1)
 *
 * This is a simple element-wise kernel. For production, fuse
 * the dequant into the GEMM tile loop (see Kernel 2/3).
 * ═══════════════════════════════════════════════════════════════ */
__global__ void dequant_fp8_blockscale_kernel(
    const __nv_fp8_e4m3*  __restrict__ input,   /* [rows, cols] fp8 */
    const float*          __restrict__ scale,   /* [rows/128, cols/128] fp32 */
    float*                __restrict__ output,  /* [rows, cols] fp32 */
    int rows, int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (idx >= total) return;

    int row = idx / cols;
    int col = idx % cols;

    /* Scale indices: each 128-element block has one scale */
    int scale_row = row / QBLOCK;
    int scale_col = col / QBLOCK;
    int scale_cols = cols / QBLOCK;

    float s = scale[scale_row * scale_cols + scale_col];
    float val = (float)input[idx];
    output[idx] = val * s;
}


/* ═══════════════════════════════════════════════════════════════
 * Kernel 2: Fused GEMM1 + SwiGLU
 *
 * Computes: SwiGLU( hidden_states[token_ids] @ W13.T )
 *
 * Input A:  [T, H]        FP8  (hidden_states, block-scaled)
 * Input B:  [W13_N, H]    FP8  (W13 weights for one expert, block-scaled)
 * Output C: [num_tokens, I_DIM] FP32 (SwiGLU output)
 *
 * Key design:
 * - Loads A and B as FP8, dequants in registers using block scales
 * - Accumulates W1 and W3 results simultaneously
 * - Applies SwiGLU in the epilogue: silu(W3_out) * W1_out
 *
 * Block: (BN/TN, BM/TM) = (16, 16) = 256 threads
 * Shared memory: A[BM][BK+1] + B_W1[BK][BN+1] + B_W3[BK][BN+1]
 *              = 64*33 + 32*65 + 32*65 = 2112 + 2080 + 2080 = 6272 floats ≈ 25 KB
 * ═══════════════════════════════════════════════════════════════ */
__global__ void gemm1_swiglu_kernel(
    const __nv_fp8_e4m3*  __restrict__ A,          /* [T, H] fp8 */
    const float*          __restrict__ A_scale,     /* [H/128, T] fp32 */
    const __nv_fp8_e4m3*  __restrict__ B,          /* [W13_N, H] fp8 (one expert) */
    const float*          __restrict__ B_scale,     /* [W13_N/128, H/128] fp32 */
    float*                __restrict__ C,          /* [num_tokens, I_DIM] fp32 output */
    const int64_t*        __restrict__ token_ids,  /* [num_tokens] token indices into A */
    int num_tokens,
    int T                                          /* original T, for scale stride */
) {
    /* Thread and tile indices */
    int bx = blockIdx.x;   /* N-tile index (over I_DIM = 2048) */
    int by = blockIdx.y;   /* M-tile index (over num_tokens) */
    int tx = threadIdx.x;  /* 0..15 (N direction) */
    int ty = threadIdx.y;  /* 0..15 (M direction) */

    /* Global offsets */
    int row_start = by * BM;            /* first token in this M-tile */
    int col_start = bx * BN;            /* first output channel (in I_DIM space) */
    int col_w1 = col_start;             /* W1 columns: [0, I_DIM) */
    int col_w3 = col_start + I_DIM;     /* W3 columns: [I_DIM, W13_N) */

    /* Shared memory tiles */
    __shared__ float smem_A[BM][BK + 1];       /* +1 to avoid bank conflicts */
    __shared__ float smem_W1[BK][BN + 1];
    __shared__ float smem_W3[BK][BN + 1];

    /* Register accumulators: each thread computes TM x TN */
    float acc_w1[TM][TN] = {{0.0f}};
    float acc_w3[TM][TN] = {{0.0f}};

    /* Linear thread ID for cooperative loading */
    int tid = ty * (BN / TN) + tx;
    int num_threads = (BM / TM) * (BN / TN);  /* 256 */

    /* K-loop over H dimension */
    for (int k0 = 0; k0 < H_DIM; k0 += BK) {
        /* ── Load A tile [BM, BK] into shared memory ── */
        /* Each thread loads BM*BK/256 = 64*32/256 = 8 elements */
        for (int i = tid; i < BM * BK; i += num_threads) {
            int m = i / BK;
            int k = i % BK;
            int global_m = row_start + m;
            int global_k = k0 + k;

            if (global_m < num_tokens && global_k < H_DIM) {
                int token = (int)token_ids[global_m];
                /* FP8 dequant: val * scale */
                float raw = (float)A[token * H_DIM + global_k];
                /* A_scale layout: [H/128, T], so scale[k_block][token] */
                float s = A_scale[(global_k / QBLOCK) * T + token];
                smem_A[m][k] = raw * s;
            } else {
                smem_A[m][k] = 0.0f;
            }
        }

        /* ── Load B tiles [BK, BN] for W1 and W3 ── */
        for (int i = tid; i < BK * BN; i += num_threads) {
            int k = i / BN;
            int n = i % BN;
            int global_k = k0 + k;
            int global_n_w1 = col_w1 + n;
            int global_n_w3 = col_w3 + n;

            if (global_k < H_DIM) {
                /* B layout: [W13_N, H], row = output channel, col = K */
                /* B_scale layout: [W13_N/128, H/128] */
                int scale_k = global_k / QBLOCK;

                if (global_n_w1 < W13_N) {
                    float raw = (float)B[global_n_w1 * H_DIM + global_k];
                    float s = B_scale[(global_n_w1 / QBLOCK) * (H_DIM / QBLOCK) + scale_k];
                    smem_W1[k][n] = raw * s;
                } else {
                    smem_W1[k][n] = 0.0f;
                }

                if (global_n_w3 < W13_N) {
                    float raw = (float)B[global_n_w3 * H_DIM + global_k];
                    float s = B_scale[(global_n_w3 / QBLOCK) * (H_DIM / QBLOCK) + scale_k];
                    smem_W3[k][n] = raw * s;
                } else {
                    smem_W3[k][n] = 0.0f;
                }
            } else {
                smem_W1[k][n] = 0.0f;
                smem_W3[k][n] = 0.0f;
            }
        }

        __syncthreads();

        /* ── Compute TM x TN sub-tile ── */
        for (int kk = 0; kk < BK; kk++) {
            float a_reg[TM];
            float b_w1_reg[TN];
            float b_w3_reg[TN];

            /* Load A fragment */
            for (int i = 0; i < TM; i++)
                a_reg[i] = smem_A[ty * TM + i][kk];

            /* Load B fragments */
            for (int j = 0; j < TN; j++) {
                b_w1_reg[j] = smem_W1[kk][tx * TN + j];
                b_w3_reg[j] = smem_W3[kk][tx * TN + j];
            }

            /* Outer product accumulation */
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    acc_w1[i][j] += a_reg[i] * b_w1_reg[j];
                    acc_w3[i][j] += a_reg[i] * b_w3_reg[j];
                }
            }
        }

        __syncthreads();
    }

    /* ── Epilogue: SwiGLU and store ── */
    for (int i = 0; i < TM; i++) {
        int global_m = row_start + ty * TM + i;
        if (global_m >= num_tokens) continue;

        for (int j = 0; j < TN; j++) {
            int global_n = col_start + tx * TN + j;
            if (global_n >= I_DIM) continue;

            /* SwiGLU: silu(w3) * w1, where silu(x) = x * sigmoid(x) */
            float w1 = acc_w1[i][j];
            float w3 = acc_w3[i][j];
            float sig = 1.0f / (1.0f + expf(-w3));
            float silu_w3 = w3 * sig;
            float result = silu_w3 * w1;

            C[global_m * I_DIM + global_n] = result;
        }
    }
}


/* ═══════════════════════════════════════════════════════════════
 * Kernel 3: Fused GEMM2 + Weighted Scatter-Add
 *
 * Computes: output[token_id] += (intermediate @ W2.T) * routing_weight
 *
 * Input A:  [num_tokens, I_DIM]  FP32 (SwiGLU output from Kernel 2)
 * Input B:  [H, I_DIM]          FP8  (W2 weights for one expert, block-scaled)
 * Output C: [T, H]              FP32 (accumulated output, atomicAdd)
 *
 * Key design:
 * - A is already FP32 (no dequant needed)
 * - B (W2) loaded as FP8, dequant in registers
 * - Epilogue: multiply by routing weight, atomicAdd to output
 *
 * Block: (BN/TN, BM/TM) = (16, 16) = 256 threads
 * Shared memory: A[BM][BK+1] + B[BK][BN+1]
 *              = 64*33 + 32*65 = 2112 + 2080 = 4192 floats ≈ 16.4 KB
 * ═══════════════════════════════════════════════════════════════ */
__global__ void gemm2_scatter_kernel(
    const float*          __restrict__ A,          /* [num_tokens, I_DIM] fp32 */
    const __nv_fp8_e4m3*  __restrict__ B,          /* [H, I_DIM] fp8 (one expert) */
    const float*          __restrict__ B_scale,     /* [H/128, I_DIM/128] fp32 */
    float*                __restrict__ C,          /* [T, H] fp32 output (accumulate) */
    const int64_t*        __restrict__ token_ids,  /* [num_tokens] original token indices */
    const float*          __restrict__ token_wts,  /* [num_tokens] routing weights */
    int num_tokens
) {
    int bx = blockIdx.x;   /* N-tile index (over H = 7168) */
    int by = blockIdx.y;   /* M-tile index (over num_tokens) */
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_start = by * BM;
    int col_start = bx * BN;

    __shared__ float smem_A[BM][BK + 1];
    __shared__ float smem_B[BK][BN + 1];

    float acc[TM][TN] = {{0.0f}};

    int tid = ty * (BN / TN) + tx;
    int num_threads = (BM / TM) * (BN / TN);

    /* K-loop over I_DIM */
    for (int k0 = 0; k0 < I_DIM; k0 += BK) {

        /* Load A tile [BM, BK] — already fp32, no dequant */
        for (int i = tid; i < BM * BK; i += num_threads) {
            int m = i / BK;
            int k = i % BK;
            int global_m = row_start + m;
            int global_k = k0 + k;
            if (global_m < num_tokens && global_k < I_DIM)
                smem_A[m][k] = A[global_m * I_DIM + global_k];
            else
                smem_A[m][k] = 0.0f;
        }

        /* Load B tile [BK, BN] — FP8 dequant */
        for (int i = tid; i < BK * BN; i += num_threads) {
            int k = i / BN;
            int n = i % BN;
            int global_k = k0 + k;
            int global_n = col_start + n;

            if (global_k < I_DIM && global_n < H_DIM) {
                /* B layout: [H, I_DIM], row = output channel, col = K */
                float raw = (float)B[global_n * I_DIM + global_k];
                int scale_n = global_n / QBLOCK;
                int scale_k = global_k / QBLOCK;
                float s = B_scale[scale_n * (I_DIM / QBLOCK) + scale_k];
                smem_B[k][n] = raw * s;
            } else {
                smem_B[k][n] = 0.0f;
            }
        }

        __syncthreads();

        /* Compute TM x TN sub-tile */
        for (int kk = 0; kk < BK; kk++) {
            float a_reg[TM];
            float b_reg[TN];

            for (int i = 0; i < TM; i++)
                a_reg[i] = smem_A[ty * TM + i][kk];
            for (int j = 0; j < TN; j++)
                b_reg[j] = smem_B[kk][tx * TN + j];

            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }

        __syncthreads();
    }

    /* ── Epilogue: weighted atomicAdd to output ── */
    for (int i = 0; i < TM; i++) {
        int global_m = row_start + ty * TM + i;
        if (global_m >= num_tokens) continue;

        int token = (int)token_ids[global_m];
        float wt = token_wts[global_m];

        for (int j = 0; j < TN; j++) {
            int global_n = col_start + tx * TN + j;
            if (global_n >= H_DIM) continue;

            float val = acc[i][j] * wt;
            atomicAdd(&C[token * H_DIM + global_n], val);
        }
    }
}


/* ═══════════════════════════════════════════════════════════════
 * Host Launch Wrappers
 *
 * These can be called from Python via ctypes or pybind11.
 * Example binding.py integration:
 *
 *   import ctypes
 *   lib = ctypes.CDLL("./kernel.so")
 *   lib.launch_gemm1_swiglu(...)
 * ═══════════════════════════════════════════════════════════════ */

extern "C" {

void launch_dequant(
    const void* input, const float* scale, float* output,
    int rows, int cols, cudaStream_t stream
) {
    int total = rows * cols;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dequant_fp8_blockscale_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_fp8_e4m3*)input, scale, output, rows, cols
    );
}

void launch_gemm1_swiglu(
    const void* A, const float* A_scale,
    const void* B, const float* B_scale,
    float* C, const int64_t* token_ids,
    int num_tokens, int T, cudaStream_t stream
) {
    /* Grid: (I_DIM / BN, ceil(num_tokens / BM)) */
    dim3 grid(I_DIM / BN, (num_tokens + BM - 1) / BM);
    dim3 block(BN / TN, BM / TM);  /* (16, 16) = 256 threads */

    gemm1_swiglu_kernel<<<grid, block, 0, stream>>>(
        (const __nv_fp8_e4m3*)A, A_scale,
        (const __nv_fp8_e4m3*)B, B_scale,
        C, token_ids, num_tokens, T
    );
}

void launch_gemm2_scatter(
    const float* A,
    const void* B, const float* B_scale,
    float* C, const int64_t* token_ids, const float* token_wts,
    int num_tokens, cudaStream_t stream
) {
    /* Grid: (H_DIM / BN, ceil(num_tokens / BM)) */
    dim3 grid(H_DIM / BN, (num_tokens + BM - 1) / BM);
    dim3 block(BN / TN, BM / TM);  /* (16, 16) = 256 threads */

    gemm2_scatter_kernel<<<grid, block, 0, stream>>>(
        A, (const __nv_fp8_e4m3*)B, B_scale,
        C, token_ids, token_wts, num_tokens
    );
}

}  /* extern "C" */
