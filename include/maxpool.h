#ifndef FASTCODE_MAXPOOL_H
#define FASTCODE_MAXPOOL_H

#include <immintrin.h>
#include <math.h>

// Helper: Scalar fallback for boundary pixels and remainder tiles
static inline float scalar_maxpool_pixel(
    const float *input,
    int N, int H, int W, int C,
    int n, int out_h, int out_w, int c,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w)
{
    float max_val = -INFINITY;
    int h_start = out_h * stride_h - padding_h;
    int w_start = out_w * stride_w - padding_w;

    for (int kh = 0; kh < kernel_h; ++kh) {
        int in_h = h_start + kh;
        if (in_h < 0 || in_h >= H) continue;

        for (int kw = 0; kw < kernel_w; ++kw) {
            int in_w = w_start + kw;
            if (in_w < 0 || in_w >= W) continue;

            int idx = ((n * H + in_h) * W + in_w) * C + c;
            if (input[idx] > max_val) {
                max_val = input[idx];
            }
        }
    }
    return max_val;
}

// Optimized 14x16 Input -> 10x12 Output AVX-512 Kernel
// Assumes a 5x5 kernel, stride 1, pad 2, and unit channel stride (C=1 or contiguous spatial loads)
static inline void maxpool_tile_14x16_avx512(
    const float *in_ptr,
    float *out_ptr,
    int in_stride_h,
    int out_stride_h)
{
    __m512 in[14];

    // 1. Issue 14 Vector Loads into ZMM0..ZMM13
    #pragma unroll
    for (int r = 0; r < 14; ++r) {
        in[r] = _mm512_loadu_ps(in_ptr + r * in_stride_h);
    }

    // 2. Stage 1: Vertical 5x1 Reduction across 10 Output Rows (Ports 0 & 1)
    __m512 v_max[10];
    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        __m512 m01   = _mm512_max_ps(in[r + 0], in[r + 1]);
        __m512 m23   = _mm512_max_ps(in[r + 2], in[r + 3]);
        __m512 m0123 = _mm512_max_ps(m01, m23);
        v_max[r]     = _mm512_max_ps(m0123, in[r + 4]);
    }

    // 3. Stage 2 & 3: Horizontal 1x5 Shift (Port 5) + Reduction (Ports 0 & 1)
    const __mmask16 store_mask = 0x0FFF; // Bits 0..11 set (12 valid outputs)

    #pragma unroll
    for (int r = 0; r < 10; ++r) {
        __m512 v = v_max[r];

        // Shift relative positions using circular alignr
        __m512i vi = _mm512_castps_si512(v);
        __m512 L2 = _mm512_castsi512_ps(_mm512_alignr_epi32(vi, vi, 14)); // i-2
        __m512 L1 = _mm512_castsi512_ps(_mm512_alignr_epi32(vi, vi, 15)); // i-1
        __m512 R1 = _mm512_castsi512_ps(_mm512_alignr_epi32(vi, vi, 1));  // i+1
        __m512 R2 = _mm512_castsi512_ps(_mm512_alignr_epi32(vi, vi, 2));  // i+2

        // Pairwise horizontal max reduction
        __m512 h_max0  = _mm512_max_ps(L2, L1);
        __m512 h_max1  = _mm512_max_ps(R1, R2);
        __m512 h_max2  = _mm512_max_ps(h_max0, h_max1);
        __m512 h_final = _mm512_max_ps(h_max2, v);

        // Align element 2 to position 0 for masked store
        __m512i h_final_i = _mm512_castps_si512(h_final);
        __m512 h_aligned = _mm512_castsi512_ps(
            _mm512_alignr_epi32(h_final_i, h_final_i, 2));

        // Store 12 valid float32 outputs to L1 memory
        _mm512_mask_storeu_ps(out_ptr + r * out_stride_h, store_mask, h_aligned);
    }
}

int fast_maxpool2d(
    const float *input,
    float *output,
    int N, int H, int W, int C,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w);

#endif