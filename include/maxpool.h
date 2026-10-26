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

static inline __m512 maxpool_horizontal_5(__m512 v)
{
    __m512i vi = _mm512_castps_si512(v);
    __m512 h = v;
    __m512 shift = _mm512_castsi512_ps(_mm512_alignr_epi32(vi, vi, 14));
    h = _mm512_max_ps(h, shift);
    shift = _mm512_castsi512_ps(_mm512_alignr_epi32(vi, vi, 15));
    h = _mm512_max_ps(h, shift);
    __m512 right1 = _mm512_castsi512_ps(_mm512_alignr_epi32(vi, vi, 1));
    __m512 right2 = _mm512_castsi512_ps(_mm512_alignr_epi32(vi, vi, 2));
    h = _mm512_max_ps(_mm512_max_ps(h, right1), right2);
    __m512i hi = _mm512_castps_si512(h);
    return _mm512_castsi512_ps(_mm512_alignr_epi32(hi, hi, 2));
}

// Optimized 16x16 Input -> 12x12 Output AVX-512 kernel.
// This tile is valid only for one contiguous spatial plane. It must not be
// applied directly to interleaved NHWC data with C > 1.
static inline void maxpool_tile_16x16_avx512(
    const float *in_ptr,
    float *out_ptr,
    int in_stride_h,
    int out_stride_h)
{
    __m512 in[16];

    // Keep all input rows live while four independent vertical reductions run.
    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        in[r] = _mm512_loadu_ps(in_ptr + r * in_stride_h);
    }

    const __mmask16 store_mask = 0x0FFF;
    for (int r = 0; r < 12; r += 4) {
        __m512 v0 = _mm512_max_ps(
            _mm512_max_ps(_mm512_max_ps(in[r], in[r + 1]), in[r + 2]),
            _mm512_max_ps(in[r + 3], in[r + 4]));
        __m512 v1 = _mm512_max_ps(
            _mm512_max_ps(_mm512_max_ps(in[r + 1], in[r + 2]), in[r + 3]),
            _mm512_max_ps(in[r + 4], in[r + 5]));
        __m512 v2 = _mm512_max_ps(
            _mm512_max_ps(_mm512_max_ps(in[r + 2], in[r + 3]), in[r + 4]),
            _mm512_max_ps(in[r + 5], in[r + 6]));
        __m512 v3 = _mm512_max_ps(
            _mm512_max_ps(_mm512_max_ps(in[r + 3], in[r + 4]), in[r + 5]),
            _mm512_max_ps(in[r + 6], in[r + 7]));

        _mm512_mask_storeu_ps(out_ptr + (r + 0) * out_stride_h,
                              store_mask, maxpool_horizontal_5(v0));
        _mm512_mask_storeu_ps(out_ptr + (r + 1) * out_stride_h,
                              store_mask, maxpool_horizontal_5(v1));
        _mm512_mask_storeu_ps(out_ptr + (r + 2) * out_stride_h,
                              store_mask, maxpool_horizontal_5(v2));
        _mm512_mask_storeu_ps(out_ptr + (r + 3) * out_stride_h,
                              store_mask, maxpool_horizontal_5(v3));
    }

}

int fast_maxpool2d(
    const float *input,
    float *output,
    int N, int H, int W, int C,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w);

/*
 * Max pooling for planar NCHW-style storage:
 * input[n, c, h, w] is contiguous in w, then h, then c, then n.
 *
 * This entry point exists so C=3 data can use the optimized C=1 tile one
 * plane at a time. It does not accept interleaved NHWC storage.
 */
int fast_maxpool2d_planar(
    const float *input,
    float *output,
    int N, int H, int W, int C,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w);

#endif