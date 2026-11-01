#ifndef FASTCODE_MAXPOOL_H
#define FASTCODE_MAXPOOL_H

#include <immintrin.h>
#include <math.h>

/*
 * CHW16 storage is [N][ceil(C / 16)][H][W][16].  The channel block is the
 * innermost dimension, so one ZMM load contains one spatial value.
 */
static inline float scalar_maxpool_pixel_chw16(
    const float *input,
    int H, int W, int C,
    int n, int out_h, int out_w, int c,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w)
{
    const int blocks = (C + 15) / 16;
    const int block = c / 16;
    const int lane = c % 16;
    const size_t plane_stride = (size_t)H * W * 16;
    const size_t batch_stride = (size_t)blocks * plane_stride;
    float max_val = -INFINITY;
    int h_start = out_h * stride_h - padding_h;
    int w_start = out_w * stride_w - padding_w;

    for (int kh = 0; kh < kernel_h; ++kh) {
        int in_h = h_start + kh;
        if (in_h < 0 || in_h >= H) continue;
        for (int kw = 0; kw < kernel_w; ++kw) {
            int in_w = w_start + kw;
            if (in_w < 0 || in_w >= W) continue;
            size_t index = (size_t)n * batch_stride +
                           (size_t)block * plane_stride +
                           ((size_t)in_h * W + in_w) * 16 + lane;
            if (input[index] > max_val) max_val = input[index];
        }
    }
    return max_val;
}

/*
 * Reduce a 9x9 input footprint into a 5x5 output tile for one 16-channel
 * block. The 25 accumulators are the largest square output tile that fits
 * alongside an input vector in the 32 ZMM registers.
 */
static inline void maxpool_tile_5x5_chw16_avx512(
    const float *restrict in_ptr,
    float *restrict out_ptr,
    int in_stride_h,
    int out_stride_h)
{
    __m512 a[5][5];
    for (int oh = 0; oh < 5; ++oh) {
        for (int ow = 0; ow < 5; ++ow) {
            a[oh][ow] = _mm512_set1_ps(-INFINITY);
        }
    }

    /*
     * The loop bounds are fixed for this tile. GCC's unroll directives expose
     * the independent accumulator chains while retaining the compact source.
     */
#pragma GCC unroll 9
    for (int kh = 0; kh < 9; ++kh) {
        const float *row = in_ptr + kh * in_stride_h;
#pragma GCC unroll 9
        for (int kw = 0; kw < 9; ++kw) {
            __m512 value = _mm512_loadu_ps(row + kw * 16);
            const int oh_begin = kh > 4 ? kh - 4 : 0;
            const int oh_end = kh < 5 ? kh : 4;
            const int ow_begin = kw > 4 ? kw - 4 : 0;
            const int ow_end = kw < 5 ? kw : 4;
            for (int oh = oh_begin; oh <= oh_end; ++oh) {
                for (int ow = ow_begin; ow <= ow_end; ++ow) {
                    a[oh][ow] = _mm512_max_ps(a[oh][ow], value);
                }
            }
        }
    }

    for (int oh = 0; oh < 5; ++oh) {
        for (int ow = 0; ow < 5; ++ow) {
            _mm512_storeu_ps(
                out_ptr + oh * out_stride_h + ow * 16, a[oh][ow]);
        }
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
