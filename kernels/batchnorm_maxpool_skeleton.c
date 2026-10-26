#include <immintrin.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/batchnorm.h"
#include "../include/maxpool.h"

int fast_batchnorm(
    const float *input,
    float *output,
    int N,
    int H,
    int W,
    int C,
    const float *running_mean,
    const float *running_var,
    const float *gamma,
    const float *beta,
    float epsilon)
{
    if (!input || !output || !running_mean || !running_var || !gamma || !beta) {
        return -1;
    }
    if (N <= 0 || H <= 0 || W <= 0 || C <= 0) {
        return -1;
    }

    float stack_scale[2048] __attribute__((aligned(64)));
    float stack_bias[2048] __attribute__((aligned(64)));
    float *scale = stack_scale;
    float *bias = stack_bias;
    int allocated = 0;

    if (C > 2048) {
        size_t bytes = (size_t)C * sizeof(float);
        size_t aligned_bytes = (bytes + 63) & ~(size_t)63;
        scale = (float *)aligned_alloc(64, aligned_bytes);
        bias = (float *)aligned_alloc(64, aligned_bytes);
        if (!scale || !bias) {
            free(scale);
            free(bias);
            return -1;
        }
        allocated = 1;
    }

    for (int c = 0; c < C; ++c) {
        float inv_std = 1.0f / sqrtf(running_var[c] + epsilon);
        scale[c] = gamma[c] * inv_std;
        bias[c] = beta[c] - running_mean[c] * scale[c];
    }

    size_t total_pixels = (size_t)N * H * W;
    for (size_t i = 0; i < total_pixels; ++i) {
        const float *input_pixel = input + i * C;
        float *output_pixel = output + i * C;
        int c = 0;

        for (; c <= C - 64; c += 64) {
            __m512 s0 = _mm512_load_ps(&scale[c + 0]);
            __m512 b0 = _mm512_load_ps(&bias[c + 0]);
            __m512 s1 = _mm512_load_ps(&scale[c + 16]);
            __m512 b1 = _mm512_load_ps(&bias[c + 16]);
            __m512 s2 = _mm512_load_ps(&scale[c + 32]);
            __m512 b2 = _mm512_load_ps(&bias[c + 32]);
            __m512 s3 = _mm512_load_ps(&scale[c + 48]);
            __m512 b3 = _mm512_load_ps(&bias[c + 48]);

            __m512 x0 = _mm512_loadu_ps(input_pixel + c + 0);
            __m512 x1 = _mm512_loadu_ps(input_pixel + c + 16);
            __m512 x2 = _mm512_loadu_ps(input_pixel + c + 32);
            __m512 x3 = _mm512_loadu_ps(input_pixel + c + 48);

            __m512 y0 = _mm512_fmadd_ps(x0, s0, b0);
            __m512 y1 = _mm512_fmadd_ps(x1, s1, b1);
            __m512 y2 = _mm512_fmadd_ps(x2, s2, b2);
            __m512 y3 = _mm512_fmadd_ps(x3, s3, b3);

            _mm512_storeu_ps(output_pixel + c + 0, y0);
            _mm512_storeu_ps(output_pixel + c + 16, y1);
            _mm512_storeu_ps(output_pixel + c + 32, y2);
            _mm512_storeu_ps(output_pixel + c + 48, y3);
        }

        for (; c <= C - 16; c += 16) {
            __m512 s = _mm512_load_ps(&scale[c]);
            __m512 b = _mm512_load_ps(&bias[c]);
            __m512 x = _mm512_loadu_ps(input_pixel + c);
            __m512 y = _mm512_fmadd_ps(x, s, b);
            _mm512_storeu_ps(output_pixel + c, y);
        }

        if (c < C) {
            int remainder = C - c;
            __mmask16 mask = (__mmask16)((1U << remainder) - 1);
            __m512 s = _mm512_maskz_load_ps(mask, &scale[c]);
            __m512 b = _mm512_maskz_load_ps(mask, &bias[c]);
            __m512 x = _mm512_maskz_loadu_ps(mask, input_pixel + c);
            __m512 y = _mm512_fmadd_ps(x, s, b);
            _mm512_mask_storeu_ps(output_pixel + c, mask, y);
        }
    }

    if (allocated) {
        free(scale);
        free(bias);
    }
    return 0;
}

int fast_maxpool2d(
    const float *input,
    float *output,
    int N,
    int H,
    int W,
    int C,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int padding_h,
    int padding_w)
{
    if (!input || !output || N <= 0 || H <= 0 || W <= 0 || C <= 0 ||
        kernel_h <= 0 || kernel_w <= 0 || stride_h <= 0 || stride_w <= 0 ||
        padding_h < 0 || padding_w < 0) {
        return -1;
    }

    int H_out = (H + 2 * padding_h - kernel_h) / stride_h + 1;
    int W_out = (W + 2 * padding_w - kernel_w) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) return -1;

    int can_use_fast_tile = (kernel_h == 5 && kernel_w == 5 &&
                             stride_h == 1 && stride_w == 1 &&
                             padding_h == 2 && padding_w == 2 && C == 1);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            if (can_use_fast_tile) {
                for (int oh = 0; oh <= H_out - 10; oh += 10) {
                    int ih = oh - padding_h;
                    if (ih < 0 || ih + 14 > H) continue;

                    for (int ow = 0; ow <= W_out - 12; ow += 12) {
                        int iw = ow - padding_w;
                        if (iw < 0 || iw + 16 > W) continue;

                        const float *in_ptr =
                            &input[((n * H + ih) * W + iw) * C + c];
                        float *out_ptr =
                            &output[((n * H_out + oh) * W_out + ow) * C + c];
                        maxpool_tile_14x16_avx512(in_ptr, out_ptr, W, W_out);
                    }
                }
            }

            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    int covered_by_fast_tile = 0;
                    if (can_use_fast_tile) {
                        int tile_oh = (oh / 10) * 10;
                        int tile_ow = (ow / 12) * 12;
                        int ih = tile_oh - padding_h;
                        int iw = tile_ow - padding_w;
                        covered_by_fast_tile =
                            tile_oh <= H_out - 10 &&
                            tile_ow <= W_out - 12 &&
                            ih >= 0 && ih + 14 <= H &&
                            iw >= 0 && iw + 16 <= W;
                    }
                    if (covered_by_fast_tile) {
                        continue;
                    }

                    int out_idx = ((n * H_out + oh) * W_out + ow) * C + c;
                    output[out_idx] = scalar_maxpool_pixel(
                        input, N, H, W, C, n, oh, ow, c,
                        kernel_h, kernel_w, stride_h, stride_w,
                        padding_h, padding_w);
                }
            }
        }
    }
    return 0;
}
