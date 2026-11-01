#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include "../include/batchnorm.h"
#include "../include/maxpool.h"

int fast_batchnorm_prepare(
    float *scale,
    float *bias,
    int C,
    const float *running_mean,
    const float *running_var,
    const float *gamma,
    const float *beta,
    float epsilon)
{
    if (!scale || !bias || !running_mean || !running_var || !gamma || !beta ||
        C <= 0 || epsilon < 0.0f) {
        return -1;
    }
    const int blocks = (C + 15) / 16;
    for (int c = 0; c < C; ++c) {
        scale[c] = gamma[c] / sqrtf(running_var[c] + epsilon);
        bias[c] = beta[c] - running_mean[c] * scale[c];
    }
    for (int c = C; c < blocks * 16; ++c) {
        scale[c] = 0.0f;
        bias[c] = 0.0f;
    }
    return 0;
}

int fast_batchnorm_apply(
    const float *input,
    float *output,
    int N,
    int H,
    int W,
    int C,
    const float *scale,
    const float *bias)
{
    if (!input || !output || !scale || !bias ||
        N <= 0 || H <= 0 || W <= 0 || C <= 0) {
        return -1;
    }

    const int blocks = (C + 15) / 16;
    const size_t spatial = (size_t)H * W;
    const size_t block_stride = spatial * 16;
    const size_t batch_stride = (size_t)blocks * block_stride;

    for (int n = 0; n < N; ++n) {
        for (int block = 0; block < blocks; ++block) {
            const float *input_block = input + (size_t)n * batch_stride +
                                       (size_t)block * block_stride;
            float *output_block = output + (size_t)n * batch_stride +
                                  (size_t)block * block_stride;
            const int valid = C - block * 16 < 16 ? C - block * 16 : 16;
            const __mmask16 mask = (__mmask16)((1U << valid) - 1);
            __m512 s = _mm512_load_ps(scale + block * 16);
            __m512 b = _mm512_load_ps(bias + block * 16);

            for (size_t p = 0; p < spatial; ++p) {
                __m512 x = _mm512_loadu_ps(input_block + p * 16);
                __m512 y = _mm512_fmadd_ps(x, s, b);
                if (valid == 16) {
                    _mm512_storeu_ps(output_block + p * 16, y);
                } else {
                    _mm512_mask_storeu_ps(output_block + p * 16, mask, y);
                }
            }
        }
    }

    return 0;
}

int fast_batchnorm_apply_unrolled(
    const float *input,
    float *output,
    int N,
    int H,
    int W,
    int C,
    const float *scale,
    const float *bias)
{
    if (!input || !output || !scale || !bias ||
        N <= 0 || H <= 0 || W <= 0 || C <= 0) {
        return -1;
    }

    const int blocks = (C + 15) / 16;
    const size_t spatial = (size_t)H * W;
    const size_t block_stride = spatial * 16;
    const size_t batch_stride = (size_t)blocks * block_stride;

    for (int n = 0; n < N; ++n) {
        for (int block = 0; block < blocks; ++block) {
            const float *input_block = input + (size_t)n * batch_stride +
                                       (size_t)block * block_stride;
            float *output_block = output + (size_t)n * batch_stride +
                                  (size_t)block * block_stride;
            const int valid = C - block * 16 < 16 ? C - block * 16 : 16;
            const __mmask16 mask = (__mmask16)((1U << valid) - 1);
            __m512 s = _mm512_load_ps(scale + block * 16);
            __m512 b = _mm512_load_ps(bias + block * 16);
            size_t p = 0;

            if (valid == 16) {
                for (; p + 8 <= spatial; p += 8) {
                    __m512 y0 = _mm512_loadu_ps(input_block + (p + 0) * 16);
                    __m512 y1 = _mm512_loadu_ps(input_block + (p + 1) * 16);
                    __m512 y2 = _mm512_loadu_ps(input_block + (p + 2) * 16);
                    __m512 y3 = _mm512_loadu_ps(input_block + (p + 3) * 16);
                    __m512 y4 = _mm512_loadu_ps(input_block + (p + 4) * 16);
                    __m512 y5 = _mm512_loadu_ps(input_block + (p + 5) * 16);
                    __m512 y6 = _mm512_loadu_ps(input_block + (p + 6) * 16);
                    __m512 y7 = _mm512_loadu_ps(input_block + (p + 7) * 16);
                    y0 = _mm512_fmadd_ps(y0, s, b);
                    y1 = _mm512_fmadd_ps(y1, s, b);
                    y2 = _mm512_fmadd_ps(y2, s, b);
                    y3 = _mm512_fmadd_ps(y3, s, b);
                    y4 = _mm512_fmadd_ps(y4, s, b);
                    y5 = _mm512_fmadd_ps(y5, s, b);
                    y6 = _mm512_fmadd_ps(y6, s, b);
                    y7 = _mm512_fmadd_ps(y7, s, b);
                    _mm512_storeu_ps(output_block + (p + 0) * 16, y0);
                    _mm512_storeu_ps(output_block + (p + 1) * 16, y1);
                    _mm512_storeu_ps(output_block + (p + 2) * 16, y2);
                    _mm512_storeu_ps(output_block + (p + 3) * 16, y3);
                    _mm512_storeu_ps(output_block + (p + 4) * 16, y4);
                    _mm512_storeu_ps(output_block + (p + 5) * 16, y5);
                    _mm512_storeu_ps(output_block + (p + 6) * 16, y6);
                    _mm512_storeu_ps(output_block + (p + 7) * 16, y7);
                }
            }
            for (; p < spatial; ++p) {
                __m512 y = _mm512_fmadd_ps(
                    _mm512_loadu_ps(input_block + p * 16), s, b);
                if (valid == 16) {
                    _mm512_storeu_ps(output_block + p * 16, y);
                } else {
                    _mm512_mask_storeu_ps(output_block + p * 16, mask, y);
                }
            }
        }
    }
    return 0;
}

static inline void batchnorm_8_nocopy(
    const float *input,
    float *output,
    const __m512 scale,
    const __m512 bias)
{
    __asm__ volatile(
        ".intel_syntax noprefix\n\t"
        "vmovaps zmm0, %[scale]\n\t"
        "vmovaps zmm1, %[bias]\n\t"
        "vmovups zmm2, [%[input] + 0]\n\t"
        "vmovups zmm3, [%[input] + 64]\n\t"
        "vmovups zmm4, [%[input] + 128]\n\t"
        "vmovups zmm5, [%[input] + 192]\n\t"
        "vmovups zmm6, [%[input] + 256]\n\t"
        "vmovups zmm7, [%[input] + 320]\n\t"
        "vmovups zmm8, [%[input] + 384]\n\t"
        "vmovups zmm9, [%[input] + 448]\n\t"
        "vfmadd213ps zmm2, zmm0, zmm1\n\t"
        "vfmadd213ps zmm3, zmm0, zmm1\n\t"
        "vfmadd213ps zmm4, zmm0, zmm1\n\t"
        "vfmadd213ps zmm5, zmm0, zmm1\n\t"
        "vfmadd213ps zmm6, zmm0, zmm1\n\t"
        "vfmadd213ps zmm7, zmm0, zmm1\n\t"
        "vfmadd213ps zmm8, zmm0, zmm1\n\t"
        "vfmadd213ps zmm9, zmm0, zmm1\n\t"
        "vmovups [%[output] + 0], zmm2\n\t"
        "vmovups [%[output] + 64], zmm3\n\t"
        "vmovups [%[output] + 128], zmm4\n\t"
        "vmovups [%[output] + 192], zmm5\n\t"
        "vmovups [%[output] + 256], zmm6\n\t"
        "vmovups [%[output] + 320], zmm7\n\t"
        "vmovups [%[output] + 384], zmm8\n\t"
        "vmovups [%[output] + 448], zmm9\n\t"
        ".att_syntax prefix\n\t"
        :
        : [input] "r"(input), [output] "r"(output),
          [scale] "v"(scale), [bias] "v"(bias)
        : "zmm0", "zmm1", "zmm2", "zmm3", "zmm4", "zmm5",
          "zmm6", "zmm7", "zmm8", "zmm9", "memory");
}

int fast_batchnorm_apply_unrolled_nocopy(
    const float *input,
    float *output,
    int N,
    int H,
    int W,
    int C,
    const float *scale,
    const float *bias)
{
    if (!input || !output || !scale || !bias ||
        N <= 0 || H <= 0 || W <= 0 || C <= 0) {
        return -1;
    }

    const int blocks = (C + 15) / 16;
    const size_t spatial = (size_t)H * W;
    const size_t block_stride = spatial * 16;
    const size_t batch_stride = (size_t)blocks * block_stride;
    for (int n = 0; n < N; ++n) {
        for (int block = 0; block < blocks; ++block) {
            const float *input_block = input + (size_t)n * batch_stride +
                                       (size_t)block * block_stride;
            float *output_block = output + (size_t)n * batch_stride +
                                  (size_t)block * block_stride;
            const int valid = C - block * 16 < 16 ? C - block * 16 : 16;
            __m512 s = _mm512_load_ps(scale + block * 16);
            __m512 b = _mm512_load_ps(bias + block * 16);
            size_t p = 0;
            if (valid == 16) {
                for (; p + 8 <= spatial; p += 8) {
                    batchnorm_8_nocopy(
                        input_block + p * 16, output_block + p * 16, s, b);
                }
            }
            for (; p < spatial; ++p) {
                __m512 y = _mm512_fmadd_ps(
                    _mm512_loadu_ps(input_block + p * 16), s, b);
                const __mmask16 mask = (__mmask16)((1U << valid) - 1);
                if (valid == 16) {
                    _mm512_storeu_ps(output_block + p * 16, y);
                } else {
                    _mm512_mask_storeu_ps(output_block + p * 16, mask, y);
                }
            }
        }
    }
    return 0;
}

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
    if (!input || !output || !running_mean || !running_var || !gamma || !beta ||
        N <= 0 || H <= 0 || W <= 0 || C <= 0 || epsilon < 0.0f) {
        return -1;
    }
    const int padded_c = ((C + 15) / 16) * 16;
    float *scale = (float *)aligned_alloc(64, (size_t)padded_c * sizeof(float));
    float *bias = (float *)aligned_alloc(64, (size_t)padded_c * sizeof(float));
    if (!scale || !bias) {
        free(scale);
        free(bias);
        return -1;
    }
    int status = fast_batchnorm_prepare(
        scale, bias, C, running_mean, running_var, gamma, beta, epsilon);
    if (status == 0) {
        status = fast_batchnorm_apply_unrolled(
            input, output, N, H, W, C, scale, bias);
    }
    free(scale);
    free(bias);
    return status;
}

int fast_maxpool2d(
    const float *input,
    float *output,
    int N, int H, int W, int C,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int padding_h, int padding_w)
{
    if (!input || !output || N <= 0 || H <= 0 || W <= 0 || C <= 0 ||
        kernel_h <= 0 || kernel_w <= 0 || stride_h <= 0 || stride_w <= 0 ||
        padding_h < 0 || padding_w < 0) {
        return -1;
    }

    int H_out = (H + 2 * padding_h - kernel_h) / stride_h + 1;
    int W_out = (W + 2 * padding_w - kernel_w) / stride_w + 1;
    if (H_out <= 0 || W_out <= 0) return -1;

    const int blocks = (C + 15) / 16;
    const size_t input_block_stride = (size_t)H * W * 16;
    const size_t output_block_stride = (size_t)H_out * W_out * 16;
    const size_t input_batch_stride = (size_t)blocks * input_block_stride;
    const size_t output_batch_stride = (size_t)blocks * output_block_stride;
    const int fast_tile = kernel_h == 5 && kernel_w == 5 &&
                          stride_h == 1 && stride_w == 1 &&
                          padding_h == 2 && padding_w == 2;

    for (int n = 0; n < N; ++n) {
        for (int block = 0; block < blocks; ++block) {
            const int valid = C - block * 16 < 16 ? C - block * 16 : 16;
            const float *input_block = input + (size_t)n * input_batch_stride +
                                       (size_t)block * input_block_stride;
            float *output_block = output + (size_t)n * output_batch_stride +
                                  (size_t)block * output_block_stride;
            for (int oh = 0; oh < H_out; ++oh) {
                for (int ow = 0; ow < W_out; ++ow) {
                    int tile_valid = fast_tile && valid == 16 &&
                                     oh % 5 == 2 && ow % 5 == 2 &&
                                     oh + 4 < H_out && ow + 4 < W_out &&
                                     oh >= 2 && ow >= 2 &&
                                     oh + 4 <= H && ow + 4 <= W;
                    if (tile_valid) {
                        maxpool_tile_5x5_chw16_avx512(
                            input_block + ((size_t)(oh - 2) * W + ow - 2) * 16,
                            output_block + ((size_t)oh * W_out + ow) * 16,
                            W * 16, W_out * 16);
                        ow += 4;
                        continue;
                    }
                    for (int c = 0; c < valid; ++c) {
                        output_block[((size_t)oh * W_out + ow) * 16 + c] =
                            scalar_maxpool_pixel_chw16(
                                input, H, W, C, n, oh, ow,
                                block * 16 + c, kernel_h, kernel_w,
                                stride_h, stride_w, padding_h, padding_w);
                    }
                }
            }
        }
    }
    return 0;
}
