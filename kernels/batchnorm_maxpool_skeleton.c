#include <stdio.h>

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
    (void)input;
    (void)output;
    (void)N;
    (void)H;
    (void)W;
    (void)C;
    (void)running_mean;
    (void)running_var;
    (void)gamma;
    (void)beta;
    (void)epsilon;
    fprintf(stderr, "fast_batchnorm is a skeleton; implement the kernel first\n");
    return -1;
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
