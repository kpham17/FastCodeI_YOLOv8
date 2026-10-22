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
    (void)input;
    (void)output;
    (void)N;
    (void)H;
    (void)W;
    (void)C;
    (void)kernel_h;
    (void)kernel_w;
    (void)stride_h;
    (void)stride_w;
    (void)padding_h;
    (void)padding_w;
    fprintf(stderr, "fast_maxpool2d is a skeleton; implement the kernel first\n");
    return -1;
}
