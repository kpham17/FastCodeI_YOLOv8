#ifndef FASTCODE_MAXPOOL_H
#define FASTCODE_MAXPOOL_H

/*
 * Max pooling in NHWC layout. Padding values should be negative infinity,
 * matching torch.nn.functional.max_pool2d for floating-point inputs.
 * The caller owns input and output buffers. Returns 0 on success.
 */
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
    int padding_w);

#endif
