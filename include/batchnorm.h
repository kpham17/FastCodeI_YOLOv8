#ifndef FASTCODE_BATCHNORM_H
#define FASTCODE_BATCHNORM_H

/*
 * Batch normalization inference in NHWC layout.
 * The caller owns all buffers. Returns 0 on success.
 */
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
    float epsilon);

#endif
