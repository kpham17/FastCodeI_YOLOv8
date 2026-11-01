#ifndef FASTCODE_BATCHNORM_H
#define FASTCODE_BATCHNORM_H

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Input and output use CHW16 storage: [N][ceil(C / 16)][H][W][16].
 * C is the logical channel count and the final block is padded in storage.
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

int fast_batchnorm_prepare(
    float *scale,
    float *bias,
    int C,
    const float *running_mean,
    const float *running_var,
    const float *gamma,
    const float *beta,
    float epsilon);

/* Applies already prepared scale/bias arrays without allocation or setup. */
int fast_batchnorm_apply(
    const float *input,
    float *output,
    int N,
    int H,
    int W,
    int C,
    const float *scale,
    const float *bias);

int fast_batchnorm_apply_unrolled(
    const float *input,
    float *output,
    int N,
    int H,
    int W,
    int C,
    const float *scale,
    const float *bias);

int fast_batchnorm_apply_unrolled_nocopy(
    const float *input,
    float *output,
    int N,
    int H,
    int W,
    int C,
    const float *scale,
    const float *bias);

#ifdef __cplusplus
}
#endif

#endif
