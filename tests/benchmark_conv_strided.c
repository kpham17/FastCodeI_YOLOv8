#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../include/conv2D.h"

static volatile float sink;
static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + t.tv_nsec * 1e-9;
}
static int compare_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static void copy_chunk(float *out, const float *in, int H, int C)
{
    for (int p = 0; p < H * H; ++p)
        memcpy(out + (size_t)p * C, in + (size_t)p * 2 * C + C,
               (size_t)C * sizeof(float));
}
static float *run(int mode, float *parent, float *dense, int H, int C,
                  float *packed)
{
    if (mode == 2)
        return conv_3x3_strided(parent, H, H, C, packed, C, 1, 2 * C, C);
    if (mode == 1) copy_chunk(dense, parent, H, C);
    return conv_3x3(dense, H, H, C, packed, C, 1);
}
int main(int argc, char **argv)
{
    int samples = argc > 1 ? atoi(argv[1]) : 31;
    if (samples < 3 || samples > 1001 || samples % 2 == 0) return 2;
    const int shapes[][2] = {{160,16},{80,32},{40,64},{20,128},
                            {256,16},{128,32},{64,64},{32,128},
                            {320,16},{160,32},{80,64},{40,128}};
    puts("H,W,C,OC,samples,dense_conv_us,copy_plus_conv_us,strided_conv_us,copy_over_strided");
    for (size_t s = 0; s < sizeof(shapes)/sizeof(shapes[0]); ++s) {
        int H = shapes[s][0], C = shapes[s][1];
        size_t n = (size_t)H * H * C;
        float *parent = _mm_malloc(2 * n * sizeof(float), 64);
        float *dense = _mm_malloc(n * sizeof(float), 64);
        float *weights = _mm_malloc((size_t)9 * C * C * sizeof(float), 64);
        double *times = malloc((size_t)3 * samples * sizeof(double));
        if (!parent || !dense || !weights || !times) return 2;
        for (size_t i = 0; i < 2 * n; ++i)
            parent[i] = (float)((int)(i * 17u % 101) - 50) * 0.003f;
        for (size_t i = 0; i < (size_t)9 * C * C; ++i)
            weights[i] = (float)((int)(i * 23u % 103) - 51) * 0.003f;
        float *packed = kernel_pack(weights, 9 * C, C);
        copy_chunk(dense, parent, H, C);
        float *a = run(0, parent, dense, H, C, packed);
        float *b = run(2, parent, dense, H, C, packed);
        if (!a || !b || memcmp(a, b, n * sizeof(float))) {
            fprintf(stderr, "FAIL: benchmark output mismatch\n"); return 1;
        }
        _mm_free(a); _mm_free(b);
        for (int i = 0; i < 3; ++i)
            for (int mode = 0; mode < 3; ++mode) {
                float *out = run(mode, parent, dense, H, C, packed);
                if (!out) return 2;
                sink = out[n / 2]; _mm_free(out);
            }
        for (int i = 0; i < samples; ++i) {
            for (int j = 0; j < 3; ++j) {
                int mode = (i + j) % 3;
                double start = now();
                float *out = run(mode, parent, dense, H, C, packed);
                double elapsed = now() - start;
                if (!out) return 2;
                times[(size_t)mode * samples + i] = elapsed * 1e6;
                sink = out[n / 2]; _mm_free(out);
            }
        }
        double medians[3];
        for (int mode = 0; mode < 3; ++mode) {
            double *t = times + (size_t)mode * samples;
            qsort(t, samples, sizeof(double), compare_double);
            medians[mode] = t[samples / 2];
        }
        printf("%d,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.4f\n", H,H,C,C,samples,
               medians[0],medians[1],medians[2],medians[1]/medians[2]);
        fflush(stdout);
        _mm_free(parent); _mm_free(dense); _mm_free(weights);
        _mm_free(packed); free(times);
    }
    return 0;
}
