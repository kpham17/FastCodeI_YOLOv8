#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../include/conv2D.h"

/* Catches using selected_C as the physical pixel pitch, wrong offsets,
 * kernel-row/column transposition, padding errors and partial-tile writes. */
static float value(size_t i)
{
    unsigned x = (unsigned)i * 1664525u + 1013904223u;
    x ^= x >> 13;
    return ((int)(x % 401u) - 200) * 0.001f;
}

static int check_case(int H, int W, int C, int OC, int stride,
                      int full_C, int offset, int unaligned, int dense_compare)
{
    size_t count = (size_t)H * W * full_C;
    size_t bytes = (count + unaligned) * sizeof(float);
    float *allocation = _mm_malloc(bytes, 64);
    float *snapshot = malloc(bytes);
    float *weights = _mm_malloc((size_t)9 * C * OC * sizeof(float), 64);
    if (!allocation || !snapshot || !weights) exit(2);
    float *input = allocation + unaligned;
    for (size_t i = 0; i < count + unaligned; ++i) allocation[i] = NAN;
    for (int p = 0; p < H * W; ++p)
        for (int c = 0; c < C; ++c)
            input[(size_t)p * full_C + offset + c] = value((size_t)p * C + c);
    memcpy(snapshot, allocation, bytes);
    for (size_t i = 0; i < (size_t)9 * C * OC; ++i) weights[i] = value(i + 131);
    float *packed = kernel_pack(weights, 9 * C, OC);
    float *actual = conv_3x3_strided(input, H, W, C, packed, OC, stride,
                                     full_C, offset);
    if (!actual) { fprintf(stderr, "unexpected NULL\n"); return 1; }
    int OH = (H - 1) / stride + 1, OW = (W - 1) / stride + 1;
    for (int h = 0; h < OH; ++h) {
        for (int w = 0; w < OW; ++w) {
            for (int oc = 0; oc < OC; ++oc) {
                double expected = 0;
                for (int kh = 0; kh < 3; ++kh) {
                    for (int kw = 0; kw < 3; ++kw) {
                        int ih = h * stride + kh - 1, iw = w * stride + kw - 1;
                        if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                        for (int c = 0; c < C; ++c) {
                            size_t ii = ((size_t)ih * W + iw) * full_C + offset + c;
                            size_t ki = ((size_t)(kw * 3 + kh) * C + c) * OC + oc;
                            expected += (double)input[ii] * weights[ki];
                        }
                    }
                }
                float a = actual[((size_t)h * OW + w) * OC + oc];
                if (!isfinite(a) || fabs(a - expected) > 2e-5 + 2e-5 * fabs(expected)) {
                    fprintf(stderr, "FAIL H=%d W=%d C=%d OC=%d s=%d pitch=%d offset=%d at %d,%d,%d: %g != %.9g\n",
                            H, W, C, OC, stride, full_C, offset, h, w, oc, a, expected);
                    return 1;
                }
            }
        }
    }
    if (memcmp(snapshot, allocation, bytes)) {
        fprintf(stderr, "FAIL: input modified\n"); return 1;
    }
    if (dense_compare) {
        float *dense = _mm_malloc((size_t)H * W * C * sizeof(float), 64);
        for (int p = 0; p < H * W; ++p)
            memcpy(dense + (size_t)p * C, input + (size_t)p * full_C + offset,
                   (size_t)C * sizeof(float));
        float *reference = conv_3x3(dense, H, W, C, packed, OC, stride);
        if (memcmp(actual, reference, (size_t)OH * OW * OC * sizeof(float))) {
            fprintf(stderr, "FAIL: copied-chunk convolution differs\n"); return 1;
        }
        _mm_free(reference); _mm_free(dense);
    }
    _mm_free(actual); _mm_free(packed); _mm_free(weights);
    free(snapshot); _mm_free(allocation);
    return 0;
}

int main(void)
{
    int cases = 0;
    const int widths[] = {1, 2, 3, 4, 5, 7, 8, 9};
    const int channels[] = {8, 16, 24, 32, 64};
    const int outputs[] = {16, 32, 48, 64, 80, 96, 128};
    for (int H = 1; H <= 5; H += 2)
        for (int wi = 0; wi < 8; ++wi)
            for (int stride = 1; stride <= 2; ++stride)
                for (int branch = 0; branch < 3; ++branch) {
                    int C = channels[cases % 5], OC = outputs[cases % 7];
                    int offset = branch == 0 ? 0 : (branch == 1 ? C : 3);
                    int pitch = branch == 2 ? 2 * C + 3 : 2 * C;
                    int OW = (widths[wi] - 1) / stride + 1;
                    if (check_case(H, widths[wi], C, OC, stride, pitch, offset,
                                   branch == 2, OW % 4 == 0)) return 1;
                    ++cases;
                }
    if (check_case(5, 8, 16, 48, 1, 16, 0, 0, 1)) return 1;
    ++cases;
    if (check_case(3, 5, 24, 80, 3, 51, 3, 1, 0) ||
        check_case(5, 9, 16, 64, 4, 32, 16, 0, 0) ||
        check_case(2, 3, 16, 48, 7, 32, 16, 0, 0)) return 1;
    cases += 3;
    /* Representative C2f shapes at 640, 1024, and 1280 input resolution. */
    const int shapes[][2] = {{160,16},{80,32},{40,64},{20,128},
                            {256,16},{128,32},{64,64},{32,128},
                            {320,16},{160,32},{80,64},{40,128}};
    for (size_t i = 0; i < sizeof(shapes)/sizeof(shapes[0]); ++i) {
        int H = shapes[i][0], C = shapes[i][1];
        if (check_case(H, H, C, C, 1, 2 * C, C, 0, 1)) return 1;
        ++cases;
    }
    float *p = _mm_malloc(64 * sizeof(float), 64);
#define BAD(...) do { if (conv_3x3_strided(__VA_ARGS__) != NULL) { \
    fprintf(stderr, "FAIL: invalid arguments accepted at line %d\n", __LINE__); return 1; } } while (0)
    BAD(NULL, 1, 1, 16, p, 16, 1, 32, 16);
    BAD(p, 1, 1, 16, NULL, 16, 1, 32, 16);
    BAD(p, 0, 1, 16, p, 16, 1, 32, 16);
    BAD(p, 1, 1, 16, p, 16, 0, 32, 16);
    BAD(p, 1, 1, 7, p, 16, 1, 32, 16);
    BAD(p, 1, 1, 16, p, 24, 1, 32, 16);
    BAD(p, 1, 1, 16, p, 16, 1, 15, 0);
    BAD(p, 1, 1, 16, p, 16, 1, 32, 17);
    BAD(p, 1, 1, 16, p, 16, 1, 32, -1);
    BAD(p, 1, INT_MAX, 16, p, 16, 1, 32, 16);
    BAD(p, 1, 1, 16, p, 16, INT_MAX, 32, 16);
    BAD(p, 1, 1, 1 << 28, p, 16, 1, 1 << 28, 0);
    _mm_free(p);
    printf("PASS: %d scalar-reference cases, copied-chunk equivalence, input immutability, 12 invalid-argument cases\n", cases);
    return 0;
}
