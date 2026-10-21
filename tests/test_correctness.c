#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../include/conv2D_512.h"
#include "../include/silu.h"

#define CHECK(condition, message)                                      \
    do {                                                               \
        if (!(condition)) {                                            \
            fprintf(stderr, "FAIL: %s (line %d)\n", message, __LINE__); \
            return 1;                                                  \
        }                                                              \
    } while (0)

static float test_value(int i)
{
    return (float)((i * 17) % 23 - 11) * 0.025f;
}

static int close_enough(float actual, float expected, float atol, float rtol)
{
    int i = isfinite(actual) &&
           fabsf(actual - expected) <= atol + rtol * fabsf(expected);
    //if (!i){
    //  printf("%f %f\n", actual, expected);
    //}
    return i;
}

static int test_silu(void)
{
    const int H = 1, W = 7, C = 16;
    const int count = H * W * C;
    float *input = (float *)_mm_malloc((size_t)count * sizeof(float), 32);

    CHECK(input != NULL, "allocate SiLU input");

    for (int i = 0; i < count; ++i) {
        input[i] = -6.0f + 12.0f * (float)i / (float)(count - 1);
    }

    float *actual = fast_SILU(input, H, W, C);
    CHECK(actual != NULL, "fast_SILU returned NULL");

    for (int i = 0; i < count; ++i) {
        const float x = input[i];
        const float expected = x / (1.0f + expf(-x));
        
        CHECK(close_enough(actual[i], expected, 0.025f, 0.02f),
              "SiLU differs from scalar reference");
    }
    printf("SiLU done \n");
    _mm_free(actual);
    _mm_free(input);
    return 0;
}

static int test_conv_1x1(void)
{
    /* Width is divisible by 4; output channels match the 4x24 kernel. */
    const int H = 1, W = 4, IC = 32, OC = 32;
    const int pixels = H * W;
    float *input = (float *)_mm_malloc(
        (size_t)pixels * IC * sizeof(float), 32);
    float *kernel = (float *)_mm_malloc(
        (size_t)IC * OC * sizeof(float), 32);

    CHECK(input != NULL && kernel != NULL, "allocate 1x1 convolution inputs");

    for (int i = 0; i < pixels * IC; ++i) input[i] = test_value(i);
    for (int i = 0; i < IC * OC; ++i) kernel[i] = test_value(i + 3);

    float *actual = conv_1x1(input, H, W, IC, kernel, OC);
    CHECK(actual != NULL, "conv_1x1 returned NULL");

    for (int p = 0; p < pixels; ++p) {
        for (int oc = 0; oc < OC; ++oc) {
            float expected = 0.0f;
            for (int ic = 0; ic < IC; ++ic) {
                expected += input[p * IC + ic] * kernel[ic * OC + oc];
            }
            //printf("%f %f\n", actual[p*OC+oc], expected);            

            CHECK(close_enough(actual[p * OC + oc], expected, 0.002f, 0.002f),
                  "1x1 convolution differs from scalar reference");
        }
    }
    printf("Conv 1x1 done \n");

    _mm_free(actual);
    _mm_free(kernel);
    _mm_free(input);
    return 0;
}


static int test_conv_3x3(void)
{
    const int H = 20, W = 16, IC = 16, OC = 16, stride = 2;
    const int output_h = (H - 1) / stride + 1;
    const int output_w = (W - 1) / stride + 1;
    const int pixels = H * W;
    const int resized_IC = (IC +7)/8*8;
    const int kernel_count = 3 * 3 * IC * OC;
    const int resized_kernel_count = 3 * 3 * resized_IC * OC;

    float *input = (float *)_mm_malloc(
        (size_t)pixels * IC * sizeof(float), 32);
    float *kernel = (float *)_mm_malloc(
        (size_t)kernel_count * sizeof(float), 32);
    float *kernel_resized = (float *)_mm_malloc(
        (size_t)resized_kernel_count * sizeof(float), 32);
    memset(kernel_resized, 0, resized_kernel_count * sizeof(float));
        
    float *w_input = input;
    CHECK(input != NULL && kernel != NULL, "allocate 3x3 convolution inputs");

    for (int i = 0; i < pixels * IC; ++i) input[i] = test_value(i + 5);
    for (int i = 0; i < kernel_count; ++i) kernel[i] = test_value(i + 9);
    
    if (IC != resized_IC){
        w_input = padinput_8x(input, H, W, IC);
    } 

    
    for (int kw = 0; kw < 3; ++kw) {
    for (int kh = 0; kh < 3; ++kh) {
        int k_spatial = kw * 3 + kh; 
        for (int ic = 0; ic < IC; ++ic) {
            for (int oc = 0; oc < OC; ++oc) {
                // Index mapping for original kernel
                int orig_idx = ((k_spatial * IC) + ic) * OC + oc;
                
                // Index mapping for resized/padded kernel
                int resized_idx = ((k_spatial * resized_IC) + ic) * OC + oc;
                
                kernel_resized[resized_idx] = kernel[orig_idx];
            }
        }
    }
    }

    float *actual = conv_3x3(w_input, H, W, resized_IC, kernel_resized, OC, stride);
    CHECK(actual != NULL, "conv_3x3 returned NULL");
    
    if (IC % 8 != 0){
        _mm_free(w_input);
    }
    

    for (int oh = 0; oh < output_h; ++oh) {
        for (int ow = 0; ow < output_w; ++ow) {
            for (int oc = 0; oc < OC; ++oc) {
                float expected = 0.0f;
                for (int kw = 0; kw < 3; ++kw) {
                    for (int kh = 0; kh < 3; ++kh) {
                        const int ih = oh * stride + kh - 1;
                        const int iw = ow * stride + kw - 1;

                        if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;

                        for (int ic = 0; ic < IC; ++ic) {
                            const int input_index = (ih * W + iw) * IC + ic;
                            const int kernel_index =
                                ((kw * 3 + kh) * IC + ic) * OC + oc;
                            expected += input[input_index] * kernel[kernel_index];
                        }
                    }
                }

                const int output_index = (oh * output_w + ow) * OC + oc;
                CHECK(close_enough(actual[output_index], expected, 0.003f, 0.003f),
                      "3x3 convolution differs from scalar reference");
            }
        }
    }

    _mm_free(actual);
    _mm_free(kernel);
    _mm_free(kernel_resized);
    _mm_free(input);
    
    printf("Conv 3x3 done \n");

    return 0;
}

int main(void)
{
    CHECK(test_silu() == 0, "SiLU test");
    CHECK(test_conv_1x1() == 0, "1x1 convolution test");
    CHECK(test_conv_3x3() == 0, "3x3 convolution test");

    puts("All enabled correctness tests passed.");
    return 0;
}