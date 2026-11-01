#include "../include/batchnorm.h"
#include <torch/torch.h>
#include <torch/nn/functional.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>

extern "C" int fast_maxpool2d(
    const float *, float *, int, int, int, int,
    int, int, int, int, int, int);

static uint64_t tsc()
{
    unsigned aux;
    _mm_lfence();
    uint64_t value = __rdtscp(&aux);
    _mm_lfence();
    return value;
}

static void pin_cpu(int cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        perror("sched_setaffinity");
        exit(EXIT_FAILURE);
    }
}

static void run_custom(
    const float *input, float *output, float *scale, float *bias,
    int N, int H, int W, int C, int repeats, double flops)
{
    for (int i = 0; i < 1000; ++i) {
        if (fast_batchnorm_apply_unrolled(
                input, output, N, H, W, C, scale, bias) != 0)
            exit(EXIT_FAILURE);
    }
    uint64_t start = tsc();
    for (int i = 0; i < repeats; ++i) {
        fast_batchnorm_apply_unrolled(
            input, output, N, H, W, C, scale, bias);
    }
    double cycles = (double)(tsc() - start) / repeats;
    printf("custom_batchnorm_unrolled,%.3f,%.6f,%.6f\n",
           cycles, flops / cycles, (flops / cycles) / 32.0);
}

static void run_custom_full(
    const float *input, float *output,
    const float *mean, const float *variance,
    const float *gamma, const float *beta,
    int N, int H, int W, int C, int repeats, double flops)
{
    for (int i = 0; i < 1000; ++i) {
        if (fast_batchnorm(input, output, N, H, W, C,
                           mean, variance, gamma, beta, 1e-5f) != 0) {
            exit(EXIT_FAILURE);
        }
    }
    uint64_t start = tsc();
    for (int i = 0; i < repeats; ++i) {
        fast_batchnorm(input, output, N, H, W, C,
                       mean, variance, gamma, beta, 1e-5f);
    }
    double cycles = (double)(tsc() - start) / repeats;
    printf("custom_batchnorm_full,%.3f,%.6f,%.6f\n",
           cycles, flops / cycles, (flops / cycles) / 32.0);
}

static void run_custom_maxpool(
    const float *input, float *output, int N, int H, int W, int C,
    int repeats, double comparisons)
{
    for (int i = 0; i < 1000; ++i) {
        if (fast_maxpool2d(input, output, N, H, W, C,
                           5, 5, 1, 1, 2, 2) != 0) {
            exit(EXIT_FAILURE);
        }
    }
    uint64_t start = tsc();
    for (int i = 0; i < repeats; ++i) {
        fast_maxpool2d(input, output, N, H, W, C, 5, 5, 1, 1, 2, 2);
    }
    double cycles = (double)(tsc() - start) / repeats;
    printf("custom_maxpool,%.3f,%.6f,NA\n",
           cycles, comparisons / cycles);
}

int main(int argc, char **argv)
{
    const int cpu = argc > 1 ? atoi(argv[1]) : 0;
    const int repeats = argc > 2 ? atoi(argv[2]) : 100000;
    const int N = 1, H = 8, W = 8, C = 16;
    const int elements = N * H * W * C;
    pin_cpu(cpu);
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    torch::NoGradGuard no_grad;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto x = torch::randn({N, C, H, W}, options);
    auto torch_mean = torch::randn({C}, options);
    auto torch_variance = torch::rand({C}, options) + 0.5;
    auto torch_gamma = torch::randn({C}, options);
    auto torch_beta = torch::randn({C}, options);
    namespace F = torch::nn::functional;
    auto bn_options = F::BatchNormFuncOptions()
        .weight(torch_gamma).bias(torch_beta).training(false).eps(1e-5);
    auto pool_options = F::MaxPool2dFuncOptions({5, 5})
        .stride({1, 1}).padding({2, 2});

    for (int i = 0; i < 1000; ++i) {
        auto y = F::batch_norm(x, torch_mean, torch_variance, bn_options);
        auto z = F::max_pool2d(x, pool_options);
        (void)y;
        (void)z;
    }

    const double bn_flops = (double)elements * 2.0;
    const double pool_comparisons =
        (double)N * H * W * C * 24.0;
    printf("shape=%dx%dx%dx%d,cpu=%d,repeats=%d\n",
           N, H, W, C, cpu, repeats);
    puts("kernel,tsc_cycles,work_per_tsc_cycle,peak_utilization");

    uint64_t start = tsc();
    for (int i = 0; i < repeats; ++i) {
        auto y = F::batch_norm(x, torch_mean, torch_variance, bn_options);
        (void)y;
    }
    double bn_cycles = (double)(tsc() - start) / repeats;
    printf("libtorch_batchnorm,%.3f,%.6f,%.6f\n",
           bn_cycles, bn_flops / bn_cycles,
           (bn_flops / bn_cycles) / 32.0);

    start = tsc();
    for (int i = 0; i < repeats; ++i) {
        auto z = F::max_pool2d(x, pool_options);
        (void)z;
    }
    double pool_cycles = (double)(tsc() - start) / repeats;
    printf("libtorch_maxpool,%.3f,%.6f,NA\n",
           pool_cycles, pool_comparisons / pool_cycles);

    const size_t blocked_count = (size_t)N * H * W * 16;
    float *input = (float *)aligned_alloc(64, blocked_count * sizeof(float));
    float *output = (float *)aligned_alloc(64, blocked_count * sizeof(float));
    float *scale = (float *)aligned_alloc(64, 16 * sizeof(float));
    float *bias = (float *)aligned_alloc(64, 16 * sizeof(float));
    float *pool_output = (float *)aligned_alloc(
        64, blocked_count * sizeof(float));
    float *mean = (float *)aligned_alloc(64, 16 * sizeof(float));
    float *variance = (float *)aligned_alloc(64, 16 * sizeof(float));
    float *gamma = (float *)aligned_alloc(64, 16 * sizeof(float));
    float *beta = (float *)aligned_alloc(64, 16 * sizeof(float));
    if (!input || !output || !scale || !bias || !pool_output ||
        !mean || !variance || !gamma || !beta)
        return EXIT_FAILURE;
    for (size_t i = 0; i < blocked_count; ++i) input[i] = (float)(i % 31) * 0.01f;
    for (int i = 0; i < C; ++i) {
        mean[i] = 0.1f * i;
        variance[i] = 1.0f + 0.01f * i;
        gamma[i] = 1.0f;
        beta[i] = 0.0f;
    }
    if (fast_batchnorm_prepare(
            scale, bias, C, mean, variance, gamma, beta, 1e-5f)) {
        return EXIT_FAILURE;
    }
    run_custom_full(input, output, mean, variance, gamma, beta,
                    N, H, W, C, repeats, bn_flops);
    run_custom(input, output, scale, bias, N, H, W, C, repeats, bn_flops);
    run_custom_maxpool(input, pool_output, N, H, W, C, repeats,
                       pool_comparisons);
    free(input);
    free(output);
    free(scale);
    free(bias);
    free(pool_output);
    free(mean);
    free(variance);
    free(gamma);
    free(beta);
    return 0;
}
