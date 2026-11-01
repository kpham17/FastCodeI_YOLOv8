#define _GNU_SOURCE

#include "../include/batchnorm.h"
#include "../include/maxpool.h"
#include <errno.h>
#include <linux/perf_event.h>
#include <math.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>

static volatile float sink;

typedef struct {
    int cycles;
    int reference_cycles;
} counters;

static int open_counter(unsigned config)
{
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type = PERF_TYPE_HARDWARE;
    attr.size = sizeof(attr);
    attr.config = config;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    return (int)syscall(
        SYS_perf_event_open, &attr, 0, -1, -1, 0);
}

static counters open_counters(void)
{
    counters result = {
        open_counter(PERF_COUNT_HW_CPU_CYCLES),
        open_counter(PERF_COUNT_HW_REF_CPU_CYCLES)
    };
    return result;
}

static void reset_counters(counters c)
{
    if (c.cycles >= 0) ioctl(c.cycles, PERF_EVENT_IOC_RESET, 0);
    if (c.reference_cycles >= 0) {
        ioctl(c.reference_cycles, PERF_EVENT_IOC_RESET, 0);
    }
}

static void start_counters(counters c)
{
    if (c.cycles >= 0) ioctl(c.cycles, PERF_EVENT_IOC_ENABLE, 0);
    if (c.reference_cycles >= 0) {
        ioctl(c.reference_cycles, PERF_EVENT_IOC_ENABLE, 0);
    }
}

static void stop_counters(counters c)
{
    if (c.cycles >= 0) ioctl(c.cycles, PERF_EVENT_IOC_DISABLE, 0);
    if (c.reference_cycles >= 0) {
        ioctl(c.reference_cycles, PERF_EVENT_IOC_DISABLE, 0);
    }
}

static uint64_t read_counter(int fd)
{
    uint64_t value = 0;
    if (fd < 0 || read(fd, &value, sizeof(value)) != sizeof(value)) return 0;
    return value;
}

static void print_metric(double value)
{
    if (value > 0.0) printf("%.6f", value);
    else fputs("NA", stdout);
}

static uint64_t now_ns(void)
{
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &time) != 0) {
        perror("clock_gettime");
        exit(EXIT_FAILURE);
    }
    return (uint64_t)time.tv_sec * 1000000000ull + (uint64_t)time.tv_nsec;
}

static uint64_t read_tsc(void)
{
    unsigned aux;
    _mm_lfence();
    uint64_t ticks = __rdtscp(&aux);
    _mm_lfence();
    return ticks;
}

static void pin_cpu(int cpu)
{
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        fprintf(stderr, "sched_setaffinity(%d): %s\n", cpu, strerror(errno));
        exit(EXIT_FAILURE);
    }
}

static float *allocate_floats(size_t count)
{
    float *result = NULL;
    if (posix_memalign((void **)&result, 64, count * sizeof(*result)) != 0) {
        perror("posix_memalign");
        exit(EXIT_FAILURE);
    }
    return result;
}

static void run_maxpool_tile(
    const float *input, float *output, int width)
{
    maxpool_tile_5x5_chw16_avx512(
        input + ((size_t)2 * width + 2) * 16,
        output + ((size_t)2 * width + 2) * 16,
        width * 16, width * 16);
}

int main(int argc, char **argv)
{
    const int cpu = argc > 1 ? atoi(argv[1]) : 0;
    const int N = 1, H = 14, W = 16, C = 16;
    const int blocks = (C + 15) / 16;
    const int H_out = H, W_out = W;
    const size_t input_count = (size_t)N * blocks * H * W * 16;
    const size_t output_count = (size_t)N * blocks * H_out * W_out * 16;
    const size_t parameter_count = (size_t)C;
    float *input = allocate_floats(input_count);
    float *batchnorm_output = allocate_floats(output_count);
    float *batchnorm_unrolled_output = allocate_floats(output_count);
    float *batchnorm_nocopy_output = allocate_floats(output_count);
    float *maxpool_output = allocate_floats(output_count);
    float *mean = allocate_floats(parameter_count);
    float *variance = allocate_floats(parameter_count);
    float *gamma = allocate_floats(parameter_count);
    float *beta = allocate_floats(parameter_count);
    float *scale = allocate_floats((size_t)blocks * 16);
    float *bias = allocate_floats((size_t)blocks * 16);

    pin_cpu(cpu);
    for (size_t i = 0; i < input_count; ++i) input[i] = (float)(i % 31) * 0.01f;
    for (int c = 0; c < C; ++c) {
        mean[c] = 0.1f * c;
        variance[c] = 1.0f + 0.01f * c;
        gamma[c] = 1.0f;
        beta[c] = 0.0f;
    }
    if (fast_batchnorm_prepare(
            scale, bias, C, mean, variance, gamma, beta, 1e-5f) != 0) {
        fprintf(stderr, "batchnorm preparation failed\n");
        return EXIT_FAILURE;
    }

    const int warmups = 1000;
    const int repeats = 10000;
    for (int i = 0; i < warmups; ++i) {
        if (fast_batchnorm_apply(
                input, batchnorm_output, N, H, W, C, scale, bias) != 0) {
            fprintf(stderr, "batchnorm validation failed during warmup\n");
            return EXIT_FAILURE;
        }
        if (fast_batchnorm_apply_unrolled(
                input, batchnorm_unrolled_output, N, H, W, C,
                scale, bias) != 0) {
            fprintf(stderr, "unrolled batchnorm validation failed during warmup\n");
            return EXIT_FAILURE;
        }
        if (fast_batchnorm_apply_unrolled_nocopy(
                input, batchnorm_nocopy_output, N, H, W, C,
                scale, bias) != 0) {
            fprintf(stderr, "no-copy batchnorm validation failed during warmup\n");
            return EXIT_FAILURE;
        }
        run_maxpool_tile(input, maxpool_output, W);
        if (!isfinite(maxpool_output[((size_t)2 * W + 2) * 16])) {
            fprintf(stderr, "maxpool validation failed during warmup\n");
            return EXIT_FAILURE;
        }
    }

    counters bn_counter = open_counters();
    reset_counters(bn_counter);
    start_counters(bn_counter);
    uint64_t start_ns = now_ns();
    uint64_t start_tsc = read_tsc();
    for (int i = 0; i < repeats; ++i) {
        if (fast_batchnorm_apply(
                input, batchnorm_output, N, H, W, C, scale, bias) != 0) {
            return EXIT_FAILURE;
        }
    }
    uint64_t bn_tsc = read_tsc();
    uint64_t bn_ns = now_ns();
    stop_counters(bn_counter);
    const uint64_t bn_core = read_counter(bn_counter.cycles);
    const uint64_t bn_reference = read_counter(bn_counter.reference_cycles);
    counters unrolled_counter = open_counters();
    reset_counters(unrolled_counter);
    start_counters(unrolled_counter);
    uint64_t unrolled_start_ns = now_ns();
    uint64_t unrolled_start_tsc = read_tsc();
    for (int i = 0; i < repeats; ++i) {
        if (fast_batchnorm_apply_unrolled(
                input, batchnorm_unrolled_output, N, H, W, C,
                scale, bias) != 0) {
            return EXIT_FAILURE;
        }
    }
    uint64_t unrolled_tsc = read_tsc();
    uint64_t unrolled_ns = now_ns();
    stop_counters(unrolled_counter);
    const uint64_t unrolled_core = read_counter(unrolled_counter.cycles);
    const uint64_t unrolled_reference =
        read_counter(unrolled_counter.reference_cycles);
    counters nocopy_counter = open_counters();
    reset_counters(nocopy_counter);
    start_counters(nocopy_counter);
    uint64_t nocopy_start_ns = now_ns();
    uint64_t nocopy_start_tsc = read_tsc();
    for (int i = 0; i < repeats; ++i) {
        if (fast_batchnorm_apply_unrolled_nocopy(
                input, batchnorm_nocopy_output, N, H, W, C,
                scale, bias) != 0) {
            return EXIT_FAILURE;
        }
    }
    uint64_t nocopy_tsc = read_tsc();
    uint64_t nocopy_ns = now_ns();
    stop_counters(nocopy_counter);
    const uint64_t nocopy_core = read_counter(nocopy_counter.cycles);
    const uint64_t nocopy_reference =
        read_counter(nocopy_counter.reference_cycles);
    counters pool_counter = open_counters();
    reset_counters(pool_counter);
    start_counters(pool_counter);
    for (int i = 0; i < repeats; ++i) {
        run_maxpool_tile(input, maxpool_output, W);
    }
    uint64_t pool_tsc = read_tsc();
    uint64_t end_ns = now_ns();
    stop_counters(pool_counter);
    const uint64_t pool_core = read_counter(pool_counter.cycles);
    const uint64_t pool_reference = read_counter(pool_counter.reference_cycles);

    sink = batchnorm_output[input_count / 2] + maxpool_output[output_count / 2];
    const double bn_ns_call = (double)(bn_ns - start_ns) / repeats;
    const double pool_ns_call = (double)(end_ns - bn_ns) / repeats;
    const double bn_cycles = (double)(bn_tsc - start_tsc) / repeats;
    const double unrolled_ns_call =
        (double)(unrolled_ns - unrolled_start_ns) / repeats;
    const double unrolled_cycles =
        (double)(unrolled_tsc - unrolled_start_tsc) / repeats;
    const double pool_cycles = (double)(pool_tsc - bn_tsc) / repeats;
    const double bn_reference_per_call = (double)bn_reference / repeats;
    const double bn_core_per_call = (double)bn_core / repeats;
    const double unrolled_reference_per_call =
        (double)unrolled_reference / repeats;
    const double unrolled_core_per_call = (double)unrolled_core / repeats;
    const double nocopy_ns_call =
        (double)(nocopy_ns - nocopy_start_ns) / repeats;
    const double nocopy_cycles =
        (double)(nocopy_tsc - nocopy_start_tsc) / repeats;
    const double nocopy_reference_per_call =
        (double)nocopy_reference / repeats;
    const double nocopy_core_per_call = (double)nocopy_core / repeats;
    const double pool_reference_per_call = (double)pool_reference / repeats;
    const double pool_core_per_call = (double)pool_core / repeats;
    printf("shape=%dx%dx%dx%d,cpu=%d,l1_working_set=%zu\n",
           N, H, W, C, cpu,
           (input_count + output_count + parameter_count * 4) * sizeof(float));
    const double batchnorm_flops = (double)N * H * W * C * 2;
    const double batchnorm_peak_flops_per_cycle = 32.0;
    /*
     * The tile computes 8 outputs x 16 lanes, reducing 25 values per lane.
     * Maxpool has no arithmetic FLOPs; this is a comparison-equivalent
     * operation count used only to normalize its throughput.
     */
    const double maxpool_comparisons = 25.0 * 16.0 * 24.0;
    const double maxpool_peak_comparisons_per_core_cycle = 16.0;
    printf("kernel,ns_per_call,tsc_cycles,reference_cycles,core_cycles,"
           "work_per_tsc_cycle,work_per_reference_cycle,work_per_core_cycle\n");
    printf("batchnorm,%.3f,%.3f,", bn_ns_call, bn_cycles);
    print_metric(bn_reference_per_call);
    putchar(',');
    print_metric(bn_core_per_call);
    putchar(',');
    print_metric(bn_cycles > 0.0 ?
                 batchnorm_flops / bn_cycles : -1.0);
    putchar(',');
    print_metric(bn_reference_per_call > 0.0 ?
                 batchnorm_flops / bn_reference_per_call : -1.0);
    putchar(',');
    print_metric(bn_core_per_call > 0.0 ?
                 batchnorm_flops / bn_core_per_call : -1.0);
    putchar('\n');
    printf("batchnorm_unrolled_nocopy,%.3f,%.3f,", nocopy_ns_call,
           nocopy_cycles);
    print_metric(nocopy_reference_per_call);
    putchar(',');
    print_metric(nocopy_core_per_call);
    putchar(',');
    print_metric(nocopy_cycles > 0.0 ?
                 batchnorm_flops / nocopy_cycles : -1.0);
    putchar(',');
    print_metric(nocopy_reference_per_call > 0.0 ?
                 batchnorm_flops / nocopy_reference_per_call : -1.0);
    putchar(',');
    print_metric(nocopy_core_per_call > 0.0 ?
                 batchnorm_flops / nocopy_core_per_call : -1.0);
    putchar('\n');
    printf("batchnorm_unrolled,%.3f,%.3f,", unrolled_ns_call,
           unrolled_cycles);
    print_metric(unrolled_reference_per_call);
    putchar(',');
    print_metric(unrolled_core_per_call);
    putchar(',');
    print_metric(unrolled_cycles > 0.0 ?
                 batchnorm_flops / unrolled_cycles : -1.0);
    putchar(',');
    print_metric(unrolled_reference_per_call > 0.0 ?
                 batchnorm_flops / unrolled_reference_per_call : -1.0);
    putchar(',');
    print_metric(unrolled_core_per_call > 0.0 ?
                 batchnorm_flops / unrolled_core_per_call : -1.0);
    putchar('\n');
    printf("maxpool_tile_5x5,%.3f,%.3f,", pool_ns_call, pool_cycles);
    print_metric(pool_reference_per_call);
    putchar(',');
    print_metric(pool_core_per_call);
    putchar(',');
    print_metric(pool_cycles > 0.0 ?
                 maxpool_comparisons / pool_cycles : -1.0);
    putchar(',');
    print_metric(pool_reference_per_call > 0.0 ?
                 maxpool_comparisons / pool_reference_per_call : -1.0);
    putchar(',');
    print_metric(pool_core_per_call > 0.0 ?
                 maxpool_comparisons / pool_core_per_call : -1.0);
    putchar('\n');
    printf("batchnorm_peak_flops_per_core_cycle=%.1f,"
           "batchnorm_core_cycle_utilization=%.6f,"
           "maxpool_peak_comparisons_per_core_cycle=%.1f,"
           "maxpool_comparison_equiv_flops_per_tsc_cycle=%.6f,"
           "maxpool_comparisons_per_core_cycle=%.6f,"
           "maxpool_core_cycle_utilization=%.6f\n",
           batchnorm_peak_flops_per_cycle,
           bn_core_per_call > 0.0 ?
           (batchnorm_flops / bn_core_per_call) /
               batchnorm_peak_flops_per_cycle : 0.0,
           maxpool_peak_comparisons_per_core_cycle,
           pool_cycles > 0.0 ? maxpool_comparisons / pool_cycles : 0.0,
           pool_core_per_call > 0.0 ?
           maxpool_comparisons / pool_core_per_call : 0.0,
           pool_core_per_call > 0.0 ?
           (maxpool_comparisons / pool_core_per_call) /
               maxpool_peak_comparisons_per_core_cycle : 0.0);

    free(input);
    free(batchnorm_output);
    free(batchnorm_unrolled_output);
    free(batchnorm_nocopy_output);
    free(maxpool_output);
    free(mean);
    free(variance);
    free(gamma);
    free(beta);
    free(scale);
    free(bias);
    if (bn_counter.cycles >= 0) close(bn_counter.cycles);
    if (bn_counter.reference_cycles >= 0) close(bn_counter.reference_cycles);
    if (pool_counter.cycles >= 0) close(pool_counter.cycles);
    if (pool_counter.reference_cycles >= 0) {
        close(pool_counter.reference_cycles);
    }
    if (unrolled_counter.cycles >= 0) close(unrolled_counter.cycles);
    if (unrolled_counter.reference_cycles >= 0) {
        close(unrolled_counter.reference_cycles);
    }
    if (nocopy_counter.cycles >= 0) close(nocopy_counter.cycles);
    if (nocopy_counter.reference_cycles >= 0) {
        close(nocopy_counter.reference_cycles);
    }
    return 0;
}
