/* Standalone production benchmark: 11 YOLOv8n joins at three resolutions.
 * Allocation, generation, validation and cache eviction stay outside timing.
 * Use taskset before starting this program to pin both allocation and work.
 */
#define _GNU_SOURCE
#include "universal_concat_nhwc.h"
#include "universal_concat_workloads.h"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <sched.h>
#include <unistd.h>
#endif

static volatile uint64_t digest_sink;
static uint32_t random_state = 0x725b691du;
static uint32_t next_random(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
}
static double now(void) {
#if defined(_WIN32)
    LARGE_INTEGER t, f;
    QueryPerformanceCounter(&t);
    QueryPerformanceFrequency(&f);
    return (double)t.QuadPart / (double)f.QuadPart;
#else
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
#endif
}
static float *allocate(size_t n) {
    void *p = NULL;
    if (n > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "allocation size overflow\n");
        exit(1);
    }
#if defined(_WIN32)
    p = _aligned_malloc(n * sizeof(float), 64);
#else
    if (posix_memalign(&p, 64, n * sizeof(float)) != 0)
        p = NULL;
#endif
    if (!p) {
        fprintf(stderr, "allocation failed\n");
        exit(1);
    }
    return (float *)p;
}
static void release(void *p) {
#if defined(_WIN32)
    _aligned_free(p);
#else
    free(p);
#endif
}
static void fill(float *p, size_t n) {
    for (size_t j = 0; j < n; ++j) {
        uint32_t x = next_random();
        memcpy(p + j, &x, sizeof(x));
    }
}
static void evict(unsigned char *p, size_t bytes) {
    volatile unsigned char *v = p;
    uint64_t sum = 0;
    for (size_t j = 0; j < bytes; j += 64) {
        v[j] = (unsigned char)(v[j] + 1);
        sum += v[j];
    }
    digest_sink ^= sum;
}
/* Independent byte order oracle, including arbitrary FP32 representations. */
static void oracle(const uc_input_view *v, size_t k, float *d, size_t n) {
    for (size_t p = 0; p < n; ++p)
        for (size_t i = 0; i < k; ++i)
            for (size_t c = 0; c < v[i].channels; ++c) {
                memcpy(d, v[i].data + p * v[i].pixel_stride + c, sizeof(float));
                ++d;
            }
}
static int compare_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static unsigned parse(const char *s) {
    char *end;
    errno = 0;
    unsigned long n = strtoul(s, &end, 10);
    if (errno || !*s || *end || n < 1 || n > 100000) {
        fprintf(stderr, "invalid positive argument: %s\n", s);
        exit(2);
    }
    return (unsigned)n;
}
int main(int argc, char **argv) {
    unsigned rounds = 21, calls = 7, repeat = 1;
    int cold = 0, calls_set = 0, only_site = 0, only_image = 0;
    for (int j = 1; j < argc; ++j) {
        if (!strcmp(argv[j], "--cold"))
            cold = 1;
        else if (j + 1 < argc && !strcmp(argv[j], "--rounds"))
            rounds = parse(argv[++j]);
        else if (j + 1 < argc && !strcmp(argv[j], "--calls")) {
            calls = parse(argv[++j]);
            calls_set = 1;
        } else if (j + 1 < argc && !strcmp(argv[j], "--repeat"))
            repeat = parse(argv[++j]);
        else if (j + 1 < argc && !strcmp(argv[j], "--site"))
            only_site = (int)parse(argv[++j]);
        else if (j + 1 < argc && !strcmp(argv[j], "--image"))
            only_image = (int)parse(argv[++j]);
        else {
            fprintf(
                stderr,
                "usage: %s [--cold] [--rounds N] [--calls N] [--repeat N] [--site N] [--image N]\n",
                argv[0]);
            return 2;
        }
    }
    if (cold && !calls_set)
        calls = 1;
    if (only_image && only_image % 32) {
        fprintf(stderr, "inventory image sizes must be divisible by32; the kernel accepts "
                        "arbitrary dimensions\n");
        return 2;
    }
    if (only_site) {
        int found = 0;
        for (size_t i = 0; i < UC_SITE_COUNT; ++i)
            if (uc_workloads[i].site == only_site)
                found = 1;
        if (!found) {
            fprintf(stderr, "site not in YOLOv8n inventory\n");
            return 2;
        }
    }
    char hostname[256] = "local";
#if !defined(_WIN32)
    gethostname(hostname, sizeof(hostname) - 1);
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    if (sched_getaffinity(0, sizeof(affinity), &affinity) == 0)
        fprintf(stderr, "hostname=%s affinity_count=%d current_cpu=%d rounds=%u calls=%u mode=%s\n",
                hostname, CPU_COUNT(&affinity), sched_getcpu(), rounds, calls,
                cold ? "cold" : "warm");
#endif
    const size_t eviction_bytes = 64u * 1024u * 1024u;
    unsigned char *scratch = (unsigned char *)allocate(eviction_bytes / sizeof(float));
    memset(scratch, 1, eviction_bytes);
    puts("image_size,site,H,W,input_count,out_channels,repeat,median_ms,p10_ms,p90_ms,useful_GBps,"
         "status,mode,hostname");
    const int images[] = {640, 1024, 2048};
    for (size_t r = 0; r < (only_image ? 1u : 3u); ++r)
        for (size_t wi = 0; wi < UC_SITE_COUNT; ++wi) {
            const uc_workload *a = &uc_workloads[wi];
            int image = only_image ? only_image : images[r];
            if (only_site && a->site != only_site)
                continue;
            size_t side = (size_t)image / a->scale, n = side * side, co = 0;
            for (size_t i = 0; i < a->count; ++i)
                co += a->channels[i];
            if (n > SIZE_MAX / co || n * co > SIZE_MAX / sizeof(float)) {
                fprintf(stderr, "shape overflow\n");
                return 2;
            }
            uc_input_view v[4];
            float *src[4];
            for (size_t i = 0; i < a->count; ++i) {
                size_t c = a->channels[i];
                src[i] = allocate(n * c);
                fill(src[i], n * c);
                v[i] = (uc_input_view){src[i], c, c, n * c};
            }
            size_t no = n * co;
            float *out = allocate(no), *expected = allocate(no);
            double *samples = (double *)calloc(rounds, sizeof(double));
            if (!samples)
                return 1;
            oracle(v, a->count, expected, n);
            for (int w = 0; w < 5; ++w)
                if (universal_concat_nhwc_f32(v, a->count, out, no, 1, side, side) != UC_OK)
                    return 1;
            for (unsigned round = 0; round < rounds; ++round) {
                double elapsed = 0;
                if (cold) {
                    for (unsigned c = 0; c < calls; ++c) {
                        evict(scratch, eviction_bytes);
                        double start = now();
                        int status = universal_concat_nhwc_f32(v, a->count, out, no, 1, side, side);
                        double stop = now();
                        if (status != UC_OK)
                            return 1;
                        elapsed += stop - start;
                    }
                } else {
                    /* Restore same-buffer state after validation of the previous batch. */
                    if (universal_concat_nhwc_f32(v, a->count, out, no, 1, side, side) != UC_OK)
                        return 1;
                    int status = 0;
                    double start = now();
                    for (unsigned c = 0; c < calls; ++c)
                        status |= universal_concat_nhwc_f32(v, a->count, out, no, 1, side, side);
                    double stop = now();
                    if (status != UC_OK)
                        return 1;
                    elapsed = stop - start;
                }
                samples[round] = elapsed / (double)calls;
                if (memcmp(out, expected, no * sizeof(float))) {
                    fprintf(stderr, "incorrect output\n");
                    return 1;
                }
            }
            qsort(samples, rounds, sizeof(double), compare_double);
            double med = samples[rounds / 2];
            printf("%d,%d,%zu,%zu,%zu,%zu,%u,%.9f,%.9f,%.9f,%.6f,PASS,%s,%s\n", image, a->site,
                   side, side, a->count, co, repeat, med * 1000, samples[(rounds - 1) / 10] * 1000,
                   samples[rounds - 1 - (rounds - 1) / 10] * 1000,
                   (double)no * 2 * sizeof(float) / med / 1e9, cold ? "cold" : "warm", hostname);
            fflush(stdout);
            free(samples);
            release(out);
            release(expected);
            for (size_t i = 0; i < a->count; ++i)
                release(src[i]);
        }
    release(scratch);
    return 0;
}
