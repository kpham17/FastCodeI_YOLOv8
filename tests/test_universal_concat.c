#define _GNU_SOURCE
#include "universal_concat_nhwc.h"
#include "universal_concat_workloads.h"
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__unix__)
#include <sys/mman.h>
#include <unistd.h>
#endif

static size_t checks, cases;
#define CHECK(x)                                                                                   \
    do {                                                                                           \
        ++checks;                                                                                  \
        if (!(x)) {                                                                                \
            fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                                   \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
typedef struct {
    unsigned char *raw;
    float *data;
    size_t bytes, n;
} buffer;
static uint32_t rng = 0x713a9b51u;
static uint32_t random_bits(void) {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}
static void fill(float *a, size_t n) {
    const uint32_t special[] = {0,           0x80000000u, 0x7f800000u, 0xff800000u, 0x7fc01234u,
                                0x7fa54321u, 1,           0x80000001u, 0x007fffffu, 0xffffffffu};
    for (size_t j = 0; j < n; ++j) {
        uint32_t u = j % 17 < 10 ? special[j % 17] : random_bits();
        /* Rotate special values across independent streams. */
        if (j % 17 >= 10)
            u ^= (uint32_t)j;
        memcpy(a + j, &u, 4);
    }
}
static buffer alloc_buffer(size_t n, size_t offset) {
    buffer b;
    b.n = n;
    b.bytes = n * 4 + 256;
    b.raw = (unsigned char *)malloc(b.bytes);
    CHECK(b.raw);
    memset(b.raw, 0xa7, b.bytes);
    b.data = (float *)(((uintptr_t)b.raw + 127u) & ~(uintptr_t)63u) + offset;
    return b;
}
static void guards(buffer b) {
    unsigned char *start = (unsigned char *)b.data, *end = start + b.n * 4;
    for (unsigned char *p = b.raw; p < start; ++p)
        CHECK(*p == 0xa7);
    for (unsigned char *p = end; p < b.raw + b.bytes; ++p)
        CHECK(*p == 0xa7);
}
/* Deliberately independent byte oracle: no production helpers or SIMD. */
static void oracle(const uc_input_view *v, size_t k, float *out, size_t n) {
    unsigned char *d = (unsigned char *)out;
    for (size_t p = 0; p < n; ++p)
        for (size_t i = 0; i < k; ++i) {
            const unsigned char *s = (const unsigned char *)(v[i].data + p * v[i].pixel_stride);
            for (size_t j = 0; j < v[i].channels * 4; ++j)
                *d++ = s[j];
        }
}
static void verify_case(size_t b, size_t h, size_t w, size_t k, const size_t *channels, size_t pad,
                        size_t offset) {
    size_t n = b * h * w, co = 0;
    buffer in[9];
    uc_input_view v[9];
    unsigned char *snap[9];
    CHECK(k <= 9);
    for (size_t i = 0; i < k; ++i) {
        size_t stride = channels[i] + pad, span = (n - 1) * stride + channels[i];
        in[i] = alloc_buffer(span, (offset + i) % 9);
        fill(in[i].data, span);
        /* Force unique boundary words per stream and pixel. */
        for (size_t p = 0; p < n; ++p) {
            uint32_t a = (uint32_t)(0x12340000u + i * 613u + p * 17u);
            uint32_t z = (uint32_t)(0xabc00000u + i * 701u + p * 23u);
            memcpy(in[i].data + p * stride, &a, 4);
            memcpy(in[i].data + p * stride + channels[i] - 1, &z, 4);
        }
        v[i] = (uc_input_view){in[i].data, channels[i], stride, span};
        snap[i] = (unsigned char *)malloc(in[i].bytes);
        CHECK(snap[i]);
        memcpy(snap[i], in[i].raw, in[i].bytes);
        co += channels[i];
    }
    buffer out = alloc_buffer(n * co, offset), ref = alloc_buffer(n * co, 0);
    oracle(v, k, ref.data, n);
    {
        memset(out.data, 0x5d, n * co * 4);
        CHECK(universal_concat_nhwc_f32(v, k, out.data, n * co, b, h, w) == UC_OK);
        CHECK(memcmp(out.data, ref.data, n * co * 4) == 0);
        guards(out);
        for (size_t i = 0; i < k; ++i)
            CHECK(memcmp(in[i].raw, snap[i], in[i].bytes) == 0);
    }
    for (size_t i = 0; i < k; ++i) {
        guards(in[i]);
        free(in[i].raw);
        free(snap[i]);
    }
    free(out.raw);
    free(ref.raw);
    ++cases;
}
static void views(void) {
    const size_t widths[] = {1, 7, 16, 32, 64, 128};
    for (size_t ci = 0; ci < sizeof(widths) / sizeof(widths[0]); ++ci) {
        size_t c = widths[ci], n = 15;
        buffer cv = alloc_buffer(n * 2 * c, 0), bn = alloc_buffer(n * c * 3, 1);
        buffer x = alloc_buffer(n * 5 * c, 0), y = alloc_buffer(n * 5 * c, 3);
        fill(cv.data, cv.n);
        fill(bn.data, bn.n);
        uc_input_view split[] = {{cv.data, c, 2 * c, n * 2 * c},
                                 {cv.data + c, c, 2 * c, n * 2 * c - c},
                                 {bn.data, c, c, n * c},
                                 {bn.data + n * c, c, c, n * c},
                                 {bn.data + 2 * n * c, c, c, n * c}};
        uc_input_view pair[] = {{cv.data, 2 * c, 2 * c, n * 2 * c}, split[2], split[3], split[4]};
        CHECK(universal_concat_nhwc_f32(split, 5, x.data, x.n, 1, 3, 5) == 0);
        CHECK(universal_concat_nhwc_f32(pair, 4, y.data, y.n, 1, 3, 5) == 0);
        CHECK(memcmp(x.data, y.data, x.n * 4) == 0);
        oracle(split, 5, y.data, n);
        CHECK(memcmp(x.data, y.data, x.n * 4) == 0);
        guards(cv);
        guards(bn);
        guards(x);
        guards(y);
        free(cv.raw);
        free(bn.raw);
        free(x.raw);
        free(y.raw);
        ++cases;
    }
    float parent[40];
    fill(parent, 40);
    uc_input_view disjoint[] = {{parent, 3, 3, 6}, {parent + 6, 2, 2, 4}};
    float expected[10];
    oracle(disjoint, 2, expected, 2);
    CHECK(universal_concat_nhwc_f32(disjoint, 2, parent + 15, 10, 1, 1, 2) == 0);
    CHECK(memcmp(expected, parent + 15, sizeof(expected)) == 0);
}
static void reject(void) {
    float src[64], out[64], before[64];
    fill(src, 64);
    memset(out, 0xa5, sizeof(out));
    memcpy(before, out, sizeof(out));
    uc_input_view v = {src, 3, 5, 8};
#define BAD(expr, status)                                                                          \
    do {                                                                                           \
        CHECK((expr) == (status));                                                                 \
        CHECK(memcmp(out, before, sizeof(out)) == 0);                                              \
    } while (0)
#define CALL(vv, kk, oo, cap, b, h, w) universal_concat_nhwc_f32(vv, kk, oo, cap, b, h, w)
    BAD(CALL(NULL, 1, out, 64, 1, 1, 2), UC_BAD_ARGUMENT);
    BAD(CALL(&v, 0, out, 64, 1, 1, 2), UC_BAD_ARGUMENT);
    BAD(CALL(&v, 1, NULL, 64, 1, 1, 2), UC_BAD_ARGUMENT);
    BAD(CALL(&v, 1, out, 64, 0, 1, 2), UC_BAD_ARGUMENT);
    BAD(CALL(&v, 1, out, 64, 1, 0, 2), UC_BAD_ARGUMENT);
    BAD(CALL(&v, 1, out, 64, 1, 1, 0), UC_BAD_ARGUMENT);
    BAD(CALL(&v, 1, out, 5, 1, 1, 2), UC_INSUFFICIENT_CAPACITY);
    v.capacity = 7;
    BAD(CALL(&v, 1, out, 64, 1, 1, 2), UC_INSUFFICIENT_CAPACITY);
    v.capacity = 8;
    v.data = NULL;
    BAD(CALL(&v, 1, out, 64, 1, 1, 2), UC_BAD_ARGUMENT);
    v.data = src;
    v.channels = 0;
    BAD(CALL(&v, 1, out, 64, 1, 1, 2), UC_BAD_ARGUMENT);
    v.channels = 3;
    v.pixel_stride = 2;
    BAD(CALL(&v, 1, out, 64, 1, 1, 2), UC_BAD_ARGUMENT);
    v.pixel_stride = 5;
    uc_input_view alias = {out, 3, 5, 8};
    BAD(CALL(&alias, 1, out + 3, 6, 1, 1, 2), UC_OUTPUT_OVERLAP); /* stride gap */
    BAD(CALL(&alias, 1, out, 6, 1, 1, 2), UC_OUTPUT_OVERLAP);
    alias.data = out + 3;
    BAD(CALL(&alias, 1, out, 6, 1, 1, 2), UC_OUTPUT_OVERLAP);
    union {
        uc_input_view views[4];
        float words[64];
    } metadata;
    memset(&metadata, 0, sizeof(metadata));
    metadata.views[0] = v;
    unsigned char save[sizeof(metadata)];
    memcpy(save, &metadata, sizeof(save));
    CHECK(CALL(metadata.views, 1, metadata.words, 64, 1, 1, 2) == UC_OUTPUT_OVERLAP);
    CHECK(memcmp(save, &metadata, sizeof(save)) == 0);
    BAD(CALL(&v, 1, out, SIZE_MAX, SIZE_MAX, 2, 1), UC_OVERFLOW);
    BAD(CALL(&v, SIZE_MAX, out, 64, 1, 1, 2), UC_OVERFLOW);
    uc_input_view huge = {src, SIZE_MAX, SIZE_MAX, SIZE_MAX};
    BAD(CALL(&huge, 1, out, SIZE_MAX, 1, 1, 2), UC_OVERFLOW);
    huge.channels = 1;
    huge.pixel_stride = SIZE_MAX;
    BAD(CALL(&huge, 1, out, SIZE_MAX, 1, 1, 3), UC_OVERFLOW);
    huge.channels = (size_t)PTRDIFF_MAX / 4 + 1;
    huge.pixel_stride = huge.channels;
    BAD(CALL(&huge, 1, out, SIZE_MAX, 1, 1, 1), UC_OVERFLOW);
    huge = (uc_input_view){(const float *)(uintptr_t)(UINTPTR_MAX - 3), 2, 2, 2};
    BAD(CALL(&huge, 1, out, 64, 1, 1, 1), UC_OVERFLOW);
    BAD(CALL(&v, 1, (float *)(uintptr_t)(UINTPTR_MAX - 3), 64, 1, 1, 2), UC_OVERFLOW);
#undef CALL
#undef BAD
}
static void exact_allocations(void) {
    /* No guard padding: sanitizers catch even a single extra tail read. */
    for (size_t c = 1; c < 136; ++c) {
        size_t n = 3, span = 2 * (c + 3) + c;
        float *s = (float *)malloc(span * 4), *d = (float *)malloc(n * c * 4);
        float *ref = (float *)malloc(n * c * 4);
        CHECK(s && d && ref);
        fill(s, span);
        uc_input_view v = {s, c, c + 3, span};
        oracle(&v, 1, ref, n);
        {
            CHECK(universal_concat_nhwc_f32(&v, 1, d, n * c, 1, 1, n) == 0);
            CHECK(memcmp(d, ref, n * c * 4) == 0);
        }
        free(s);
        free(d);
        free(ref);
        ++cases;
    }
}
static void dense_fast_tails(void) {
    const size_t widths[] = {16, 32, 64, 128}, pixels[] = {1, 3, 5};
    for (size_t ci = 0; ci < 4; ++ci)
        for (size_t k = 2; k <= 3; ++k)
            for (size_t ni = 0; ni < 3; ++ni) {
                size_t c = widths[ci], n = pixels[ni], co = (k + 1) * c;
                uc_input_view v[3];
                float *sources[3];
                for (size_t i = 0; i < k; ++i) {
                    size_t channels = i ? c : 2 * c;
                    sources[i] = (float *)malloc(n * channels * 4);
                    CHECK(sources[i]);
                    fill(sources[i], n * channels);
                    v[i] = (uc_input_view){sources[i], channels, channels, n * channels};
                }
                float *out = (float *)malloc(n * co * 4), *ref = (float *)malloc(n * co * 4);
                CHECK(out && ref);
                oracle(v, k, ref, n);
                {
                    memset(out, 0x5d, n * co * 4);
                    CHECK(universal_concat_nhwc_f32(v, k, out, n * co, 1, 1, n) == UC_OK);
                    CHECK(memcmp(out, ref, n * co * 4) == 0);
                }
                for (size_t i = 0; i < k; ++i)
                    free(sources[i]);
                free(out);
                free(ref);
                ++cases;
            }
    /* Also cover the neck and four-way SPPF at short and odd pixel counts. */
    const size_t sites[] = {4, 7, 9};
    for (size_t si = 0; si < 3; ++si)
        for (size_t ni = 0; ni < 3; ++ni) {
            const uc_workload *a = &uc_workloads[sites[si]];
            size_t n = pixels[ni], co = 0;
            float *sources[4];
            uc_input_view v[4];
            for (size_t i = 0; i < a->count; ++i) {
                size_t c = a->channels[i];
                co += c;
                sources[i] = (float *)malloc(n * c * 4);
                CHECK(sources[i]);
                fill(sources[i], n * c);
                v[i] = (uc_input_view){sources[i], c, c, n * c};
            }
            float *out = (float *)malloc(n * co * 4), *ref = (float *)malloc(n * co * 4);
            CHECK(out && ref);
            oracle(v, a->count, ref, n);
            {
                memset(out, 0x5d, n * co * 4);
                CHECK(universal_concat_nhwc_f32(v, a->count, out, n * co, 1, 1, n) == UC_OK);
                CHECK(memcmp(out, ref, n * co * 4) == 0);
            }
            for (size_t i = 0; i < a->count; ++i)
                free(sources[i]);
            free(out);
            free(ref);
            ++cases;
        }
}
static void page_boundary_tails(void) {
#if defined(__unix__)
    /* Also check GCC's optimized machine code without relying on allocator
     * padding or a particular compiler's sanitizer runtime. */
    const size_t widths[] = {1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129};
    size_t page = (size_t)sysconf(_SC_PAGESIZE);
    for (size_t i = 0; i < sizeof(widths) / sizeof(widths[0]); ++i) {
        size_t c = widths[i], n = 3, stride = c + 3, span = (n - 1) * stride + c;
        CHECK(span * 4 < page && n * c * 4 < page);
        unsigned char *s = (unsigned char *)mmap(NULL, page * 2, PROT_READ | PROT_WRITE,
                                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        unsigned char *d = (unsigned char *)mmap(NULL, page * 2, PROT_READ | PROT_WRITE,
                                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        CHECK(s != MAP_FAILED && d != MAP_FAILED);
        CHECK(mprotect(s + page, page, PROT_NONE) == 0 && mprotect(d + page, page, PROT_NONE) == 0);
        float *src = (float *)(s + page - span * 4), *out = (float *)(d + page - n * c * 4);
        float *ref = (float *)malloc(n * c * 4);
        CHECK(ref);
        fill(src, span);
        uc_input_view v = {src, c, stride, span};
        oracle(&v, 1, ref, n);
        {
            CHECK(universal_concat_nhwc_f32(&v, 1, out, n * c, 1, 1, n) == UC_OK);
            CHECK(memcmp(out, ref, n * c * 4) == 0);
        }
        free(ref);
        CHECK(munmap(s, page * 2) == 0 && munmap(d, page * 2) == 0);
        ++cases;
    }
#endif
}
int main(void) {
    size_t small[] = {3, 1, 9, 7, 11, 2, 5, 13, 1};
    verify_case(2, 3, 5, 3, small, 4, 1);
    const size_t images[] = {640, 1024, 2048};
    for (size_t r = 0; r < 3; ++r)
        for (size_t j = 0; j < UC_SITE_COUNT; ++j) {
            const uc_workload *a = &uc_workloads[j];
            size_t side = images[r] / a->scale;
            verify_case(1, side, side, a->count, a->channels, 0, 0);
        }
    for (size_t k = 1; k <= 9; ++k)
        for (size_t offset = 0; offset < 9; ++offset)
            verify_case(2, 3, 7, k, small, offset % 4, offset);
    /* Production channel signatures also work at entirely different sizes. */
    for (size_t j = 0; j < UC_SITE_COUNT; ++j)
        verify_case(2, 11, 17, uc_workloads[j].count, uc_workloads[j].channels, 3, j % 9);
    views();
    reject();
    exact_allocations();
    dense_fast_tails();
    page_boundary_tails();
    CHECK(strcmp(uc_status_string(UC_OK), "ok") == 0);
    puts("PASS: all 33 production configurations, arbitrary sizes/strides, views, bit patterns, "
         "guards and rejection tests");
    printf("cases=%zu assertions=%zu\n", cases, checks);
    return 0;
}
