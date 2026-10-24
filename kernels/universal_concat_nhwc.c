/* FP32 channel concatenation: strided NHWC inputs, dense NHWC output.
 * All validation precedes the first store.
 * The only SIMD implementation is cached AVX2; other CPUs use memcpy.
 */
#include "universal_concat_nhwc.h"
#include <stdint.h>
#include <string.h>

_Static_assert(sizeof(float) == 4, "FP32 storage requires four-byte float");

const char *uc_status_string(int s) {
    switch (s) {
    case UC_OK:
        return "ok";
    case UC_BAD_ARGUMENT:
        return "invalid argument";
    case UC_INSUFFICIENT_CAPACITY:
        return "insufficient capacity";
    case UC_OUTPUT_OVERLAP:
        return "output overlaps input or descriptors";
    case UC_OVERFLOW:
        return "size or address overflow";
    default:
        return "unknown status";
    }
}
/* Checked sizes also bound later pointer offsets and multiplications. */
static int multiply(size_t a, size_t b, size_t *r) {
    if (a && b > SIZE_MAX / a)
        return 0;
    *r = a * b;
    return 1;
}
static int span_bytes(size_t floats, size_t *bytes) {
    return multiply(floats, sizeof(float), bytes) && *bytes <= (size_t)PTRDIFF_MAX;
}
static int address_ok(const void *p, size_t bytes) {
    return bytes <= UINTPTR_MAX && (uintptr_t)p <= UINTPTR_MAX - bytes;
}
/* Integer differences avoid pointer ordering across unrelated allocations. */
static int overlap(const void *a, size_t an, const void *b, size_t bn) {
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? y - x < an : x - y < bn;
}
static int validate(const uc_input_view *v, size_t k, float *out, size_t cap, size_t b, size_t h,
                    size_t w, size_t *n, size_t *co) {
    size_t t, meta_bytes, out_floats, out_bytes;
    if (!v || !out || !k || !b || !h || !w)
        return UC_BAD_ARGUMENT;
    if (!multiply(k, sizeof(*v), &meta_bytes) || meta_bytes > (size_t)PTRDIFF_MAX ||
        !address_ok(v, meta_bytes) || !multiply(b, h, &t) || !multiply(t, w, n))
        return UC_OVERFLOW;
    *co = 0;
    for (size_t i = 0; i < k; ++i) {
        size_t tail, span, bytes;
        if (!v[i].data || !v[i].channels || v[i].pixel_stride < v[i].channels)
            return UC_BAD_ARGUMENT;
        if (*co > SIZE_MAX - v[i].channels || !multiply(*n - 1, v[i].pixel_stride, &tail) ||
            tail > SIZE_MAX - v[i].channels)
            return UC_OVERFLOW;
        *co += v[i].channels;
        span = tail + v[i].channels;
        if (!span_bytes(span, &bytes) || !address_ok(v[i].data, bytes))
            return UC_OVERFLOW;
        if (v[i].capacity < span)
            return UC_INSUFFICIENT_CAPACITY;
    }
    if (!multiply(*n, *co, &out_floats) || !span_bytes(out_floats, &out_bytes) ||
        !address_ok(out, out_bytes))
        return UC_OVERFLOW;
    if (cap < out_floats)
        return UC_INSUFFICIENT_CAPACITY;
    if (overlap(out, out_bytes, v, meta_bytes))
        return UC_OUTPUT_OVERLAP;
    for (size_t i = 0; i < k; ++i) {
        size_t bytes = ((*n - 1) * v[i].pixel_stride + v[i].channels) * sizeof(float);
        if (overlap(out, out_bytes, v[i].data, bytes))
            return UC_OUTPUT_OVERLAP;
    }
    return UC_OK;
}
/* Portable fallback uses the same per-input strides as the vector loop. */
static void concat_memcpy(const uc_input_view *v, size_t k, float *d, size_t n, size_t co) {
    for (size_t p = 0; p < n; ++p) {
        float *pixel = d + p * co;
        for (size_t i = 0; i < k; ++i) {
            memcpy(pixel, v[i].data + p * v[i].pixel_stride, v[i].channels * sizeof(float));
            pixel += v[i].channels;
        }
    }
}
#if !defined(UC_DISABLE_AVX2) && defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#define UC_VECTOR 1
#define AVX2 __attribute__((target("avx2")))
#define INLINE_AVX2 AVX2 __attribute__((always_inline)) inline

/* GCC/Clang's feature test includes the OS support needed to use AVX state. */
static int has_avx2(void) {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") != 0;
}

/* Copy whole vectors, then exactly the remaining bytes. Integer moves preserve
 * all FP32 bit patterns, including signaling NaNs, without FP operations. */
static INLINE_AVX2 void copy_avx2(float *d, const float *s, size_t c) {
    size_t j = 0;
    for (; c - j >= 8; j += 8) {
        __m256i x = _mm256_loadu_si256((const __m256i_u *)(s + j));
        _mm256_storeu_si256((__m256i_u *)(d + j), x);
    }
    if (j < c)
        memcpy(d + j, s + j, (c - j) * sizeof(float));
}
/* Shared fixed-width schedule for the measured dense C2f/neck cases.
 * Constant widths and input counts let GCC inline and unroll the vector moves. */
static INLINE_AVX2 void concat_dense(const uc_input_view *v, size_t k, float *d, size_t n,
                                     size_t ca, size_t cb) {
    const float *a = v[0].data, *b = v[1].data;
    const float *c = k == 3 ? v[2].data : NULL;
    for (size_t p = 0; p < n; ++p) {
        copy_avx2(d, a, ca);
        copy_avx2(d + ca, b, cb);
        if (k == 3) {
            copy_avx2(d + ca + cb, c, cb);
            c += cb;
        }
        a += ca;
        b += cb;
        d += ca + (k - 1) * cb;
    }
}
/* Return 1 after copying an exact supported tuple, otherwise use the generic loop. */
static AVX2 int fixed_vector(const uc_input_view *v, size_t k, float *d, size_t n) {
    if (k < 2 || k > 3)
        return 0;
    for (size_t i = 0; i < k; ++i)
        if (v[i].channels != v[i].pixel_stride)
            return 0;
    size_t c = v[1].channels;
    if (k == 2 && v[0].channels == 64 && c == 128) {
        concat_dense(v, 2, d, n, 64, 128);
        return 1;
    }
    if (c > SIZE_MAX / 2 || v[0].channels != 2 * c || (k == 3 && v[2].channels != c))
        return 0;
    /* C=16 fixed copying won for the measured small tensors on ece020.
     * Larger tensors use the general loop; this cutoff is target tuning. */
    if (c == 16 && n >= 32768)
        return 0;
/* The first C2f input contains both adjacent cv1 splits: 2*C channels. */
#define C2F_WIDTH(C) case C: \
    if (k == 2) concat_dense(v, 2, d, n, 2 * (C), C); \
    else concat_dense(v, 3, d, n, 2 * (C), C); \
    return 1
    switch (c) {
        C2F_WIDTH(16);
        C2F_WIDTH(32);
        C2F_WIDTH(64);
    default:
        return 0;
    }
#undef C2F_WIDTH
}
/* General path: arbitrary input count, channel widths and pixel strides. */
static AVX2 void concat_vector(const uc_input_view *v, size_t k, float *d, size_t n, size_t co) {
    for (size_t p = 0; p < n; ++p) {
        float *pixel = d + p * co;
        for (size_t i = 0; i < k; ++i) {
            copy_avx2(pixel, v[i].data + p * v[i].pixel_stride, v[i].channels);
            pixel += v[i].channels;
        }
    }
}
#else
#define UC_VECTOR 0
#endif

int universal_concat_nhwc_f32(const uc_input_view *v, size_t k, float *out, size_t cap,
                              size_t b, size_t h, size_t w) {
    size_t n, co;
    int status = validate(v, k, out, cap, b, h, w, &n, &co);
    if (status != UC_OK)
        return status;
    /* A single dense input is already one contiguous span. */
    if (k == 1 && v[0].pixel_stride == co) {
        memcpy(out, v[0].data, n * co * sizeof(float));
        return UC_OK;
    }
#if UC_VECTOR
    if (has_avx2()) {
        if (!fixed_vector(v, k, out, n))
            concat_vector(v, k, out, n, co);
        return UC_OK;
    }
#endif
    concat_memcpy(v, k, out, n, co);
    return UC_OK;
}
