/* Dense NHWC channel concatenation. All validation precedes the first store.
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
/* Constant byte counts give the compiler/libc an optimized memcpy baseline. */
static inline void copy_span(float *d, const float *s, size_t c) {
    switch (c) {
    case 16:
        memcpy(d, s, 64);
        break;
    case 32:
        memcpy(d, s, 128);
        break;
    case 64:
        memcpy(d, s, 256);
        break;
    case 128:
        memcpy(d, s, 512);
        break;
    case 256:
        memcpy(d, s, 1024);
        break;
    default:
        memcpy(d, s, c * sizeof(float));
    }
}
static int all_dense(const uc_input_view *v, size_t k) {
    for (size_t i = 0; i < k; ++i)
        if (v[i].pixel_stride != v[i].channels)
            return 0;
    return 1;
}
#define MEMCPY2(A, B)                                                                              \
    static void memcpy_##A##_##B(const uc_input_view *v, float *d, size_t n) {                     \
        const float *a = v[0].data, *b = v[1].data;                                                \
        for (size_t p = 0; p < n; ++p) {                                                           \
            memcpy(d + p * ((A) + (B)), a + p * (A), (A) * 4);                                     \
            memcpy(d + p * ((A) + (B)) + (A), b + p * (B), (B) * 4);                               \
        }                                                                                          \
    }
MEMCPY2(32, 16)
MEMCPY2(64, 32) MEMCPY2(128, 64) MEMCPY2(256, 128) MEMCPY2(64, 128) MEMCPY2(128, 256)
#undef MEMCPY2
    static int fixed_memcpy(const uc_input_view *v, size_t k, float *d, size_t n) {
    if (k < 2 || k > 4 || !all_dense(v, k))
        return 0;
    if (k == 2) {
#define MATCH(A, B)                                                                                \
    if (v[0].channels == (A) && v[1].channels == (B)) {                                            \
        memcpy_##A##_##B(v, d, n);                                                                 \
        return 1;                                                                                  \
    }
        MATCH(32, 16) MATCH(64, 32) MATCH(128, 64) MATCH(256, 128) MATCH(64, 128) MATCH(128, 256)
#undef MATCH
    }
    return 0;
}
static void concat_memcpy(const uc_input_view *v, size_t k, float *d, size_t n, size_t co) {
    if (fixed_memcpy(v, k, d, n))
        return;
    for (size_t p = 0; p < n; ++p) {
        float *pixel = d + p * co;
        for (size_t i = 0; i < k; ++i) {
            copy_span(pixel, v[i].data + p * v[i].pixel_stride, v[i].channels);
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
    for (; c - j >= 32; j += 32) {
        __m256i x0 = _mm256_loadu_si256((const __m256i_u *)(s + j));
        __m256i x1 = _mm256_loadu_si256((const __m256i_u *)(s + j + 8));
        __m256i x2 = _mm256_loadu_si256((const __m256i_u *)(s + j + 16));
        __m256i x3 = _mm256_loadu_si256((const __m256i_u *)(s + j + 24));
        _mm256_storeu_si256((__m256i_u *)(d + j), x0);
        _mm256_storeu_si256((__m256i_u *)(d + j + 8), x1);
        _mm256_storeu_si256((__m256i_u *)(d + j + 16), x2);
        _mm256_storeu_si256((__m256i_u *)(d + j + 24), x3);
    }
    for (; c - j >= 8; j += 8) {
        __m256i x = _mm256_loadu_si256((const __m256i_u *)(s + j));
        _mm256_storeu_si256((__m256i_u *)(d + j), x);
    }
    if (j < c)
        memcpy(d + j, s + j, (c - j) * sizeof(float));
}
/* Register-tiled 16/32 channel output: cv1 supplies the first 2c
 * channels contiguously; Bottleneck output(s) supply the remaining c each.
 * Load all cv1 registers before storing, minimizing memory interleaving. */
static AVX2 void concat_c16(const float *s, const float *a, const float *b, size_t m, float *d,
                            size_t n) {
    const size_t stride = (m + 2) * 16;
    for (size_t p = 0; p < n; p++, s += 32, a += 16, d += stride) {
        __m256 x0 = _mm256_loadu_ps(s + 0), x1 = _mm256_loadu_ps(s + 8);
        __m256 x2 = _mm256_loadu_ps(s + 16), x3 = _mm256_loadu_ps(s + 24);
        _mm256_storeu_ps(d + 0, x0);
        _mm256_storeu_ps(d + 8, x1);
        _mm256_storeu_ps(d + 16, x2);
        _mm256_storeu_ps(d + 24, x3);
        __m256 y0 = _mm256_loadu_ps(a + 0), y1 = _mm256_loadu_ps(a + 8);
        _mm256_storeu_ps(d + 32, y0);
        _mm256_storeu_ps(d + 40, y1);
        if (m == 2) {
            const float *bb = b + p * 16;
            __m256 z0 = _mm256_loadu_ps(bb + 0), z1 = _mm256_loadu_ps(bb + 8);
            _mm256_storeu_ps(d + 48, z0);
            _mm256_storeu_ps(d + 56, z1);
        }
    }
}
static AVX2 void concat_c32(const float *s, const float *a, const float *b, size_t m, float *d,
                            size_t n) {
    const size_t stride = (m + 2) * 32;
    for (size_t p = 0; p < n; p++, s += 64, a += 32, d += stride) {
        __m256 x0 = _mm256_loadu_ps(s + 0), x1 = _mm256_loadu_ps(s + 8);
        __m256 x2 = _mm256_loadu_ps(s + 16), x3 = _mm256_loadu_ps(s + 24);
        __m256 x4 = _mm256_loadu_ps(s + 32), x5 = _mm256_loadu_ps(s + 40);
        __m256 x6 = _mm256_loadu_ps(s + 48), x7 = _mm256_loadu_ps(s + 56);
        _mm256_storeu_ps(d + 0, x0);
        _mm256_storeu_ps(d + 8, x1);
        _mm256_storeu_ps(d + 16, x2);
        _mm256_storeu_ps(d + 24, x3);
        _mm256_storeu_ps(d + 32, x4);
        _mm256_storeu_ps(d + 40, x5);
        _mm256_storeu_ps(d + 48, x6);
        _mm256_storeu_ps(d + 56, x7);
        __m256 y0 = _mm256_loadu_ps(a + 0), y1 = _mm256_loadu_ps(a + 8);
        __m256 y2 = _mm256_loadu_ps(a + 16), y3 = _mm256_loadu_ps(a + 24);
        _mm256_storeu_ps(d + 64, y0);
        _mm256_storeu_ps(d + 72, y1);
        _mm256_storeu_ps(d + 80, y2);
        _mm256_storeu_ps(d + 88, y3);
        if (m == 2) {
            const float *bb = b + p * 32;
            __m256 z0 = _mm256_loadu_ps(bb + 0), z1 = _mm256_loadu_ps(bb + 8);
            __m256 z2 = _mm256_loadu_ps(bb + 16), z3 = _mm256_loadu_ps(bb + 24);
            _mm256_storeu_ps(d + 96, z0);
            _mm256_storeu_ps(d + 104, z1);
            _mm256_storeu_ps(d + 112, z2);
            _mm256_storeu_ps(d + 120, z3);
        }
    }
}

/* Large C2f joins: 4-pixel unroll, each c-float copy uses AVX2.
 * Keep this structure to preserve the measured larger-channel behavior. */
#define COPY3(C)                                                                                   \
    do {                                                                                           \
        copy_avx2(d, s, 2 * (C));                                                                  \
        copy_avx2(d + 2 * (C), a, (C));                                                            \
        s += 2 * (C);                                                                              \
        a += (C);                                                                                  \
        d += 3 * (C);                                                                              \
    } while (0)
#define COPY4(C)                                                                                   \
    do {                                                                                           \
        copy_avx2(d, s, 2 * (C));                                                                  \
        copy_avx2(d + 2 * (C), a, (C));                                                            \
        copy_avx2(d + 3 * (C), b, (C));                                                            \
        s += 2 * (C);                                                                              \
        a += (C);                                                                                  \
        b += (C);                                                                                  \
        d += 4 * (C);                                                                              \
    } while (0)
#define LARGE_CONCAT(C)                                                                            \
    static AVX2 void concat_##C(const float *s, const float *const *bn, size_t m, float *d,        \
                                size_t n) {                                                        \
        size_t p = 0;                                                                              \
        const float *a = bn[0];                                                                    \
        if (m == 1) {                                                                              \
            for (; p + 4 <= n; p += 4) {                                                           \
                COPY3(C);                                                                          \
                COPY3(C);                                                                          \
                COPY3(C);                                                                          \
                COPY3(C);                                                                          \
            }                                                                                      \
            for (; p < n; ++p) {                                                                   \
                COPY3(C);                                                                          \
            }                                                                                      \
        } else {                                                                                   \
            const float *b = bn[1];                                                                \
            for (; p + 4 <= n; p += 4) {                                                           \
                COPY4(C);                                                                          \
                COPY4(C);                                                                          \
                COPY4(C);                                                                          \
                COPY4(C);                                                                          \
            }                                                                                      \
            for (; p < n; ++p) {                                                                   \
                COPY4(C);                                                                          \
            }                                                                                      \
        }                                                                                          \
    }
LARGE_CONCAT(64)
LARGE_CONCAT(128)
#undef LARGE_CONCAT
#undef COPY3
#undef COPY4

#define DEFINE_NECK(CA, CB)                                                                        \
    AVX2 static void concat_##CA##_##CB##_avx2(const float *restrict a, const float *restrict b,   \
                                               float *restrict out, size_t npixels) {              \
        for (size_t p = 0; p < npixels; ++p) {                                                     \
            for (size_t c = 0; c < (CA); c += 8) {                                                 \
                __m256i v = _mm256_loadu_si256((const __m256i_u *)(a + c));                        \
                _mm256_storeu_si256((__m256i_u *)(out + c), v);                                    \
            }                                                                                      \
            for (size_t c = 0; c < (CB); c += 8) {                                                 \
                __m256i v = _mm256_loadu_si256((const __m256i_u *)(b + c));                        \
                _mm256_storeu_si256((__m256i_u *)(out + (CA) + c), v);                             \
            }                                                                                      \
            a += (CA);                                                                             \
            b += (CB);                                                                             \
            out += (CA) + (CB);                                                                    \
        }                                                                                          \
    }
    DEFINE_NECK(64, 128) DEFINE_NECK(128, 256)
#undef DEFINE_NECK

    /* Only these two sequential layouts are selected by the production policy.
     * Constant segment widths let the compiler inline every vector move. */
    static AVX2 void concat_c64_pair(const uc_input_view *v, float *d, size_t n) {
    const float *a = v[0].data, *b = v[1].data;
    for (size_t p = 0; p < n; ++p) {
        copy_avx2(d, a, 128);
        copy_avx2(d + 128, b, 64);
        a += 128;
        b += 64;
        d += 192;
    }
}
static AVX2 void concat_c64_triple(const uc_input_view *v, float *d, size_t n) {
    const float *a = v[0].data, *b = v[1].data, *c = v[2].data;
    for (size_t p = 0; p < n; ++p) {
        copy_avx2(d, a, 128);
        copy_avx2(d + 128, b, 64);
        copy_avx2(d + 192, c, 64);
        a += 128;
        b += 64;
        c += 64;
        d += 256;
    }
}
static AVX2 int fixed_vector(const uc_input_view *v, size_t k, float *d, size_t n) {
    if (k < 2 || k > 3 || !all_dense(v, k))
        return 0;
    size_t c = v[1].channels;
    if (k == 2 && v[0].channels == 64 && c == 128) {
        concat_64_128_avx2(v[0].data, v[1].data, d, n);
        return 1;
    }
    if (k == 2 && v[0].channels == 128 && c == 256) {
        concat_128_256_avx2(v[0].data, v[1].data, d, n);
        return 1;
    }
    if (c > SIZE_MAX / 2 || v[0].channels != 2 * c || (k == 3 && v[2].channels != c))
        return 0;
    const float *bn[] = {v[1].data, k == 3 ? v[2].data : NULL};
    switch (c) {
    case 16:
        concat_c16(v[0].data, bn[0], bn[1], k - 1, d, n);
        break;
    case 32:
        concat_c32(v[0].data, bn[0], bn[1], k - 1, d, n);
        break;
    case 64:
        concat_64(v[0].data, bn, k - 1, d, n);
        break;
    case 128:
        concat_128(v[0].data, bn, k - 1, d, n);
        break;
    default:
        return 0;
    }
    return 1;
}
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

int universal_concat_nhwc_f32(const uc_input_view *v, size_t k, float *out, size_t cap, size_t b,
                              size_t h, size_t w) {
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
        /* Measured on Xeon Silver4208: these large C2f layouts favor a
         * sequential output cursor. The feature bit only qualifies this CPU
         * policy; no AVX-512 instructions are compiled or executed here.
         * Total pixel count keeps the policy independent of B/H/W shape. */
        if (n >= 16384 && (k == 2 || k == 3) && v[0].channels == 128 && v[1].channels == 64 &&
            (k == 2 || v[2].channels == 64) && all_dense(v, k) &&
            __builtin_cpu_supports("avx512f")) {
            if (k == 2)
                concat_c64_pair(v, out, n);
            else
                concat_c64_triple(v, out, n);
            return UC_OK;
        }
        /* Dense common tuples retain their measured C2f/neck scheduling.
         * Everything else uses the same safe per-input-stride copy loop. */
        if (!fixed_vector(v, k, out, n))
            concat_vector(v, k, out, n, co);
        return UC_OK;
    }
#endif
    concat_memcpy(v, k, out, n, co);
    return UC_OK;
}
