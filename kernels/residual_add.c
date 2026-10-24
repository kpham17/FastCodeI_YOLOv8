/* Standalone FP32 NHWC residual addition, one CPU thread, no scratch space.
 * Dense inputs use a linear stream; a chunk-B shortcut has pixel pitch 2*C.
 * The strided tiles process 128 outputs with fixed shortcut displacements.
 * Load/add/store ordering and shape dispatch are tuned for ece020 (GCC 8.5).
 * dst may equal conv; shortcut must remain disjoint from dst and read-only.
 */
#include "residual_add.h"
#include <immintrin.h>
#include <stdint.h>
#if !defined(__x86_64__) || !(defined(__GNUC__) || defined(__clang__))
#error "This optimized submission targets x86-64 GCC/Clang."
#endif
#define AVX2   __attribute__((target("avx2")))
#define AVX512 __attribute__((target("avx512f")))
#define ALWAYS_INLINE __attribute__((always_inline)) inline
/* Scalar reference/fallback. The compiler is asked not to vectorize this
 * routine so tests also compare against a truly scalar implementation. */
__attribute__((optimize("no-tree-vectorize")))
static void add_scalar(float *d, const float *a, const float *b,
                       size_t pixels, size_t C, size_t pitch) {
    for (size_t p = 0; p < pixels; ++p)
        for (size_t c = 0; c < C; ++c)
            d[p*C + c] = a[p*C + c] + b[p*pitch + c];
}
/* Four AVX2 operations per iteration; the dense case is one linear stream.
 * For other pitches the fallback still vectorizes within each pixel. */
AVX2 static void add_avx2(float *d, const float *a, const float *b,
                          size_t pixels, size_t C, size_t pitch) {
    if (pitch == C) {
        const size_t n = pixels*C;
        size_t i=0;
        for (; n-i >= 32; i+=32)
            for (size_t k=0; k<32; k+=8)
                _mm256_storeu_ps(d+i+k, _mm256_add_ps(
                    _mm256_loadu_ps(a+i+k), _mm256_loadu_ps(b+i+k)));
        for (; n-i >= 8; i+=8)
            _mm256_storeu_ps(d+i, _mm256_add_ps(
                _mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i)));
        for (; i<n; ++i) d[i]=a[i]+b[i];
    } else {
        for (size_t p=0; p<pixels; ++p) {
            const size_t o=p*C, s=p*pitch;
            size_t c=0;
            for (; C-c>=8; c+=8)
                _mm256_storeu_ps(d+o+c, _mm256_add_ps(
                    _mm256_loadu_ps(a+o+c), _mm256_loadu_ps(b+s+c)));
            for (; c<C; ++c) d[o+c]=a[o+c]+b[s+c];
        }
    }
}
/* Wide vectors are selected only for the small dense shapes measured faster. */
AVX512 static void add_avx512_dense(float *d, const float *a,
                                    const float *b, size_t n) {
    size_t i=0;
    for (; n-i>=64; i+=64)
        for (size_t k=0; k<64; k+=16)
            _mm512_storeu_ps(d+i+k, _mm512_add_ps(
                _mm512_loadu_ps(a+i+k), _mm512_loadu_ps(b+i+k)));
    for (; n-i>=16; i+=16)
        _mm512_storeu_ps(d+i, _mm512_add_ps(
            _mm512_loadu_ps(a+i), _mm512_loadu_ps(b+i)));
    for (; i<n; ++i) d[i]=a[i]+b[i];
}
/* Preserve the generic pixel loop: GCC 8 specialization was slower at 640 P4. */
AVX512 __attribute__((noipa))
static void add_avx512_by_pixel(float *d, const float *a, const float *b,
                                size_t pixels, size_t C, size_t pitch) {
    for (size_t p=0; p<pixels; ++p) {
        size_t i=p*C, s=p*pitch, c=0;
        for (; C-c>=16; c+=16)
            _mm512_storeu_ps(d+i+c, _mm512_add_ps(
                _mm512_loadu_ps(a+i+c), _mm512_loadu_ps(b+s+c)));
        for (; c<C; ++c) d[i+c]=a[i+c]+b[s+c];
    }
}
/* Register/register AVX2 variant for 128-output strided tiles.
 * Process each tile in two 64-float halves (8 + 8 YMM registers).
 * Every addressing displacement below is a literal BYTE offset.
 * No index multiply or shortcut-address calculation occurs inside this asm.
 * Loads of both inputs complete before any vaddps; stores follow the adds.
 */
#define A_LOAD_0 \
    "vmovups 0(%[a]), %%ymm0\n\t" \
    "vmovups 32(%[a]), %%ymm1\n\t" \
    "vmovups 64(%[a]), %%ymm2\n\t" \
    "vmovups 96(%[a]), %%ymm3\n\t" \
    "vmovups 128(%[a]), %%ymm4\n\t" \
    "vmovups 160(%[a]), %%ymm5\n\t" \
    "vmovups 192(%[a]), %%ymm6\n\t" \
    "vmovups 224(%[a]), %%ymm7\n\t"
#define B_LOAD_0_C16 \
    "vmovups 0(%[b]), %%ymm8\n\t" \
    "vmovups 32(%[b]), %%ymm9\n\t" \
    "vmovups 128(%[b]), %%ymm10\n\t" \
    "vmovups 160(%[b]), %%ymm11\n\t" \
    "vmovups 256(%[b]), %%ymm12\n\t" \
    "vmovups 288(%[b]), %%ymm13\n\t" \
    "vmovups 384(%[b]), %%ymm14\n\t" \
    "vmovups 416(%[b]), %%ymm15\n\t"
#define B_LOAD_0_C32 \
    "vmovups 0(%[b]), %%ymm8\n\t" \
    "vmovups 32(%[b]), %%ymm9\n\t" \
    "vmovups 64(%[b]), %%ymm10\n\t" \
    "vmovups 96(%[b]), %%ymm11\n\t" \
    "vmovups 256(%[b]), %%ymm12\n\t" \
    "vmovups 288(%[b]), %%ymm13\n\t" \
    "vmovups 320(%[b]), %%ymm14\n\t" \
    "vmovups 352(%[b]), %%ymm15\n\t"
#define B_LOAD_0_C64 \
    "vmovups 0(%[b]), %%ymm8\n\t" \
    "vmovups 32(%[b]), %%ymm9\n\t" \
    "vmovups 64(%[b]), %%ymm10\n\t" \
    "vmovups 96(%[b]), %%ymm11\n\t" \
    "vmovups 128(%[b]), %%ymm12\n\t" \
    "vmovups 160(%[b]), %%ymm13\n\t" \
    "vmovups 192(%[b]), %%ymm14\n\t" \
    "vmovups 224(%[b]), %%ymm15\n\t"
#define B_LOAD_0_C128 \
    "vmovups 0(%[b]), %%ymm8\n\t" \
    "vmovups 32(%[b]), %%ymm9\n\t" \
    "vmovups 64(%[b]), %%ymm10\n\t" \
    "vmovups 96(%[b]), %%ymm11\n\t" \
    "vmovups 128(%[b]), %%ymm12\n\t" \
    "vmovups 160(%[b]), %%ymm13\n\t" \
    "vmovups 192(%[b]), %%ymm14\n\t" \
    "vmovups 224(%[b]), %%ymm15\n\t"
#define STORE_0 \
    "vmovups %%ymm0, 0(%[d])\n\t" \
    "vmovups %%ymm1, 32(%[d])\n\t" \
    "vmovups %%ymm2, 64(%[d])\n\t" \
    "vmovups %%ymm3, 96(%[d])\n\t" \
    "vmovups %%ymm4, 128(%[d])\n\t" \
    "vmovups %%ymm5, 160(%[d])\n\t" \
    "vmovups %%ymm6, 192(%[d])\n\t" \
    "vmovups %%ymm7, 224(%[d])\n\t"
#define A_LOAD_1 \
    "vmovups 256(%[a]), %%ymm0\n\t" \
    "vmovups 288(%[a]), %%ymm1\n\t" \
    "vmovups 320(%[a]), %%ymm2\n\t" \
    "vmovups 352(%[a]), %%ymm3\n\t" \
    "vmovups 384(%[a]), %%ymm4\n\t" \
    "vmovups 416(%[a]), %%ymm5\n\t" \
    "vmovups 448(%[a]), %%ymm6\n\t" \
    "vmovups 480(%[a]), %%ymm7\n\t"
#define B_LOAD_1_C16 \
    "vmovups 512(%[b]), %%ymm8\n\t" \
    "vmovups 544(%[b]), %%ymm9\n\t" \
    "vmovups 640(%[b]), %%ymm10\n\t" \
    "vmovups 672(%[b]), %%ymm11\n\t" \
    "vmovups 768(%[b]), %%ymm12\n\t" \
    "vmovups 800(%[b]), %%ymm13\n\t" \
    "vmovups 896(%[b]), %%ymm14\n\t" \
    "vmovups 928(%[b]), %%ymm15\n\t"
#define B_LOAD_1_C32 \
    "vmovups 512(%[b]), %%ymm8\n\t" \
    "vmovups 544(%[b]), %%ymm9\n\t" \
    "vmovups 576(%[b]), %%ymm10\n\t" \
    "vmovups 608(%[b]), %%ymm11\n\t" \
    "vmovups 768(%[b]), %%ymm12\n\t" \
    "vmovups 800(%[b]), %%ymm13\n\t" \
    "vmovups 832(%[b]), %%ymm14\n\t" \
    "vmovups 864(%[b]), %%ymm15\n\t"
#define B_LOAD_1_C64 \
    "vmovups 512(%[b]), %%ymm8\n\t" \
    "vmovups 544(%[b]), %%ymm9\n\t" \
    "vmovups 576(%[b]), %%ymm10\n\t" \
    "vmovups 608(%[b]), %%ymm11\n\t" \
    "vmovups 640(%[b]), %%ymm12\n\t" \
    "vmovups 672(%[b]), %%ymm13\n\t" \
    "vmovups 704(%[b]), %%ymm14\n\t" \
    "vmovups 736(%[b]), %%ymm15\n\t"
#define B_LOAD_1_C128 \
    "vmovups 256(%[b]), %%ymm8\n\t" \
    "vmovups 288(%[b]), %%ymm9\n\t" \
    "vmovups 320(%[b]), %%ymm10\n\t" \
    "vmovups 352(%[b]), %%ymm11\n\t" \
    "vmovups 384(%[b]), %%ymm12\n\t" \
    "vmovups 416(%[b]), %%ymm13\n\t" \
    "vmovups 448(%[b]), %%ymm14\n\t" \
    "vmovups 480(%[b]), %%ymm15\n\t"
#define STORE_1 \
    "vmovups %%ymm0, 256(%[d])\n\t" \
    "vmovups %%ymm1, 288(%[d])\n\t" \
    "vmovups %%ymm2, 320(%[d])\n\t" \
    "vmovups %%ymm3, 352(%[d])\n\t" \
    "vmovups %%ymm4, 384(%[d])\n\t" \
    "vmovups %%ymm5, 416(%[d])\n\t" \
    "vmovups %%ymm6, 448(%[d])\n\t" \
    "vmovups %%ymm7, 480(%[d])\n\t"
#define ADD_REG_REG \
    "vaddps %%ymm8, %%ymm0, %%ymm0\n\t" \
    "vaddps %%ymm9, %%ymm1, %%ymm1\n\t" \
    "vaddps %%ymm10, %%ymm2, %%ymm2\n\t" \
    "vaddps %%ymm11, %%ymm3, %%ymm3\n\t" \
    "vaddps %%ymm12, %%ymm4, %%ymm4\n\t" \
    "vaddps %%ymm13, %%ymm5, %%ymm5\n\t" \
    "vaddps %%ymm14, %%ymm6, %%ymm6\n\t" \
    "vaddps %%ymm15, %%ymm7, %%ymm7\n\t"
#define DEFINE_TILE(C) \
AVX2 static ALWAYS_INLINE void block_ymm_##C(float*d,const float*a,const float*b) { \
    __asm__ __volatile__( \
        A_LOAD_0 B_LOAD_0_C##C ADD_REG_REG STORE_0 \
        A_LOAD_1 B_LOAD_1_C##C ADD_REG_REG STORE_1 \
        : : [a] "r"(a), [b] "r"(b), [d] "r"(d) \
        : "memory", "ymm0", "ymm1", "ymm2", "ymm3", "ymm4", "ymm5", \
          "ymm6", "ymm7", "ymm8", "ymm9", "ymm10", "ymm11", \
          "ymm12", "ymm13", "ymm14", "ymm15"); \
}
DEFINE_TILE(16)
DEFINE_TILE(32)
DEFINE_TILE(64)
DEFINE_TILE(128)
#undef DEFINE_TILE
#undef A_LOAD_0
#undef A_LOAD_1
#undef STORE_0
#undef STORE_1
#undef ADD_REG_REG
#undef B_LOAD_0_C16
#undef B_LOAD_0_C32
#undef B_LOAD_0_C64
#undef B_LOAD_0_C128
#undef B_LOAD_1_C16
#undef B_LOAD_1_C32
#undef B_LOAD_1_C64
#undef B_LOAD_1_C128
/* One tile = 128 FP32 outputs, regardless of C.  Tile pointers advance
 * by fixed strides (128, 128, 256 floats).  Both hot loops count DOWN to
 * zero; the prefetch loop runs only while all prefetch addresses are valid.
 * Keep the original register/register 8+8-YMM arithmetic unchanged.
 */
#define STRIDED_CASE(CH, STEP, AHEAD, COUNT, PREFETCH) \
    case CH: { \
        const size_t tiles = pixels / (STEP); \
        size_t pf_tiles = 0; \
        if (PREFETCH) { \
            const size_t safe = (pixels >= (AHEAD)+(COUNT)) \
                ? pixels - ((AHEAD)+(COUNT)-1) : 0; \
            pf_tiles = (safe + (STEP)-1) / (STEP); \
            for (size_t left=pf_tiles; left!=0; --left) { \
                for (size_t k=AHEAD; k<(AHEAD)+(COUNT); ++k) \
                    _mm_prefetch((const char*)(b+k*(2*CH)),_MM_HINT_T0); \
                block_ymm_##CH(d,a,b); \
                d+=128; a+=128; b+=256; \
            } \
        } \
        for (size_t left=tiles-pf_tiles; left!=0; --left) { \
            block_ymm_##CH(d,a,b); \
            d+=128; a+=128; b+=256; \
        } \
        if (pixels % (STEP)) \
            add_avx2(d,a,b,pixels % (STEP),CH,2*CH); \
        break; \
    }
#define STRIDED_2C_IMPL(NAME,PREFETCH) \
AVX2 static void NAME(float*d,const float*a,const float*b,size_t pixels,size_t C) { \
    switch(C) { \
        STRIDED_CASE(16,8,32,4,PREFETCH) \
        STRIDED_CASE(32,4,16,4,PREFETCH) \
        STRIDED_CASE(64,2,8,2,PREFETCH) \
        STRIDED_CASE(128,1,4,1,PREFETCH) \
    } \
}
STRIDED_2C_IMPL(strided_disp,0)
STRIDED_2C_IMPL(strided_pf4k,1)
#undef STRIDED_2C_IMPL
#undef STRIDED_CASE
/* 2048 P5 view favors an interleaved AVX2 loop. The prefetch remains inside
 * a future shortcut pixel; all arithmetic loads/stores allow misalignment. */
AVX2 static void p5_interleaved(float*d,const float*a,const float*b,size_t P) {
    for(size_t p=0;p<P;++p) {
        if(P-p>5)_mm_prefetch((const char*)(b+(p+4)*256),_MM_HINT_T0);
        _Pragma("GCC unroll 16")
        for(size_t c=0;c<128;c+=8)
            _mm256_storeu_ps(d+p*128+c,_mm256_add_ps(
                _mm256_loadu_ps(a+p*128+c),_mm256_loadu_ps(b+p*256+c)));
    }
}
/* Two cache-sized 2048 dense tensors favor the generic pixel loop over the
 * flattened unroll with aligned in-place buffers. Other buffers keep the old
 * path. Prevent IPA specialization, matching the measured control. */
AVX2 __attribute__((noipa))
static void add_avx2_by_pixel(float*d,const float*a,const float*b,
                              size_t P,size_t C,size_t S) {
    for(size_t p=0;p<P;++p) {
        size_t c=0;
        for(;C-c>=8;c+=8)
            _mm256_storeu_ps(d+c,_mm256_add_ps(
                _mm256_loadu_ps(a+c),_mm256_loadu_ps(b+c)));
        for(;c<C;++c)d[c]=a[c]+b[c];
        d+=C;a+=C;b+=S;
    }
}
/* Runtime dispatch is deliberately short; shapes outside the model remain
 * correct through the scalar/AVX2 fallback. No copies or scratch buffers. */
void residual_add_contiguous(float *dst, const float *conv,
                             const float *shortcut, size_t pixels, size_t C) {
    if (!pixels || !C) return;
    const int small_avx512 = (C==64 && pixels==1600) ||
        (C==128 && (pixels==400 || pixels==1024));
    if (small_avx512 && __builtin_cpu_supports("avx512f")) {
        /* Preserve the measured generic loop for 640 P4. */
        if (C==64 && pixels==1600)
            add_avx512_by_pixel(dst,conv,shortcut,pixels,C,C);
        else
            add_avx512_dense(dst,conv,shortcut,pixels*C);
    } else if (__builtin_cpu_supports("avx2")) {
        if (((C==32 && pixels==65536) || (C==64 && pixels==16384)) &&
            dst==conv && (((uintptr_t)dst | (uintptr_t)shortcut) & 63u)==0)
            add_avx2_by_pixel(dst,conv,shortcut,pixels,C,C);
        else add_avx2(dst,conv,shortcut,pixels,C,C);
    } else
        add_scalar(dst,conv,shortcut,pixels,C,C);
}
void residual_add_strided(float *dst, const float *conv,
                          const float *shortcut, size_t pixels,
                          size_t C, size_t pitch) {
    if (!pixels || !C) return;
    if (!__builtin_cpu_supports("avx2")) {
        add_scalar(dst,conv,shortcut,pixels,C,pitch); return;
    }
    if (pitch == 2*C && (C==16 || C==32 || C==64 || C==128)) {
        /* Select the measured schedule without materializing chunk B. */
        if (C==128 && pixels==4096) {
            p5_interleaved(dst,conv,shortcut,pixels);
        } else if ((C==16 && (pixels==25600 || pixels==65536)) ||
                   (C==32 && pixels==65536) ||
                   (C==64 && (pixels==1600 || pixels==4096)) ||
                   (C==128 && (pixels==400 || pixels==1024))) {
            strided_pf4k(dst,conv,shortcut,pixels,C);
        } else {
            strided_disp(dst,conv,shortcut,pixels,C);
        }
    } else {
        /* General zero-copy view, including non-YOLO pitch and tails. */
        add_avx2(dst,conv,shortcut,pixels,C,pitch);
    }
}
