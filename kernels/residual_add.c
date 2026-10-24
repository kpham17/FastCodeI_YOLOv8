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

/* Vector R starts at dense byte offset R*32. Every C shortcut floats are
 * followed by C skipped floats, giving R*32 + floor(R*8/C)*C*4 bytes.
 * GAS evaluates the expression at assembly time. Keep loads, then adds,
 * then stores: the simpler interleaved schedule lost on these strided cases. */
#define REGS(OP,C) \
    OP(0,C) OP(1,C) OP(2,C) OP(3,C) OP(4,C) OP(5,C) OP(6,C) OP(7,C) \
    OP(8,C) OP(9,C) OP(10,C) OP(11,C) OP(12,C) OP(13,C) OP(14,C) OP(15,C)
#define LOAD(R,C) "vmovups " #R "*32(%[a]), %%ymm" #R "\n\t"
#define ADD(R,C) "vaddps " #R "*32+(" #R "*8/" #C ")*" #C \
    "*4(%[b]), %%ymm" #R ", %%ymm" #R "\n\t"
#define STORE(R,C) "vmovups %%ymm" #R ", " #R "*32(%[d])\n\t"
#define DEFINE_TILE(C) \
AVX2 static ALWAYS_INLINE void block_ymm_##C(float*d,const float*a,const float*b) { \
    __asm__ __volatile__(REGS(LOAD,C) REGS(ADD,C) REGS(STORE,C) \
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
#undef REGS
#undef LOAD
#undef ADD
#undef STORE

/* One shared case describes each 128-output tile. The shortcut prefetch
 * starts 4 KiB ahead; COUNT bounds every prefetched pixel inside the view.
 * PREFETCH is constant, so the plain variant contains no prefetch branch. */
#define STRIDED_CASE(CH, STEP, AHEAD, COUNT, PREFETCH) \
    case CH: { \
        size_t p=0; \
        for (; pixels-p>=STEP; p+=STEP) { \
            if ((PREFETCH) && pixels-p>AHEAD+COUNT-1) \
                for (size_t k=AHEAD; k<AHEAD+COUNT; ++k) \
                    _mm_prefetch((const char*)(b+(p+k)*(2*CH)),_MM_HINT_T0); \
            block_ymm_##CH(d+p*CH,a+p*CH,b+p*(2*CH)); \
        } \
        if (p<pixels) add_avx2(d+p*CH,a+p*CH,b+p*(2*CH),pixels-p,CH,2*CH); \
        break; \
    }
#define STRIDED_2C_IMPL(NAME, PREFETCH) \
AVX2 static void NAME(float*d,const float*a,const float*b,size_t pixels,size_t C) { \
    switch (C) { \
        STRIDED_CASE(16,8,32,4,PREFETCH) \
        STRIDED_CASE(32,4,16,4,PREFETCH) \
        case 64: { \
            size_t p=0; \
            for (;pixels-p>=2;p+=2) { \
                if ((PREFETCH) && pixels-p>9) { \
                    _mm_prefetch((const char*)(b+(p+8)*128),_MM_HINT_T0); \
                    _mm_prefetch((const char*)(b+(p+9)*128),_MM_HINT_T0); \
                } \
                block_ymm_64(d+p*64,a+p*64,b+p*128); \
            } \
            if (p<pixels) add_avx2(d+p*64,a+p*64,b+p*128,pixels-p,64,128); \
            break; \
        } \
        case 128: \
            for (size_t p=0;p<pixels;++p) { \
                if ((PREFETCH) && pixels-p>4) \
                    _mm_prefetch((const char*)(b+(p+4)*256),_MM_HINT_T0); \
                block_ymm_128(d+p*128,a+p*128,b+p*256); \
            } \
            break; \
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
    __builtin_cpu_init();
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
    __builtin_cpu_init();
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
