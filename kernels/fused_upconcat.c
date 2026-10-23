#include "fused_upconcat.h"
#include <immintrin.h>

/* Unaligned 256-bit moves preserve every FP32 bit (including NaN payloads).
 * Keep 128 source channels (16 YMM registers) live and replicate them before
 * loading the next block. This is the measured AVX2 register-block strategy. */
#define LOAD(P) _mm256_loadu_si256((const __m256i *)(const void *)(P))
#define STORE(P,V) _mm256_storeu_si256((__m256i *)(void *)(P),(V))
#define LOAD_128(P) \
    const __m256i v0=LOAD((P)),v1=LOAD((P)+8),v2=LOAD((P)+16),v3=LOAD((P)+24); \
    const __m256i v4=LOAD((P)+32),v5=LOAD((P)+40),v6=LOAD((P)+48),v7=LOAD((P)+56); \
    const __m256i v8=LOAD((P)+64),v9=LOAD((P)+72),v10=LOAD((P)+80),v11=LOAD((P)+88); \
    const __m256i v12=LOAD((P)+96),v13=LOAD((P)+104),v14=LOAD((P)+112),v15=LOAD((P)+120)
#define STORE_128(P) do { \
    STORE((P),v0); STORE((P)+8,v1); STORE((P)+16,v2); STORE((P)+24,v3); \
    STORE((P)+32,v4); STORE((P)+40,v5); STORE((P)+48,v6); STORE((P)+56,v7); \
    STORE((P)+64,v8); STORE((P)+72,v9); STORE((P)+80,v10); STORE((P)+88,v11); \
    STORE((P)+96,v12); STORE((P)+104,v13); STORE((P)+112,v14); STORE((P)+120,v15); \
} while(0)

/* Copy the skip suffix of a pixel. N is 64 or 128 floats. */
#define COPY_SKIP(D,S,N) do { \
    for (size_t k=0; k<(N); k+=32) { \
        const __m256i q0=LOAD((S)+k),q1=LOAD((S)+k+8); \
        const __m256i q2=LOAD((S)+k+16),q3=LOAD((S)+k+24); \
        STORE((D)+k,q0); STORE((D)+k+8,q1); \
        STORE((D)+k+16,q2); STORE((D)+k+24,q3); \
    } \
} while(0)

/* Stage A: QUAD order. Load each 128-channel input block only once, then
 * store it into all four output pixels before copying their distinct skips. */
void fused_upconcat_a(const float *restrict up, const float *restrict skip,
                      float *restrict out, size_t h, size_t w) {
    enum { CU=256, CS=128, CO=CU+CS };
    const size_t out_row=2*w*CO, skip_row=2*w*CS;
    for (size_t y=0; y<h; ++y) {
        float *top=out+2*y*out_row, *bottom=top+out_row;
        const float *st=skip+2*y*skip_row, *sb=st+skip_row;
        for (size_t x=0; x<w; ++x) {
            const float *u=up+(y*w+x)*CU;
            float *a=top+2*x*CO, *b=a+CO;
            float *c=bottom+2*x*CO, *d=c+CO;
            const float *sa=st+2*x*CS, *sc=sb+2*x*CS;
            for (size_t k=0; k<CU; k+=128) {
                LOAD_128(u+k);
                STORE_128(a+k);
                __asm__ __volatile__("" ::: "memory"); /* Preserve tested store order. */
                STORE_128(b+k);
                __asm__ __volatile__("" ::: "memory");
                STORE_128(c+k);
                __asm__ __volatile__("" ::: "memory");
                STORE_128(d+k);
            }
            COPY_SKIP(a+CU,sa,CS);
            COPY_SKIP(b+CU,sa+CS,CS);
            COPY_SKIP(c+CU,sc,CS);
            COPY_SKIP(d+CU,sc+CS,CS);
        }
    }
}

/* Stage B: ROW order. Complete one output row at a time for more sequential
 * stores; source pixels are reread for the second row, generally from cache. */
void fused_upconcat_b(const float *restrict up, const float *restrict skip,
                      float *restrict out, size_t h, size_t w) {
    enum { CU=128, CS=64, CO=CU+CS };
    const size_t out_row=2*w*CO, skip_row=2*w*CS;
    for (size_t y=0; y<h; ++y) for (size_t dy=0; dy<2; ++dy) {
        float *o=out+(2*y+dy)*out_row;
        const float *s=skip+(2*y+dy)*skip_row;
        for (size_t x=0; x<w; ++x) {
            const float *u=up+(y*w+x)*CU;
            float *a=o+2*x*CO, *b=a+CO;
            const float *sa=s+2*x*CS;
            LOAD_128(u);
            STORE_128(a);
            __asm__ __volatile__("" ::: "memory"); /* Preserve tested store order. */
            STORE_128(b);
            COPY_SKIP(a+CU,sa,CS);
            COPY_SKIP(b+CU,sa+CS,CS);
        }
    }
}
