#include "fused_upconcat.h"
#include <immintrin.h>

#if !defined(__AVX512F__) || !defined(__AVX512VL__)
#error "fused_upconcat requires AVX-512F and AVX-512VL build flags"
#endif

/* VL_MIXED: 256-bit moves with access to YMM0-31. Keep 128 source
 * channels live while copying each pixel's skip channels with spare registers.
 * Integer moves preserve every FP32 bit, including NaN payloads. */
#define ORDER() __asm__ __volatile__("" ::: "memory")
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

/* Complete this source block, then append the pixel's skip suffix.
 * ORDER is a compiler barrier, not a CPU fence. */
#define STORE_PIXEL(D,K,S,N) do { \
    STORE_128((D)+(K)); ORDER(); \
    COPY_SKIP((D)+(K)+128,(S),(N)); ORDER(); \
} while(0)

/* Stage A: write the first 128 channels to all four output pixels, then
 * interleave the second half and skips. Each source block is read once. */
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
            {
                LOAD_128(u); ORDER();
                STORE_128(a); ORDER();
                STORE_128(b); ORDER();
                STORE_128(c); ORDER();
                STORE_128(d); ORDER();
            }
            {
                LOAD_128(u+128); ORDER();
                STORE_PIXEL(a,128,sa,CS);
                STORE_PIXEL(b,128,sa+CS,CS);
                STORE_PIXEL(c,128,sc,CS);
                STORE_PIXEL(d,128,sc+CS,CS);
            }
        }
    }
}

/* Stage B: keep the whole source pixel live across all four skip copies.
 * Store A up, A skip, B up, B skip, C up, C skip, D up, D skip. */
void fused_upconcat_b(const float *restrict up, const float *restrict skip,
                      float *restrict out, size_t h, size_t w) {
    enum { CU=128, CS=64, CO=CU+CS };
    const size_t out_row=2*w*CO, skip_row=2*w*CS;
    for (size_t y=0; y<h; ++y) {
        float *top=out+2*y*out_row, *bottom=top+out_row;
        const float *st=skip+2*y*skip_row, *sb=st+skip_row;
        for (size_t x=0; x<w; ++x) {
            const float *u=up+(y*w+x)*CU;
            float *a=top+2*x*CO, *b=a+CO;
            float *c=bottom+2*x*CO, *d=c+CO;
            const float *sa=st+2*x*CS, *sc=sb+2*x*CS;
            LOAD_128(u); ORDER();
            STORE_PIXEL(a,0,sa,CS);
            STORE_PIXEL(b,0,sa+CS,CS);
            STORE_PIXEL(c,0,sc,CS);
            STORE_PIXEL(d,0,sc+CS,CS);
        }
    }
}
