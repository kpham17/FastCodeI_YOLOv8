/* Materialized channel halves of dense FP32 NHWC [pixels,2*C].
 * Unaligned integer moves preserve all FP32 representations. Constant model
 * widths remove inner-loop bounds/address work; other widths retain tails.
 * Cached stores suit immediate consumers. No allocation or CPU dispatcher.
 */
#include "chunk_nhwc.h"
#include <immintrin.h>
#include <string.h>

#define AVX2 __attribute__((target("avx2")))
#define AVX512 __attribute__((target("avx512f")))
#define INLINE __attribute__((always_inline)) inline

AVX2 static INLINE void copy_half(float *restrict d,const float *restrict s,
                                  size_t pixels,size_t C) {
    for(size_t p=0;p<pixels;p++) {
        const float *in=s+p*2*C;
        size_t c=0;
        #pragma GCC unroll 16
        for(;C-c>=8;c+=8) {
            __m256i v=_mm256_loadu_si256((const __m256i_u*)(in+c));
            _mm256_storeu_si256((__m256i_u*)(d+c),v);
        }
        if(c<C)memcpy(d+c,in+c,(C-c)*sizeof(float));
        d+=C;
    }
}

AVX512 static INLINE void copy_both(float *restrict a,float *restrict b,
                                    const float *restrict s,size_t pixels,size_t C) {
    for(size_t p=0;p<pixels;p++) {
        size_t c=0;
        #pragma GCC unroll 8
        for(;C-c>=16;c+=16) {
            __m512i x=_mm512_loadu_si512(s+c),y=_mm512_loadu_si512(s+C+c);
            _mm512_storeu_si512(a+c,x);_mm512_storeu_si512(b+c,y);
        }
        if(c<C){memcpy(a+c,s+c,(C-c)*sizeof(float));memcpy(b+c,s+C+c,(C-c)*sizeof(float));}
        s+=2*C;a+=C;b+=C;
    }
}

/* The same four widths serve P2--P5 at all supported image resolutions. */
#define WIDTHS(CALL) switch(C){ \
    case 16:CALL(16);break;case 32:CALL(32);break; \
    case 64:CALL(64);break;case 128:CALL(128);break;default:CALL(C);break;}
AVX2 static void one(float*d,const float*s,size_t pixels,size_t C) {
#define COPY(CW) copy_half(d,s,pixels,CW)
    WIDTHS(COPY)
#undef COPY
}
AVX512 static void both(float*a,float*b,const float*s,size_t pixels,size_t C) {
#define COPY(CW) copy_both(a,b,s,pixels,CW)
    WIDTHS(COPY)
#undef COPY
}
#undef WIDTHS

void chunk2_nhwc_f32(float*a,float*b,const float*s,size_t pixels,size_t C) {
    if(!pixels||!C||(!a&&!b))return;
    if(!a||!b)one(a?a:b,s+(a?0:C),pixels,C);
    else if(C==128){one(a,s,pixels,C);one(b,s+C,pixels,C);}
    else both(a,b,s,pixels,C);
}
