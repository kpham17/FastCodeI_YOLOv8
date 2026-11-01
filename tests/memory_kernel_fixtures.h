#ifndef MEMORY_KERNELS_FIXTURES_H
#define MEMORY_KERNELS_FIXTURES_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
/* Test/benchmark-only layout conversion, never called by a kernel. Physical
 * traversal increments a cursor; logical address uses NHWC coordinates. */
static void memory_pack(float *d,const float *s,size_t n,size_t h,size_t w,size_t c) {
    size_t cursor=0;
    for(size_t b=0;b<n;++b) for(size_t cb=0;cb<c;cb+=16)
        for(size_t y=0;y<h;++y) for(size_t x=0;x<w;++x)
            for(size_t lane=0;lane<16;++lane)
                memcpy(d+cursor++,s+((b*h+y)*w+x)*c+cb+lane,4);
}
static void memory_unpack(float *d,const float *s,size_t n,size_t h,size_t w,size_t c) {
    size_t cursor=0;
    for(size_t b=0;b<n;++b) for(size_t cb=0;cb<c;cb+=16)
        for(size_t y=0;y<h;++y) for(size_t x=0;x<w;++x)
            for(size_t lane=0;lane<16;++lane)
                memcpy(d+((b*h+y)*w+x)*c+cb+lane,s+cursor++,4);
}
static inline void memory_bits(float *p,size_t n,uint32_t seed) {
    const uint32_t specials[]={0,0x80000000u,0x7f800000u,0xff800000u,
                              0x7fc12345u,0x7fa00123u,1,0x807fffffu};
    for(size_t i=0;i<n;++i) {
        seed=seed*1664525u+1013904223u;
        uint32_t x=i<8?specials[i]:seed;
        memcpy(p+i,&x,4);
    }
}
static void memory_finite(float *p,size_t n,int seed) {
    for(size_t i=0;i<n;++i) p[i]=(float)((int)(i%127)-63+seed)*0.00001f;
}
static float *memory_alloc(size_t n) {
    void *p=NULL;
    if(posix_memalign(&p,64,n*sizeof(float))) abort();
    return (float *)p;
}
#endif
