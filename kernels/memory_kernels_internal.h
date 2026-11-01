#ifndef MEMORY_KERNELS_INTERNAL_H
#define MEMORY_KERNELS_INTERNAL_H
#include "memory_layout.h"
#include <stdint.h>
#include <string.h>
#if !defined(MEMORY_PORTABLE) && defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define MEMORY_VECTOR 1
#define MEMORY_TARGET2 __attribute__((target("avx2")))
#define MEMORY_TARGET5 __attribute__((target("avx512f")))
#else
#define MEMORY_VECTOR 0
#endif
_Static_assert(sizeof(float)==4,"FP32 requires four-byte floats");
static inline int memory_mul(size_t a,size_t b,size_t *r) {
    if(a && b>SIZE_MAX/a) return 0;
    *r=a*b;return 1;
}
static inline int memory_span(const void *p,size_t n,size_t *bytes) {
    return memory_mul(n,4,bytes) && *bytes<=(size_t)PTRDIFF_MAX &&
           (uintptr_t)p<=UINTPTR_MAX-*bytes;
}
static inline int memory_overlap(const void *a,size_t an,const void *b,size_t bn) {
    uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
    return x<=y?y-x<an:x-y<bn;
}
/* Paired measurements on ece020 favor ZMM for L1-sized work and AVX2
 * for production streams. A single threshold serves all three operations. */
static inline int memory_width(int fits_l1) {
#if MEMORY_VECTOR
    if(fits_l1 && __builtin_cpu_supports("avx512f")) return 16;
    if(__builtin_cpu_supports("avx2")) return 8;
#else
    (void)fits_l1;
#endif
    return 0;
}
#if MEMORY_VECTOR
#define MEMORY_COPY(NAME,TARGET,LOAD,STORE,WIDTH) \
TARGET static inline void NAME(float *d,const float *s,size_t n) { \
    size_t i=0; \
    for(;n-i>=4*WIDTH;i+=4*WIDTH) \
        for(size_t k=0;k<4*WIDTH;k+=WIDTH) STORE(d+i+k,LOAD(s+i+k)); \
    for(;i<n;i+=WIDTH) STORE(d+i,LOAD(s+i)); \
}
/* All spans contain a multiple of 16 floats; no vector tail is needed. */
MEMORY_COPY(memory_copy_avx2,MEMORY_TARGET2,_mm256_loadu_ps,_mm256_storeu_ps,8)
MEMORY_COPY(memory_copy_zmm,MEMORY_TARGET5,_mm512_loadu_ps,_mm512_storeu_ps,16)
#undef MEMORY_COPY
#endif
static inline void memory_copy(float *d,const float *s,size_t n,int width) {
#if MEMORY_VECTOR
    if(width==16) {memory_copy_zmm(d,s,n);return;}
    if(width==8) {memory_copy_avx2(d,s,n);return;}
#else
    (void)width;
#endif
    memcpy(d,s,n*4);
}
#endif
