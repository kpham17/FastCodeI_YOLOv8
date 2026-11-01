#include "residual_add.h"
#include "memory_kernels_internal.h"
static void add_scalar(float *d,const float *a,const float *b,size_t n) {
    for(size_t i=0;i<n;++i) d[i]=a[i]+b[i];
}
#if MEMORY_VECTOR
MEMORY_TARGET2 static void add_ymm(float *d,const float *a,const float *b,size_t n) {
    size_t i=0;
    for(;n-i>=32;i+=32) for(size_t k=0;k<32;k+=8)
        _mm256_storeu_ps(d+i+k,_mm256_add_ps(_mm256_loadu_ps(a+i+k),_mm256_loadu_ps(b+i+k)));
    for(;i<n;i+=8) _mm256_storeu_ps(d+i,_mm256_add_ps(_mm256_loadu_ps(a+i),_mm256_loadu_ps(b+i)));
}
MEMORY_TARGET5 static void add_zmm(float *d,const float *a,const float *b,size_t n) {
    for(;n>=128;n-=128,d+=128,a+=128,b+=128)
        for(size_t k=0;k<128;k+=16)
            _mm512_storeu_ps(d+k,_mm512_add_ps(_mm512_loadu_ps(a+k),_mm512_loadu_ps(b+k)));
    for(;n;n-=16,d+=16,a+=16,b+=16)
        _mm512_storeu_ps(d,_mm512_add_ps(_mm512_loadu_ps(a),_mm512_loadu_ps(b)));
}
#endif
int residual_add(float *d,size_t dc,const float *a,size_t ac,
                                 const float *b,size_t bc,size_t h,size_t w,size_t c) {
    size_t p,n,db,ab,bb;
    if(!d || !a || !b || !h || !w || !c || c%16) return MEMORY_BAD_ARGUMENT;
    if(!memory_mul(h,w,&p) || !memory_mul(p,c,&n) || !memory_span(d,n,&db) ||
       !memory_span(a,n,&ab) || !memory_span(b,n,&bb)) return MEMORY_OVERFLOW;
    if(dc<n || ac<n || bc<n) return MEMORY_CAPACITY;
    if((d!=a && memory_overlap(d,db,a,ab)) || memory_overlap(d,db,b,bb)) return MEMORY_OVERLAP;
#if MEMORY_VECTOR
    int selected=memory_width(db<=32768/3);
    if(selected==16) {add_zmm(d,a,b,n);return MEMORY_OK;}
    if(selected==8) {add_ymm(d,a,b,n);return MEMORY_OK;}
#endif
    add_scalar(d,a,b,n);return MEMORY_OK;
}
