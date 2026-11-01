#include "fused_upconcat.h"
#include "memory_kernels_internal.h"
static void up_reference(const float *u,float *d,size_t h,size_t w,size_t cu) {
    size_t p=h*w;
    for(size_t cb=0;cb<cu/16;++cb) for(size_t y=0;y<h;++y) for(size_t x=0;x<w;++x) {
        const float *src=u+(cb*p+y*w+x)*16;
        float *dst=d+(cb*4*p+2*y*2*w+2*x)*16;
        memcpy(dst,src,64);memcpy(dst+16,src,64);
        memcpy(dst+2*w*16,src,64);memcpy(dst+2*w*16+16,src,64);
    }
}
#if MEMORY_VECTOR
/* Load one 16-channel pixel, write its four output positions directly. */
#define DEFINE_UP(NAME,TARGET,VEC,LOAD,STORE,WIDTH) \
TARGET static void NAME(const float *u,float *d,size_t h,size_t w,size_t cu) { \
    /* W=1 makes each source pixel's four output vectors consecutive. */ \
    if(w==1) { \
        for(size_t i=0;i<h*cu;i+=16) for(size_t lane=0;lane<16;lane+=WIDTH) { \
            VEC v=LOAD(u+i+lane); \
            STORE(d+4*i+lane,v);STORE(d+4*i+16+lane,v); \
            STORE(d+4*i+32+lane,v);STORE(d+4*i+48+lane,v); \
        } \
        return; \
    } \
    size_t p=h*w,row=2*w*16; \
    for(size_t cb=0;cb<cu/16;++cb) for(size_t y=0;y<h;++y) { \
        const float *s=u+(cb*p+y*w)*16;float *a=d+cb*p*64+2*y*row; \
        for(size_t x=0;x<w;++x) for(size_t lane=0;lane<16;lane+=WIDTH) { \
            VEC v=LOAD(s+x*16+lane); \
            STORE(a+x*32+lane,v);STORE(a+x*32+16+lane,v); \
            STORE(a+row+x*32+lane,v);STORE(a+row+x*32+16+lane,v); \
        } \
    } \
}
DEFINE_UP(up_zmm_quad,MEMORY_TARGET5,__m512,_mm512_loadu_ps,_mm512_storeu_ps,16)
DEFINE_UP(up_ymm_quad,MEMORY_TARGET2,__m256,_mm256_loadu_ps,_mm256_storeu_ps,8)
#undef DEFINE_UP
#endif
int fused_upconcat_f32(const float *u,size_t uc,const float *s,size_t sc,
                              float *d,size_t dc,size_t h,size_t w,size_t cu,size_t cs) {
    size_t p,op,nu,ns,no,ub,sb,db;
    if(!u || !s || !d || !h || !w || !cu || !cs || cu%16 || cs%16) return MEMORY_BAD_ARGUMENT;
    if(h>SIZE_MAX/2 || w>SIZE_MAX/2 || cu>SIZE_MAX-cs ||
       !memory_mul(h,w,&p) || !memory_mul(p,4,&op) || !memory_mul(p,cu,&nu) ||
       !memory_mul(op,cs,&ns) || !memory_mul(op,cu+cs,&no) ||
       !memory_span(u,nu,&ub) || !memory_span(s,ns,&sb) || !memory_span(d,no,&db))
        return MEMORY_OVERFLOW;
    if(uc<nu || sc<ns || dc<no) return MEMORY_CAPACITY;
    if(memory_overlap(d,db,u,ub) || memory_overlap(d,db,s,sb)) return MEMORY_OVERLAP;
    int width=memory_width(ub<=32768 && sb<=32768-ub && db<=32768-ub-sb);
#if MEMORY_VECTOR
    if(width==16) up_zmm_quad(u,d,h,w,cu);
    else if(width==8) up_ymm_quad(u,d,h,w,cu);
    else
#endif
        up_reference(u,d,h,w,cu);
    memory_copy(d+op*cu,s,ns,width);
    return MEMORY_OK;
}
int fused_upconcat_a(const float *u,size_t uc,const float *s,size_t sc,
                            float *d,size_t dc,size_t h,size_t w) {
    return fused_upconcat_f32(u,uc,s,sc,d,dc,h,w,256,128);
}
int fused_upconcat_b(const float *u,size_t uc,const float *s,size_t sc,
                            float *d,size_t dc,size_t h,size_t w) {
    return fused_upconcat_f32(u,uc,s,sc,d,dc,h,w,128,64);
}
