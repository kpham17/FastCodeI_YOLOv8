#include "universal_concat.h"
#include "memory_kernels_internal.h"
int universal_concat_f32(const universal_concat_view *v,size_t k,float *out,
                                    size_t cap,size_t n,size_t h,size_t w) {
    size_t p,np,co=0,meta,outn,outb;
    if(!v || !out || !k || !n || !h || !w) return MEMORY_BAD_ARGUMENT;
    if(!memory_mul(k,sizeof(*v),&meta) || meta>(size_t)PTRDIFF_MAX ||
       (uintptr_t)v>UINTPTR_MAX-meta || !memory_mul(h,w,&p) || !memory_mul(n,p,&np))
        return MEMORY_OVERFLOW;
    for(size_t i=0;i<k;++i) {
        size_t count,bytes;
        if(!v[i].data || !v[i].channels || v[i].channels%16) return MEMORY_BAD_ARGUMENT;
        if(co>SIZE_MAX-v[i].channels || !memory_mul(np,v[i].channels,&count) ||
           !memory_span(v[i].data,count,&bytes)) return MEMORY_OVERFLOW;
        if(v[i].capacity<count) return MEMORY_CAPACITY;
        co+=v[i].channels;
    }
    if(!memory_mul(np,co,&outn) || !memory_span(out,outn,&outb)) return MEMORY_OVERFLOW;
    if(cap<outn) return MEMORY_CAPACITY;
    if(memory_overlap(out,outb,v,meta)) return MEMORY_OVERLAP;
    for(size_t i=0;i<k;++i)
        if(memory_overlap(out,outb,v[i].data,np*v[i].channels*4)) return MEMORY_OVERLAP;
    int selected=memory_width(outb<=32768/2);
    size_t cursor=0;
    for(size_t b=0;b<n;++b) for(size_t i=0;i<k;++i) {
        size_t count=p*v[i].channels;
        memory_copy(out+cursor,v[i].data+b*count,count,selected);cursor+=count;
    }
    return MEMORY_OK;
}
