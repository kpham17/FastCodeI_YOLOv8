#define _GNU_SOURCE
#include "residual_add.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <sys/mman.h>
#include <unistd.h>

/* Deliberately independent scalar oracle: every output is computed from the
 * initial conv value, and the selected shortcut channels, pixel by pixel. */
static void reference(float *out, const float *a, const float *b,
                      size_t P, size_t C, size_t pitch) {
    for (size_t p=0; p<P; ++p)
        for (size_t c=0; c<C; ++c)
            out[p*C+c] = a[p*C+c] + b[p*pitch+c];
}
static float sample(size_t i) {
    switch (i%19) {
        case 0: return 0.0f;
        case 1: return -0.0f;
        case 2: return INFINITY;
        case 3: return -INFINITY;
        default: return (float)((int)(i%103)-50)/19.0f;
    }
}
static void *aligned(size_t count) {
    void *v=NULL;
    if (posix_memalign(&v,64,(count?count:1)*sizeof(float))) abort();
    return v;
}
static size_t test_count=0;
static void check(size_t P, size_t C, size_t pitch, int shift) {
    const size_t n=P*C;
    const size_t parent_count=P*pitch+C+32;
    float *ap=aligned(n+16), *dp=aligned(n+16), *xp=aligned(n+16);
    float *parent=aligned(parent_count), *parent_saved=aligned(parent_count);
    float *a=ap+shift, *d=dp+shift, *expected=xp+shift;
    /* For pitch=2*C this really is the second C-channel half of cv1. */
    float *shortcut=parent+(pitch>=2*C ? C : (pitch>C ? 1:0))+shift;
    for (size_t i=0; i<n; ++i) a[i]=sample(i+P+C);
    for (size_t i=0; i<parent_count; ++i) parent[i]=sample(i*7+11);
    memcpy(parent_saved,parent,parent_count*sizeof(float));
    reference(expected,a,shortcut,P,C,pitch);

    for (int inplace=0; inplace<=1; ++inplace) {
        for (size_t i=0; i<n+16; ++i) dp[i]=12345.0f;
        if (inplace) memcpy(d,a,n*sizeof(float));

        if (pitch==C)
            residual_add_contiguous(d,inplace?d:a,shortcut,P,C);
        else
            residual_add_strided(d,inplace?d:a,shortcut,P,C,pitch);

        if (memcmp(d,expected,n*sizeof(float)) ||
            memcmp(parent,parent_saved,parent_count*sizeof(float)) ||
            dp[shift+n]!=12345.0f || dp[shift+n+1]!=12345.0f ||
            (shift && dp[0]!=12345.0f)) {
            fprintf(stderr,"FAIL P=%zu C=%zu pitch=%zu shift=%d inplace=%d\n",
                    P,C,pitch,shift,inplace);
            exit(EXIT_FAILURE);
        }
        if (!inplace) {
            for (size_t i=0;i<n;++i) {
                if (a[i]!=sample(i+P+C) && !(isinf(a[i]) && isinf(sample(i+P+C)))) {
                    fprintf(stderr,"FAIL conv modified\n");exit(EXIT_FAILURE);
                }
            }
        }
    }
    free(ap);free(dp);free(xp);free(parent);free(parent_saved);
    ++test_count;
}
/* End each input/output allocation immediately before an inaccessible page.
 * Missing tails, reading parent gaps and crossing the output bound are observable. */
typedef struct { void *base; size_t bytes; float *data; } Buffer;
static Buffer guarded(size_t n) {
    size_t page=(size_t)sysconf(_SC_PAGESIZE);
    size_t bytes=((n*sizeof(float)+page-1)/page+1)*page;
    void *base=mmap(NULL,bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
    assert(base!=MAP_FAILED);
    assert(!mprotect((char*)base+bytes-page,page,PROT_NONE));
    return (Buffer){base,bytes,(float*)((char*)base+bytes-page)-n};
}
static void release(Buffer b) { assert(!munmap(b.base,b.bytes)); }
__attribute__((optimize("no-tree-vectorize"),noinline))
static void oracle(float*d,const float*a,const float*b,size_t P,size_t C,size_t S) {
    for(size_t p=0;p<P;++p) for(size_t c=0;c<C;++c) d[p*C+c]=a[p*C+c]+b[p*S+c];
}
static void test_guard_pages(void) {
    residual_add_contiguous(NULL,NULL,NULL,0,128);
    residual_add_contiguous(NULL,NULL,NULL,17,0);
    residual_add_strided(NULL,NULL,NULL,0,128,256);
    residual_add_strided(NULL,NULL,NULL,17,0,3);
    size_t count=0;
    const size_t channels[]={1,7,8,15,16,17,31,32,33,63,64,65,127,128,129};
    for(size_t ci=0;ci<sizeof(channels)/sizeof(*channels);++ci)
    for(size_t P=1;P<=17;P+=4) for(size_t layout=0;layout<4;++layout) {
        size_t C=channels[ci],S=layout==0?C:layout==1?C+1:layout==2?2*C:2*C+3;
        size_t n=P*C,bn=(P-1)*S+C;
        Buffer a=guarded(n),b=guarded(bn),d=guarded(n),e=guarded(n);
        for(size_t i=0;i<n;++i) a.data[i]=(float)((int)(i%127)-63)*0.0001f;
        for(size_t i=0;i<bn;++i) b.data[i]=(float)((int)(i%113)-56)*0.00003f;
        float *saved=malloc(bn*sizeof(float)); assert(saved);
        memcpy(saved,b.data,bn*sizeof(float));
        oracle(e.data,a.data,b.data,P,C,S);
        for(int inplace=0;inplace<2;++inplace) {
            if(inplace)memcpy(d.data,a.data,n*sizeof(float));
            /* Also exercise the strided entry point with pitch == C. */
            residual_add_strided(d.data,inplace?d.data:a.data,b.data,P,C,S);
            if(memcmp(d.data,e.data,n*sizeof(float)) || memcmp(saved,b.data,bn*sizeof(float))) {
                fprintf(stderr,"FAIL guard P=%zu C=%zu pitch=%zu inplace=%d\n",P,C,S,inplace); exit(1);
            }
        }
        free(saved);release(a);release(b);release(d);release(e);++count;
    }
    printf("PASS: %zu guard-page configurations, both output modes, zero dimensions\n",count);
}

int main(void) {
    test_guard_pages();
    const size_t Cs[]={1,7,8,15,16,17,31,32,33,64,128};
    const size_t Ps[]={0,1,5,37};
    for (size_t i=0;i<sizeof(Cs)/sizeof(Cs[0]);++i)
        for (size_t j=0;j<sizeof(Ps)/sizeof(Ps[0]);++j)
            for (int kind=0;kind<3;++kind)
                for (int shifted=0;shifted<2;++shifted) {
                    size_t C=Cs[i],pitch=kind==0?C:kind==1?2*C:2*C+3;
                    check(Ps[j],C,pitch,shifted);
                }
    /* All fixed YOLOv8n bottleneck hidden-channel shapes. */
    const size_t images[]={640,1024,2048};
    for (size_t i=0;i<3;++i)
        for (size_t level=2;level<=5;++level) {
            size_t side=images[i] >> level;
            size_t C=(size_t)16 << (level-2);
            check(side*side,C,C,0);      /* densely stored shortcut */
            check(side*side,C,2*C,0);    /* zero-copy C2f chunk B */
            check(side*side,C,C,1);      /* unaligned dense model shape */
            check(side*side,C,2*C,1);    /* unaligned shortcut-view shape */
        }
    printf("PASS: %zu configurations (each in-place and out-of-place), bitwise output, guards, read-only shortcut\n",test_count);
    return 0;
}
