/* Production-only benchmark: 12 YOLOv8n shapes, dense and chunk-B shortcuts.
 * Allocation, scalar validation and resetting the in-place output are untimed.
 * Each result is a median of nine warmed batches targeting 8 ms each.
 * Pin the process before allocation, e.g. taskset -c 0 ./bench_residual_add.x.
 */
#define _GNU_SOURCE
#include "residual_add.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static volatile float sink;
static uint64_t now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC_RAW, &t)) abort();
    return (uint64_t)t.tv_sec*1000000000ull+(uint64_t)t.tv_nsec;
}
static float *allocate(size_t n) {
    void *p=NULL;
    if (posix_memalign(&p,64,n*sizeof(float))) abort();
    return p;
}
__attribute__((optimize("no-tree-vectorize"),noinline))
static void reference(float*d,const float*a,const float*b,size_t P,size_t C,size_t S) {
    for (size_t p=0;p<P;++p)
        for (size_t c=0;c<C;++c) d[p*C+c]=a[p*C+c]+b[p*S+c];
}
static void invoke(float*d,const float*b,size_t P,size_t C,size_t S) {
    if (S==C) residual_add_contiguous(d,d,b,P,C);
    else residual_add_strided(d,d,b,P,C,S);
}
static double batch(float*d,const float*b,size_t P,size_t C,size_t S,size_t loops) {
    uint64_t start=now();
    for (size_t i=0;i<loops;++i) invoke(d,b,P,C,S);
    uint64_t end=now();
    sink=d[(P*C)/2];
    return (double)(end-start);
}
static int compare(const void*a,const void*b) {
    double x=*(const double*)a,y=*(const double*)b;
    return (x>y)-(x<y);
}
int main(int argc,char**argv) {
    int reverse=argc==2 && !strcmp(argv[1],"--reverse");
    if (argc>2 || (argc==2 && !reverse)) {
        fprintf(stderr,"Usage: %s [--reverse]\n",argv[0]);
        return 2;
    }
    const size_t images[]={640,1024,2048};
    puts("image,level,H,W,C,layout,median_ns,min_ns,p25_ns,p75_ns,logical_GBps,loops");
    for (size_t i=0;i<24;++i) {
        size_t k=reverse?23-i:i;
        size_t image=images[k/8],level=(k%8)/2+2,view=k%2;
        size_t side=image>>level,P=side*side,C=(size_t)16<<(level-2);
        size_t S=(view?2:1)*C,n=P*C;
        float *a=allocate(n),*d=allocate(n),*parent=allocate(P*S),*gold=allocate(n);
        const float *b=parent+(view?C:0);
        for (size_t j=0;j<n;++j) a[j]=(float)((int)(j%127)-63)*0.0001f;
        for (size_t j=0;j<P*S;++j) parent[j]=(float)((int)(j%113)-56)*0.00001f;
        reference(gold,a,b,P,C,S);
        memcpy(d,a,n*sizeof(float));
        invoke(d,b,P,C,S);
        if (memcmp(d,gold,n*sizeof(float))) {
            fprintf(stderr,"FAIL %zu P%zu %s\n",image,level,view?"strided":"contiguous");
            return 1;
        }

        /* Calibrate long enough to amortize clock/call overhead. Inputs stay
         * finite across repeated adds; every subsequent batch starts fresh. */
        size_t probe=1;
        double elapsed;
        do {
            memcpy(d,a,n*sizeof(float));
            elapsed=batch(d,b,P,C,S,probe);
            if (elapsed<1000000) probe*=2;
        } while (elapsed<1000000 && probe<1048576);
        size_t loops=(size_t)(probe*8000000.0/elapsed)+1;
        double samples[9];
        for (int rep=0;rep<9;++rep) {
            memcpy(d,a,n*sizeof(float));
            samples[rep]=batch(d,b,P,C,S,loops)/loops;
        }
        qsort(samples,9,sizeof(double),compare);
        /* 12 useful bytes/add: conv load, shortcut load, destination store.
         * This reports logical traffic, not physical DRAM-controller traffic. */
        printf("%zu,P%zu,%zu,%zu,%zu,%s,%.2f,%.2f,%.2f,%.2f,%.2f,%zu\n",
               image,level,side,side,C,view?"strided":"contiguous",
               samples[4],samples[0],samples[2],samples[6],12.0*n/samples[4],loops);
        free(a);free(d);free(parent);free(gold);
        fflush(stdout);
    }
    return 0;
}
