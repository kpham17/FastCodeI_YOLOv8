/* Whole-path comparison: prepared dense input, chunk-copy + dense consumer,
 * and the existing zero-copy strided consumer. Allocation/setup is untimed
 * except the convolution's own existing per-call allocation. */
#define _GNU_SOURCE
#include "chunk_nhwc.h"
#include "residual_add.h"
#include "conv2D.h"
#include <immintrin.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static volatile float sink;
static uint64_t now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return (uint64_t)t.tv_sec*1000000000ull+t.tv_nsec;}
static float*allocate(size_t n){float*p=NULL;if(posix_memalign((void**)&p,64,n*4))abort();return p;}
static int cmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return(x>y)-(x<y);}
static void add_call(int mode,float*d,float*dense,const float*parent,size_t P,size_t C){
    if(mode==1)chunk2_nhwc_f32(NULL,dense,parent,P,C);
    if(mode==2)residual_add_strided(d,d,parent+C,P,C,2*C);
    else residual_add_contiguous(d,d,dense,P,C);
}
static double add_batch(int mode,float*d,float*dense,const float*parent,size_t P,size_t C,size_t loops){
    _mm_mfence();uint64_t t=now();
    for(size_t i=0;i<loops;i++){add_call(mode,d,dense,parent,P,C);__asm__ __volatile__("":::"memory");}
    _mm_mfence();uint64_t dt=now()-t;sink=d[P*C/2];return(double)dt/loops;
}
__attribute__((noinline))
static float*conv_call(int mode,float*dense,float*parent,float*packed,size_t H,size_t C){
    if(mode==2)return conv_3x3_strided(parent,(int)H,(int)H,(int)C,packed,(int)C,1,(int)(2*C),(int)C);
    if(mode==1)chunk2_nhwc_f32(NULL,dense,parent,H*H,C);
    return conv_3x3(dense,(int)H,(int)H,(int)C,packed,(int)C,1);
}
int main(int argc,char**argv){
    int run=argc==2?atoi(argv[1]):1;cpu_set_t set;CPU_ZERO(&set);CPU_SET(2,&set);if(sched_setaffinity(0,sizeof(set),&set))abort();
    const size_t images[]={640,1024,2048};const char*names[]={"dense_prepared","copy_plus_dense","strided_view"};
    puts("consumer,image,level,H,C,path,run,round,loops,ns,cpu,validation");
    for(size_t k=0;k<12;k++){
        size_t v=run%2==0?11-k:k,image=images[v/4],level=v%4+2,H=image>>level,C=(size_t)16<<(level-2),P=H*H,n=P*C;
        float*parent=allocate(2*n),*dense=allocate(n),*a=allocate(n),*d=allocate(n),*gold=allocate(n);
        for(size_t i=0;i<2*n;i++)parent[i]=(float)((int)(i%113)-56)*0.00001f;
        for(size_t i=0;i<n;i++)a[i]=(float)((int)(i%127)-63)*0.0001f;
        chunk2_nhwc_f32(NULL,dense,parent,P,C);
        for(size_t p=0;p<P;p++)for(size_t c=0;c<C;c++)gold[p*C+c]=a[p*C+c]+parent[p*2*C+C+c];
        size_t loops[3];
        for(int mode=0;mode<3;mode++){
            memcpy(d,a,n*4);add_call(mode,d,dense,parent,P,C);
            if(memcmp(d,gold,n*4)){fprintf(stderr,"FAIL add\n");return 1;}
            memcpy(d,a,n*4);double ns=add_batch(mode,d,dense,parent,P,C,1);loops[mode]=(size_t)(3000000/ns)+1;if(loops[mode]>65536)loops[mode]=65536;
        }
        for(int round=0;round<21;round++)for(int j=0;j<3;j++){
            int mode=(round+j+run)%3;memcpy(d,a,n*4);add_call(mode,d,dense,parent,P,C);add_call(mode,d,dense,parent,P,C);
            double ns=add_batch(mode,d,dense,parent,P,C,loops[mode]);
            printf("add,%zu,P%zu,%zu,%zu,%s,%d,%d,%zu,%.3f,2,PASS\n",image,level,H,C,names[mode],run,round,loops[mode],ns);
        }
        float *weights=allocate(9*C*C);for(size_t i=0;i<9*C*C;i++)weights[i]=(float)((int)(i%103)-51)*0.003f;
        float*packed=kernel_pack(weights,(int)(9*C),(int)C);if(!packed)abort();
        float*baseline=conv_call(0,dense,parent,packed,H,C);if(!baseline)abort();
        for(int mode=1;mode<3;mode++){
            float*out=conv_call(mode,dense,parent,packed,H,C);if(!out||memcmp(out,baseline,n*4)){fprintf(stderr,"FAIL conv\n");return 1;}_mm_free(out);
        }
        _mm_free(baseline);
        for(int round=0;round<9;round++)for(int j=0;j<3;j++){
            int mode=(round+j+run)%3;_mm_mfence();uint64_t t=now();float*out=conv_call(mode,dense,parent,packed,H,C);_mm_mfence();double ns=(double)(now()-t);
            if(!out)abort();sink=out[n/2];_mm_free(out);
            printf("conv3x3,%zu,P%zu,%zu,%zu,%s,%d,%d,1,%.3f,2,PASS\n",image,level,H,C,names[mode],run,round,ns);
        }
        fflush(stdout);fprintf(stderr,"PASS consumers image=%zu P%zu C=%zu\n",image,level,C);
        _mm_free(packed);free(weights);free(parent);free(dense);free(a);free(d);free(gold);
    }
    return sink==12345.0f;
}
