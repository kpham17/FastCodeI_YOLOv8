/* Production-only raw benchmark; CPU 2, 12 shapes, A/B/both, 15 warmed batches.
 * Allocation, initialization and correctness checks are outside timing. */
#define _GNU_SOURCE
#include "chunk_nhwc.h"
typedef void (*chunk_fn)(float*,float*,const float*,size_t,size_t);
static const struct {const char *name;chunk_fn fn;} candidates[]={{"chunk",chunk2_nhwc_f32}};
static const size_t candidate_count=1;
#include <dirent.h>
#include <immintrin.h>
#include <linux/perf_event.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
static volatile uint32_t sink;
static int fd=-1,rfd=-1;
static uint64_t now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return (uint64_t)t.tv_sec*1000000000ull+t.tv_nsec;}
static float *alloc(size_t n){float*p=NULL;if(posix_memalign((void**)&p,64,n*4))abort();return p;}
static void pin(void){cpu_set_t set;CPU_ZERO(&set);CPU_SET(2,&set);if(sched_setaffinity(0,sizeof(set),&set))abort();DIR*d=opendir("/proc/self/task");struct dirent*e;int n=0;while((e=readdir(d)))if(e->d_name[0]!='.')n++;closedir(d);if(n!=1||sched_getcpu()!=2)abort();fprintf(stderr,"pid=%ld cpu=2 socket=0 threads=1\n",(long)getpid());}
static void counters(void){struct perf_event_attr a;memset(&a,0,sizeof(a));a.size=sizeof(a);a.type=PERF_TYPE_HARDWARE;a.config=PERF_COUNT_HW_CPU_CYCLES;a.disabled=1;a.exclude_kernel=1;a.exclude_hv=1;a.read_format=PERF_FORMAT_GROUP|PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;fd=syscall(SYS_perf_event_open,&a,0,-1,-1,0);if(fd<0)return;a.config=PERF_COUNT_HW_REF_CPU_CYCLES;a.disabled=0;rfd=syscall(SYS_perf_event_open,&a,0,-1,fd,0);if(rfd<0){close(fd);fd=-1;}}
static double batch(chunk_fn f,float*a,float*b,const float*s,size_t p,size_t c,size_t loops,double*cy,double*ghz){
    if(fd>=0){ioctl(fd,PERF_EVENT_IOC_RESET,PERF_IOC_FLAG_GROUP);ioctl(fd,PERF_EVENT_IOC_ENABLE,PERF_IOC_FLAG_GROUP);}
    _mm_mfence();uint64_t t=now();
    for(size_t i=0;i<loops;i++){f(a,b,s,p,c);__asm__ __volatile__("":::"memory");}
    _mm_mfence();uint64_t elapsed=now()-t;*cy=*ghz=0;
    if(fd>=0){ioctl(fd,PERF_EVENT_IOC_DISABLE,PERF_IOC_FLAG_GROUP);uint64_t z[5];if(read(fd,z,sizeof(z))==sizeof(z)&&z[0]==2&&z[2]>=z[1]*.99){*cy=(double)z[3]/loops;*ghz=2.1*(double)z[3]/z[4];}}
    uint32_t v;memcpy(&v,(a?a:b)+p*c/2,4);sink^=v;return (double)elapsed/loops;
}
int main(int argc,char**argv){
    int run=argc==2?atoi(argv[1]):1;unsigned random=0xa76391u+run*17u;pin();counters();
    puts("image,level,H,W,C,mode,variant,run,round,loops,logical_bytes,footprint_bytes,ns,cycles,core_GHz,cpu,threads,validation");
    const size_t images[]={640,1024,2048};
    for(size_t index=0;index<36;index++){
        size_t k=(run%2==0)?35-index:index,image=images[k/12],level=(k%12)/3+2,mode=k%3;
        size_t h=image>>level,c=(size_t)16<<(level-2),p=h*h,n=p*c;
        float*s=alloc(2*n),*a=alloc(n),*b=alloc(n);
        for(size_t i=0;i<2*n;i++){uint32_t v=(uint32_t)i*2654435761u+0x7fc12345u;memcpy(s+i,&v,4);}memset(a,0,n*4);memset(b,0,n*4);
        float*x=mode==2?NULL:a,*y=mode==1?NULL:b;
        size_t loops[16];
        for(size_t v=0;v<candidate_count;v++){
            for(int j=0;j<3;j++)candidates[v].fn(x,y,s,p,c);
            for(size_t i=0;i<p;i++)if((x&&memcmp(x+i*c,s+i*2*c,c*4))||(y&&memcmp(y+i*c,s+(i*2+1)*c,c*4)))abort();
            double cy,ghz,ns=batch(candidates[v].fn,x,y,s,p,c,1,&cy,&ghz);loops[v]=(size_t)(1500000/ns)+1;if(loops[v]>65536)loops[v]=65536;
        }
        for(int round=0;round<15;round++){
            size_t order[16];for(size_t v=0;v<candidate_count;v++)order[v]=v;
            for(size_t v=candidate_count-1;v;v--){random=random*1664525u+1013904223u;size_t j=random%(v+1),t=order[v];order[v]=order[j];order[j]=t;}
            for(size_t pos=0;pos<candidate_count;pos++){
                size_t v=order[pos];candidates[v].fn(x,y,s,p,c);candidates[v].fn(x,y,s,p,c);
                double cy,ghz;
                (void)batch(candidates[v].fn,x,y,s,p,c,loops[v]*3,&cy,&ghz);
                double ns=batch(candidates[v].fn,x,y,s,p,c,loops[v],&cy,&ghz);
                if(sched_getcpu()!=2)abort();
                size_t outputs=mode==0?2:1;
                printf("%zu,P%zu,%zu,%zu,%zu,%s,%s,%d,%d,%zu,%zu,%zu,%.3f,%.3f,%.4f,2,1,PASS\n",image,level,h,h,c,mode==0?"both":mode==1?"A":"B",candidates[v].name,run,round,loops[v],outputs*n*8,(2+outputs)*n*4,ns,cy,ghz);
            }fflush(stdout);
        }
        fprintf(stderr,"PASS image=%zu P%zu C=%zu mode=%zu\n",image,level,c,mode);free(s);free(a);free(b);
    }
    if(fd>=0){close(rfd);close(fd);}return 0;
}
