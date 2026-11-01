#define _GNU_SOURCE
#include "universal_concat.h"
#include "fused_upconcat.h"
#include "residual_add.h"
#include "universal_concat_workloads.h"
#include "memory_kernel_fixtures.h"
#include "memory_kernel_timer.h"
#include <math.h>
typedef struct {
    int kernel,stage,view,inplace;size_t n,h,w,c,cs,k,channels[8];
    float *a[2][8],*b[2],*d[2];
    universal_concat_view cv[8];
    size_t an[8],bn,dn,input_bytes,footprint;char name[80];
} fixture;
static volatile uint32_t sink;
static memory_counter counter;
static int run_id,rounds=5;
static void require(int ok) {if(!ok){fputs("benchmark validation failed\n",stderr);exit(1);}}
/* Every timed call performs only the selected materializing kernel. */
__attribute__((noinline)) static void invoke(fixture *f) {
    int status;
    if(f->kernel==0) status=universal_concat_f32(f->cv,f->k,f->d[1],f->dn,f->n,f->h,f->w);
    else if(f->kernel==1) status=fused_upconcat_f32(f->a[1][0],f->an[0],f->b[1],f->bn,
        f->d[1],f->dn,f->h,f->w,f->c,f->cs);
    else status=residual_add(f->d[1],f->dn,f->inplace?f->d[1]:f->a[1][0],f->dn,
        f->b[1]+(f->view?f->dn:0),f->dn,f->h,f->w,f->c);
    require(status==0);
    __asm__ volatile("" ::: "memory");
}
static void reset(fixture *f) {
    if(f->inplace) memcpy(f->d[1],f->a[1][0],f->dn*4);
}
/* Untimed scalar logical oracles; no old production kernel is linked. */
static void reference(fixture *f,float *gold) {
    if(f->kernel==0) {
        for(size_t p=0;p<f->n*f->h*f->w;++p) {
            size_t offset=0;
            for(size_t i=0;i<f->k;++i) {
                for(size_t c=0;c<f->channels[i];++c)
                    memcpy(gold+p*f->c+offset+c,f->a[0][i]+p*f->channels[i]+c,4);
                offset+=f->channels[i];
            }
        }
    } else if(f->kernel==1) {
        for(size_t y=0;y<2*f->h;++y) for(size_t x=0;x<2*f->w;++x)
            for(size_t c=0;c<f->c+f->cs;++c) {
                const float *src=c<f->c?f->a[0][0]+((y/2)*f->w+x/2)*f->c+c:
                    f->b[0]+(y*2*f->w+x)*f->cs+c-f->c;
                memcpy(gold+(y*2*f->w+x)*(f->c+f->cs)+c,src,4);
            }
    } else {
        const float *b=f->b[0]+(f->view?f->c:0);
        for(size_t p=0;p<f->h*f->w;++p) for(size_t c=0;c<f->c;++c)
            gold[p*f->c+c]=f->a[0][0][p*f->c+c]+b[p*f->c*(f->view?2:1)+c];
    }
}

typedef struct { double ns,tsc,cycles,instructions; } sample;
static sample measure(fixture *f,size_t calls) {
    memory_counter ca=memory_counter_read(counter);
    uint64_t ta=memory_tsc(),na=memory_now();
    for(size_t i=0;i<calls;++i) invoke(f);
    uint64_t nb=memory_now(),tb=memory_tsc();
    memory_counter cb=memory_counter_read(counter);
    uint32_t bits;memcpy(&bits,f->d[1]+f->dn/2,4);sink^=bits;
    sample s={(double)(nb-na)/calls,(double)(tb-ta)/calls,memory_cycles(ca,cb)/calls,-1};
    if(s.cycles>=0) s.instructions=(double)(cb.instructions-ca.instructions)/calls;
    return s;
}
/* Fixed page alignment makes relative cache/alias placement reproducible.
 * It is not a stronger alignment requirement on the production kernels. */
static float *bench_alloc(size_t n) {
    float *p=NULL;if(posix_memalign((void **)&p,4096,n*4)) abort();return p;
}
static void make_fixture(fixture *f) {
    size_t p=f->n*f->h*f->w;
    if(f->kernel==0) {
        f->c=0;for(size_t i=0;i<f->k;++i) {f->an[i]=p*f->channels[i];f->c+=f->channels[i];}
        f->dn=p*f->c;f->input_bytes=f->dn*4;
    } else if(f->kernel==1) {
        f->k=1;f->an[0]=p*f->c;f->bn=4*p*f->cs;f->dn=4*p*(f->c+f->cs);
        f->input_bytes=(f->an[0]+f->bn)*4;
    } else {
        f->k=1;f->an[0]=p*f->c;f->bn=p*f->c*(f->view?2:1);f->dn=p*f->c;
        f->input_bytes=f->dn*8;
    }
    f->footprint=(f->dn+f->bn+(f->inplace?0:f->an[0]))*4;
    if(f->kernel==0) f->footprint=f->dn*8;
    for(int l=0;l<2;++l) {
        for(size_t i=0;i<f->k;++i) f->a[l][i]=bench_alloc(f->an[i]);
        if(f->bn) f->b[l]=bench_alloc(f->bn);
        f->d[l]=bench_alloc(f->dn);memset(f->d[l],0,f->dn*4);
    }
    for(size_t i=0;i<f->k;++i) {
        memory_finite(f->a[0][i],f->an[i],(int)i);
        memory_pack(f->a[1][i],f->a[0][i],f->n,f->h,f->w,f->kernel==0?f->channels[i]:f->c);
        if(f->kernel==0) {
            f->cv[i]=(universal_concat_view){f->a[1][i],f->channels[i],f->an[i]};
        }
    }
    if(f->bn) {
        memory_finite(f->b[0],f->bn,7);
        memory_pack(f->b[1],f->b[0],1,f->h*(f->kernel==1?2:1),f->w*(f->kernel==1?2:1),
                  f->kernel==1?f->cs:f->c*(f->view?2:1));
    }
}
static void free_fixture(fixture *f) {
    for(int l=0;l<2;++l) {for(size_t i=0;i<f->k;++i) free(f->a[l][i]);free(f->b[l]);free(f->d[l]);}
}
static void print_metric(double x) {if(x<0) fputs("NA",stdout);else printf("%.6f",x);}
static void emit(fixture *f,int round,size_t calls,sample s) {
    const char *kernel=f->kernel==0?"concat":f->kernel==1?"upconcat":"add";
    printf("%d,%d,%s,%s,%d,%zu,%zu,%.6f,",run_id,round,kernel,f->name,sched_getcpu(),calls,f->footprint,s.ns);
    print_metric(s.cycles);putchar(',');
    print_metric(s.cycles>0?(f->dn/16.0)/s.cycles:-1);putchar(',');
    print_metric(s.instructions);
    const float *a=f->inplace?f->d[1]:f->a[1][0];
    const float *b=f->kernel==0?f->a[1][1]:f->b[1]+(f->kernel==2 && f->view?f->dn:0);
    printf(",%zu,%zu,%zu,PASS\n",(size_t)((uintptr_t)a%4096),
        (size_t)((uintptr_t)b%4096),(size_t)((uintptr_t)f->d[1]%4096));
}
static void run_case(fixture *f) {
    const char *filter=getenv("MEMORY_BENCH_FILTER");
    if(filter && !strstr(f->name,filter)) return;
    const char *kernel=getenv("MEMORY_BENCH_KERNEL");
    if(kernel && strcmp(kernel,f->kernel==0?"concat":f->kernel==1?"upconcat":"add")) return;
    make_fixture(f);
    float *gold=memory_alloc(f->dn),*back=memory_alloc(f->dn);
    reference(f,gold);reset(f);invoke(f);
    memory_unpack(back,f->d[1],f->n,f->h*(f->kernel==1?2:1),f->w*(f->kernel==1?2:1),f->kernel==1?f->c+f->cs:f->c);
    require(!memcmp(back,gold,f->dn*4));
    size_t calls=1;sample probe;
    do {
        reset(f);probe=measure(f,calls);
        if(probe.ns*calls>=200000.0 || calls>=1000000) break;
        calls*=2;
    } while(1);
    calls=(size_t)(2000000.0/fmax(probe.ns,1.0))+1;
    if(calls>1000000) calls=1000000;
    for(int r=0;r<rounds;++r) {
        uint64_t start=memory_now();
        do {for(size_t warm=0;warm<calls;++warm) invoke(f);} while(memory_now()-start<10000000);
        reset(f);sample s=measure(f,calls);emit(f,r,calls,s);
    }
    free(gold);free(back);free_fixture(f);fflush(stdout);
}
int main(int argc,char **argv) {
    if(argc!=3 || strcmp(argv[1],"warm")) {
        fprintf(stderr,"Usage: %s warm RUN_ID\n",argv[0]);return 2;
    }
    run_id=atoi(argv[2]);counter=memory_counter_open();
    puts("run,round,kernel,case,cpu,calls,tensor_footprint_B,ns_per_call,core_cycles_per_call,output_64B_per_core_cycle,instructions_per_call,a_mod4096,b_mod4096,out_mod4096,correctness");
    for(size_t im=0;im<3;++im) {
        size_t image=(size_t[]){640,1024,2048}[im];
        for(size_t i=0;i<UC_SITE_COUNT;++i) {
            const uc_workload *x=&uc_workloads[i];fixture f={.kernel=0,.n=1,.h=image/x->scale,.w=image/x->scale,.k=x->count};
            memcpy(f.channels,x->channels,sizeof(x->channels));snprintf(f.name,sizeof(f.name),"image%zu_site%d",image,x->site);run_case(&f);
        }
        for(int stage=0;stage<2;++stage) {
            fixture f={.kernel=1,.n=1,.stage=stage,.h=image/(stage?16:32),.w=image/(stage?16:32),.c=stage?128:256,.cs=stage?64:128};
            snprintf(f.name,sizeof(f.name),"image%zu_%c",image,'A'+stage);run_case(&f);
        }
        for(size_t level=2;level<=5;++level) for(int view=0;view<2;++view) for(int inplace=0;inplace<2;++inplace) {
            fixture f={.kernel=2,.n=1,.h=image>>level,.w=image>>level,.c=(size_t)16<<(level-2),.view=view,.inplace=inplace};
            snprintf(f.name,sizeof(f.name),"image%zu_P%zu_%s_%s",image,level,view?"view":"dense",inplace?"in":"out");run_case(&f);
        }
    }
    fixture tiny={.kernel=0,.n=1,.h=8,.w=8,.k=2,.channels={16,16}};strcpy(tiny.name,"L1_tiny");run_case(&tiny);
    for(int stage=0;stage<2;++stage) {
        fixture f={.kernel=1,.stage=stage,.n=1,.h=stage?2:1,.w=stage?3:2,.c=stage?128:256,.cs=stage?64:128};
        snprintf(f.name,sizeof(f.name),"small_%c",'A'+stage);run_case(&f);
        f.h=1;f.w=1;snprintf(f.name,sizeof(f.name),"L1_1x1_%c",'A'+stage);run_case(&f);
        f.h=3;f.w=7;snprintf(f.name,sizeof(f.name),"rectangle_%c",'A'+stage);run_case(&f);
    }
    for(int view=0;view<2;++view) for(int inplace=0;inplace<2;++inplace) {
        fixture f={.kernel=2,.n=1,.h=8,.w=8,.c=16,.view=view,.inplace=inplace};
        snprintf(f.name,sizeof(f.name),"L1_%s_%s",view?"view":"dense",inplace?"in":"out");run_case(&f);
    }
    fixture min_concat={.kernel=0,.n=1,.h=1,.w=8,.k=2,.channels={16,16}};
    strcpy(min_concat.name,"L1_concat_2KiB");run_case(&min_concat);
    fixture min_add={.kernel=2,.n=1,.h=1,.w=8,.c=16};
    strcpy(min_add.name,"L1_add_1536B");run_case(&min_add);
    fixture min_up={.kernel=1,.n=1,.h=1,.w=1,.c=16,.cs=16};
    strcpy(min_up.name,"L1_fused_832B");run_case(&min_up);
    fprintf(stderr,"observable checksum %u\n",sink);return 0;
}
