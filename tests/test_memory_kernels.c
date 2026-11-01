#define _GNU_SOURCE
#include "universal_concat.h"
#include "fused_upconcat.h"
#include "residual_add.h"
#include "universal_concat_workloads.h"
#include "memory_kernel_fixtures.h"
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#define CHECK(x) do { if(!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);exit(1); } } while(0)
static size_t cases;
typedef struct { float *p,*base; size_t n,bytes; int guard; } buffer;
static buffer buf(size_t n,int guard,int unaligned) {
    buffer b={0}; b.n=n;b.guard=guard;
    if(guard) {
        size_t page=(size_t)sysconf(_SC_PAGESIZE);
        b.bytes=((n*4+page-1)/page+2)*page;
        b.base=mmap(NULL,b.bytes,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        CHECK(b.base!=MAP_FAILED);
        CHECK(!mprotect(b.base,page,PROT_NONE));
        CHECK(!mprotect((char*)b.base+b.bytes-page,page,PROT_NONE));
        b.p=(float*)((char*)b.base+b.bytes-page)-n;
    } else {
        b.base=memory_alloc(n+64); b.p=b.base+16+(unaligned?1:0);
        memset(b.base,0xa5,(n+64)*4);
    }
    return b;
}
static void release(buffer b) {
    if(b.guard) CHECK(!munmap(b.base,b.bytes));
    else {
        uint32_t x,y;memcpy(&x,b.p-1,4);memcpy(&y,b.p+b.n,4);
        CHECK(x==0xa5a5a5a5u && y==0xa5a5a5a5u);free(b.base);
    }
}
static void sanity(void) {
    float a[256],b[256],out[256];
    for(size_t p=0;p<4;++p) for(size_t c=0;c<64;++c) a[p*64+c]=(float)(1000*p+c);
    memory_pack(b,a,1,1,4,64);
    CHECK(b[0]==0 && b[16]==1000 && b[64]==16 && b[128]==32 && b[255]==3063);
    CHECK(tensor_index(16,0,0,1,4)==64);
    CHECK(tensor_index(63,0,3,1,4)==255);
    memory_unpack(out,b,1,1,4,64);CHECK(!memcmp(a,out,sizeof(a)));
    /* Two block-aligned halves of one cv1 parent remain dense per image. */
    float parent[3*5*64],packed[3*5*64],joined[3*5*64],back[3*5*64];
    memory_bits(parent,3*5*64,19);memory_pack(packed,parent,1,3,5,64);
    universal_concat_view halves[]={{packed,32,3*5*64},{packed+32*3*5,32,32*3*5}};
    CHECK(!universal_concat_f32(halves,2,joined,3*5*64,1,3,5));
    memory_unpack(back,joined,1,3,5,64);CHECK(!memcmp(back,parent,sizeof(parent)));
    ++cases;
}
/* Catches concatenating complete batched inputs instead of per-image blocks,
 * NHWC-like pixel order, lane changes, and partial writes past capacity. */
static void concat_case(size_t n,size_t h,size_t w,const size_t *cs,size_t k,int guard,int unaligned) {
    universal_concat_view v[8];buffer a[8],b[8];
    size_t co=0,pixels=n*h*w;
    for(size_t i=0;i<k;++i) {
        co+=cs[i];a[i]=buf(pixels*cs[i],guard,unaligned);b[i]=buf(pixels*cs[i],guard,unaligned);
        memory_bits(a[i].p,a[i].n,(uint32_t)i+177);
        memory_pack(b[i].p,a[i].p,n,h,w,cs[i]);
        v[i]=(universal_concat_view){b[i].p,cs[i],b[i].n};
    }
    buffer out=buf(pixels*co,guard,unaligned);
    float *unpacked=memory_alloc(out.n),*oracle=memory_alloc(out.n);
    for(size_t p=0;p<pixels;++p) {
        size_t offset=0;
        for(size_t i=0;i<k;++i) {
            memcpy(oracle+p*co+offset,a[i].p+p*cs[i],cs[i]*4);offset+=cs[i];
        }
    }
    {
        int status=universal_concat_f32(v,k,out.p,out.n,n,h,w);
        CHECK(status==0);memory_unpack(unpacked,out.p,n,h,w,co);
        CHECK(!memcmp(unpacked,oracle,out.n*4));
    }
    CHECK(universal_concat_f32(v,k,out.p,out.n,n,h,w)==0);
    for(size_t i=0;i<k;++i) {
        memory_unpack(unpacked,b[i].p,n,h,w,cs[i]);CHECK(!memcmp(a[i].p,unpacked,a[i].n*4));
        release(a[i]);release(b[i]);
    }
    release(out);free(unpacked);free(oracle);++cases;
}
static void up_case(size_t h,size_t w,size_t cu,size_t cs,int guard,int unaligned) {
    size_t nu=h*w*cu,ns=4*h*w*cs,no=4*h*w*(cu+cs);
    float *u=memory_alloc(nu),*s=memory_alloc(ns),*oracle=memory_alloc(no),*back=memory_alloc(no);
    buffer ub=buf(nu,guard,unaligned),sb=buf(ns,guard,unaligned),d=buf(no,guard,unaligned);
    memory_bits(u,nu,113);memory_bits(s,ns,2523);
    memory_pack(ub.p,u,1,h,w,cu);memory_pack(sb.p,s,1,2*h,2*w,cs);
    for(size_t y=0;y<2*h;++y) for(size_t x=0;x<2*w;++x) for(size_t c=0;c<cu+cs;++c) {
        const float *src=c<cu?u+((y/2)*w+x/2)*cu+c:s+(y*2*w+x)*cs+(c-cu);
        memcpy(oracle+(y*2*w+x)*(cu+cs)+c,src,4);
    }
    {
        int status=fused_upconcat_f32(ub.p,nu,sb.p,ns,d.p,no,h,w,cu,cs);
        CHECK(status==0);memory_unpack(back,d.p,1,2*h,2*w,cu+cs);CHECK(!memcmp(back,oracle,no*4));
    }
    if(cu==256 && cs==128) CHECK(!fused_upconcat_a(ub.p,nu,sb.p,ns,d.p,no,h,w));
    if(cu==128 && cs==64) CHECK(!fused_upconcat_b(ub.p,nu,sb.p,ns,d.p,no,h,w));
    memory_unpack(back,ub.p,1,h,w,cu);CHECK(!memcmp(back,u,nu*4));
    memory_unpack(back,sb.p,1,2*h,2*w,cs);CHECK(!memcmp(back,s,ns*4));
    release(ub);release(sb);release(d);free(u);free(s);free(oracle);free(back);++cases;
}
static int equal_add(const float *a,const float *b,size_t n) {
    for(size_t i=0;i<n;++i) if(memcmp(a+i,b+i,4) && !(isnan(a[i])&&isnan(b[i]))) return 0;
    return 1;
}
/* Catches cv1+C instead of cv1+C*H*W, input corruption and lost in-place
 * support. Oracle computes one FP32 addition per logical element. NaN result
 * payload selection is not specified; finite/zero/inf results are bitwise. */
static void add_case(size_t h,size_t w,size_t c,int view,int guard,int unaligned,int special) {
    size_t n=h*w*c,parent_n=n*(view?2:1);
    float *a=memory_alloc(n),*parent=memory_alloc(parent_n),*oracle=memory_alloc(n),*back=memory_alloc(n);
    buffer ab=buf(n,guard,unaligned),pb=buf(parent_n,guard,unaligned),d=buf(n,guard,unaligned);
    memory_finite(a,n,0);memory_finite(parent,parent_n,7);
    if(special) {memory_bits(a,n,999);memory_bits(parent,parent_n,888);}
    const float *b=parent+(view?c:0);size_t pitch=c*(view?2:1);
    for(size_t p=0;p<h*w;++p) for(size_t j=0;j<c;++j) oracle[p*c+j]=a[p*c+j]+b[p*pitch+j];
    memory_pack(pb.p,parent,1,h,w,pitch);
    const float *shortcut=pb.p+(view?n:0);
    {
        for(int inplace=0;inplace<2;++inplace) {
            memory_pack(ab.p,a,1,h,w,c);float *dst=inplace?ab.p:d.p;
            int status=residual_add(dst,n,ab.p,n,shortcut,n,h,w,c);
            CHECK(status==0);memory_unpack(back,dst,1,h,w,c);CHECK(equal_add(back,oracle,n));
            if(!inplace) {memory_unpack(back,ab.p,1,h,w,c);CHECK(!memcmp(back,a,n*4));}
        }
    }
    memory_pack(ab.p,a,1,h,w,c);
    CHECK(!residual_add(d.p,n,ab.p,n,shortcut,n,h,w,c));
    float *parent_back=memory_alloc(parent_n);memory_unpack(parent_back,pb.p,1,h,w,pitch);
    CHECK(!memcmp(parent_back,parent,parent_n*4));free(parent_back);
    release(ab);release(pb);release(d);free(a);free(parent);free(oracle);free(back);++cases;
}
static void validation(void) {
    float a[256]={0},b[256]={0},d[1024];memset(d,0xa5,sizeof(d));
    unsigned char snapshot[sizeof(d)];memcpy(snapshot,d,sizeof(d));
    universal_concat_view v[2]={{a,16,256},{b,16,256}};
#define BAD_CONCAT(expr,code) CHECK((expr)==(code));CHECK(!memcmp(d,snapshot,sizeof(d)))
    BAD_CONCAT(universal_concat_f32(v,2,d,1024,1,0,1),MEMORY_BAD_ARGUMENT);
    BAD_CONCAT(universal_concat_f32(NULL,2,d,1024,1,1,1),MEMORY_BAD_ARGUMENT);
    BAD_CONCAT(universal_concat_f32(v,0,d,1024,1,1,1),MEMORY_BAD_ARGUMENT);
    v[1].channels=17;BAD_CONCAT(universal_concat_f32(v,2,d,1024,1,1,1),MEMORY_BAD_ARGUMENT);
    v[1].channels=16;v[1].capacity=15;
    BAD_CONCAT(universal_concat_f32(v,2,d,1024,1,1,1),MEMORY_CAPACITY);
    v[1].capacity=256;BAD_CONCAT(universal_concat_f32(v,2,d,31,1,1,1),MEMORY_CAPACITY);
    BAD_CONCAT(universal_concat_f32(v,2,d,1024,SIZE_MAX,2,1),MEMORY_OVERFLOW);
    BAD_CONCAT(universal_concat_f32(v,SIZE_MAX,d,1024,1,1,1),MEMORY_OVERFLOW);
    CHECK(universal_concat_f32(v,2,a,256,1,1,1)==MEMORY_OVERLAP);
    v[1].data=(float *)(uintptr_t)(UINTPTR_MAX-15);
    BAD_CONCAT(universal_concat_f32(v,2,d,1024,1,1,1),MEMORY_OVERFLOW);
    v[1].data=b;
    union { universal_concat_view meta[8];float floats[64]; } alias;
    alias.meta[0]=v[0];CHECK(universal_concat_f32(alias.meta,1,alias.floats,64,1,1,1)==MEMORY_OVERLAP);
    v[1].data=a;CHECK(!universal_concat_f32(v,2,d,1024,1,1,1));
    memcpy(d,snapshot,sizeof(d));
    CHECK(fused_upconcat_f32(a,256,b,256,d,1024,1,1,17,16)==MEMORY_BAD_ARGUMENT);
    CHECK(fused_upconcat_f32(a,15,b,256,d,1024,1,1,16,16)==MEMORY_CAPACITY);
    CHECK(fused_upconcat_f32(a,256,b,256,d,127,1,1,16,16)==MEMORY_CAPACITY);
    CHECK(fused_upconcat_f32(a,256,b,256,a,256,1,1,16,16)==MEMORY_OVERLAP);
    CHECK(fused_upconcat_f32(a,256,b,256,d,1024,SIZE_MAX,1,16,16)==MEMORY_OVERFLOW);
    CHECK(fused_upconcat_f32(NULL,256,b,256,d,1024,1,1,16,16)==MEMORY_BAD_ARGUMENT);
    CHECK(!memcmp(d,snapshot,sizeof(d)));
    CHECK(residual_add(d,1024,a,256,b,256,1,1,17)==MEMORY_BAD_ARGUMENT);
    CHECK(residual_add(d,1024,a,15,b,256,1,1,16)==MEMORY_CAPACITY);
    CHECK(residual_add(d,15,a,256,b,256,1,1,16)==MEMORY_CAPACITY);
    CHECK(residual_add(a+1,255,a,256,b,256,1,1,16)==MEMORY_OVERLAP);
    CHECK(residual_add(b,256,a,256,b,256,1,1,16)==MEMORY_OVERLAP);
    CHECK(residual_add(d,1024,a,256,b,256,SIZE_MAX,2,16)==MEMORY_OVERFLOW);
    CHECK(residual_add(d,1024,NULL,256,b,256,1,1,16)==MEMORY_BAD_ARGUMENT);
    CHECK(residual_add(d,1024,a,256,b,256,0,1,16)==MEMORY_BAD_ARGUMENT);
    CHECK(!memcmp(d,snapshot,sizeof(d)));++cases;
}
int main(void) {
    CHECK(!fesetround(FE_TONEAREST));sanity();
    const size_t cs[]={16,32,48,64,16};
    concat_case(2,3,5,cs,5,0,1);concat_case(1,1,1,cs,2,1,0);
    concat_case(3,1,7,cs,4,1,0);concat_case(1,8,8,cs,2,0,0);
    for(size_t j=0;j<3;++j) {
        size_t image=(size_t[]){640,1024,2048}[j];
        for(size_t i=0;i<UC_SITE_COUNT;++i) {
            const uc_workload *x=&uc_workloads[i];size_t side=image/x->scale;
            concat_case(1,side,side,x->channels,x->count,0,0);
        }
        up_case(image/32,image/32,256,128,0,0);up_case(image/16,image/16,128,64,0,0);
        for(size_t level=2;level<=5;++level) for(int view=0;view<2;++view)
            add_case(image>>level,image>>level,(size_t)16<<(level-2),view,0,0,0);
    }
    for(size_t h=1;h<=3;++h) for(size_t w=1;w<=7;w+=2) {
        up_case(h,w,256,128,1,0);up_case(h,w,128,64,0,1);
        add_case(h,w,16,1,1,0,0);add_case(h,w,48,0,0,1,1);
    }
    up_case(3,1,48,32,1,0);up_case(2,1,16,16,0,1);up_case(4,1,256,128,0,1);
    validation();printf("PASS: %zu memory-kernel cases, logical/bitwise oracles, guards, validation\n",cases);
    return 0;
}
