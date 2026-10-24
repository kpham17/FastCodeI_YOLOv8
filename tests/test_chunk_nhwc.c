#define _GNU_SOURCE
#include "chunk_nhwc.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(__unix__)
#include <sys/mman.h>
#include <unistd.h>
#endif

static uint32_t state=0x6215b7a3u;
static uint32_t next(void){state^=state<<13;state^=state>>17;state^=state<<5;return state;}
static float *allocate(size_t n){float*p=NULL;if(posix_memalign((void**)&p,64,n*4))abort();return p;}
static void check(size_t P,size_t C,size_t off,int mode) {
    const size_t guard=32,n=P*C,total=2*n;
    float *s=allocate(total+off+2*guard+64),*a=allocate(n+off+2*guard+64),*b=allocate(n+off+2*guard+64);
    float *in=s+guard+off,*x=a+guard+off,*y=b+guard+(off*3)%16;
    unsigned char *snapshot=malloc((total+off+2*guard+64)*4);
    memset(a,0xa5,(n+off+2*guard+64)*4);memset(b,0xa5,(n+off+2*guard+64)*4);
    for(size_t i=0;i<total+off+2*guard+64;i++){uint32_t v=next();memcpy(s+i,&v,4);}
    if(total>8){const uint32_t special[]={0,0x80000000u,0x7f800000u,0xff800000u,0x7fc12345u,0x7f812345u,1,0xffffffffu};memcpy(in,special,sizeof(special));}
    memcpy(snapshot,s,(total+off+2*guard+64)*4);
    chunk2_nhwc_f32(mode==2?NULL:x,mode==1?NULL:y,in,P,C);
    if(memcmp(snapshot,s,(total+off+2*guard+64)*4))abort();
    for(int half=0;half<2;half++) {
        unsigned char *base=(unsigned char*)(half?b:a),*out=(unsigned char*)(half?y:x);
        size_t all=(n+off+2*guard+64)*4;
        for(size_t i=0;i<all;i++) {
            int active=!(mode==(half?1:2)) && base+i>=out && base+i<out+4*n;
            unsigned char expected=0xa5;
            if(active){size_t j=(size_t)(base+i-out),f=j/4;expected=((unsigned char*)in)[((f/C)*2*C+half*C+f%C)*4+j%4];}
            if(base[i]!=expected){fprintf(stderr,"FAIL P=%zu C=%zu off=%zu mode=%d half=%d byte=%zu\n",P,C,off,mode,half,i);exit(1);}
        }
    }
    free(snapshot);free(s);free(a);free(b);
}
static void guard_page_check(void) {
#if defined(__unix__)
    size_t page=(size_t)sysconf(_SC_PAGESIZE);
    for(size_t C=1;C<=129;C++) {
        size_t P=3,n=P*C;
        unsigned char *s=mmap(NULL,2*page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        unsigned char *a=mmap(NULL,2*page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        unsigned char *b=mmap(NULL,2*page,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
        if(s==MAP_FAILED||a==MAP_FAILED||b==MAP_FAILED)abort();
        if(mprotect(s+page,page,PROT_NONE)||mprotect(a+page,page,PROT_NONE)||mprotect(b+page,page,PROT_NONE))abort();
        float*in=(float*)(s+page-8*n),*x=(float*)(a+page-4*n),*y=(float*)(b+page-4*n);
        for(size_t i=0;i<2*n;i++){uint32_t v=next();memcpy(in+i,&v,4);}
        for(int mode=0;mode<3;mode++) {
            chunk2_nhwc_f32(mode==2?NULL:x,mode==1?NULL:y,in,P,C);
            for(size_t p=0;p<P;p++) {
                if(mode!=2 && memcmp(x+p*C,in+2*p*C,C*4))abort();
                if(mode!=1 && memcmp(y+p*C,in+(2*p+1)*C,C*4))abort();
            }
        }
        munmap(s,2*page);munmap(a,2*page);munmap(b,2*page);
    }
#endif
}
int main(void) {
    chunk2_nhwc_f32(NULL,NULL,NULL,0,128);chunk2_nhwc_f32(NULL,NULL,NULL,128,0);
    chunk2_nhwc_f32(NULL,NULL,NULL,8,16);
    size_t count=0;
    const size_t widths[]={1,2,3,7,8,9,15,16,17,31,32,33,63,64,65,127,128,129,256};
    const size_t pixels[]={0,1,2,3,7,8,9,15,16,17,31,65};
    for(size_t c=0;c<sizeof(widths)/sizeof(*widths);c++)for(size_t p=0;p<sizeof(pixels)/sizeof(*pixels);p++)for(size_t off=0;off<16;off++)for(int mode=0;mode<3;mode++){check(pixels[p],widths[c],off,mode);count++;}
    const size_t images[]={640,1024,2048};
    for(size_t i=0;i<3;i++)for(size_t level=2;level<=5;level++)for(int mode=0;mode<3;mode++){size_t h=images[i]>>level;check(h*h,(size_t)16<<(level-2),0,mode);count++;}
    guard_page_check();printf("PASS %zu split/half/unaligned cases plus 387 guard-page cases; inputs unchanged and bitwise outputs\n",count);
    return 0;
}
