#define _GNU_SOURCE
#include <stdlib.h>
#include "memory_kernel_timer.h"
/* Calibration only: cached 64-byte stores, eight per inner iteration.
 * This is not a concat, upsample, or residual kernel. */
__attribute__((target("avx512f"),noinline))
static void store_ring(float *p,size_t rounds) {
    const char *end=(const char *)p+16384;
    __asm__ volatile(
        "vxorps %%zmm0,%%zmm0,%%zmm0\n\t"
        "1: mov %[base],%%rax\n\t2: "
        "vmovaps %%zmm0,0(%%rax)\n\tvmovaps %%zmm0,64(%%rax)\n\t"
        "vmovaps %%zmm0,128(%%rax)\n\tvmovaps %%zmm0,192(%%rax)\n\t"
        "vmovaps %%zmm0,256(%%rax)\n\tvmovaps %%zmm0,320(%%rax)\n\t"
        "vmovaps %%zmm0,384(%%rax)\n\tvmovaps %%zmm0,448(%%rax)\n\t"
        "add $512,%%rax\n\tcmp %[end],%%rax\n\tjb 2b\n\t"
        "dec %[rounds]\n\tjnz 1b\n\t"
        :[rounds]"+&r"(rounds):[base]"r"(p),[end]"r"(end)
        :"rax","xmm0","cc","memory");
}
int main(int argc,char **argv) {
    int run=argc>1?atoi(argv[1]):0;
    if(!__builtin_cpu_supports("avx512f")) return 2;
    memory_counter counter=memory_counter_open();float *p=NULL;
    if(counter.fd<0 || posix_memalign((void **)&p,64,16384)) return 2;
    memset(p,0xa5,16384);
    puts("run,round,width_B,ring_B,vector_stores,core_cycles,stores_per_cycle,ns");
    for(int r=0;r<21;++r) {
        uint64_t start=memory_now();
        do {store_ring(p,1024);} while(memory_now()-start<10000000);
        const size_t rings=20000,stores=rings*16384/64;
        memory_counter ca=memory_counter_read(counter);uint64_t na=memory_now();
        store_ring(p,rings);
        uint64_t nb=memory_now();memory_counter cb=memory_counter_read(counter);
        double cycles=memory_cycles(ca,cb);if(cycles<=0) return 2;
        for(size_t i=0;i<4096;++i) {uint32_t bits;memcpy(&bits,p+i,4);if(bits) abort();}
        printf("%d,%d,64,16384,%zu,%.0f,%.6f,%llu\n",run,r,stores,cycles,stores/cycles,(unsigned long long)(nb-na));
    }
    free(p);return 0;
}
