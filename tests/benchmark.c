#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


#include "../include/silu.h"
#include "../include/mmm_4x48.h"

#define SILU_FLOPS 16
#define CONV_FLOPS 2
#define RUNS 10000

//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("rdtsc" : "=a"(lo), "=d"(hi));
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}


// HW -> height and width
void test_SILU(int HW, int C){
    unsigned long long t0, t1, sum;
    float *input, *output;

    const int count = HW * HW * C;
    const int totalFLOPS = SILU_FLOPS * count;
    input  = (float *)_mm_malloc(count * sizeof(float), 64);
    output = (float *)_mm_malloc(count * sizeof(float), 64);

    for (int i = 0; i != count; ++i){
        input[i] = ((float) rand())/ ((float) RAND_MAX);
    }

    printf("SILU: \t%d \t %d\t %d\t %d\t", HW, HW, C, totalFLOPS);
    
    sum = 0; 
    for (int runs = 0; runs != RUNS; ++runs){
        t0 = rdtsc();
        taylor_SILU(input, HW, HW, C, output);
        t1 = rdtsc();
        sum += (t1 - t0);
    }
    volatile float sink = output[1];
    (void) sink;
    
    printf(" %lf\t %lf\n", (double)(sum/(RUNS)), ((double)totalFLOPS)/((double)(sum/(RUNS))));

    free(input);
    // free(output);
}

// HW -> height and width
void test_MMM(int m, int k, int n){
    unsigned long long t0, t1, sum;
    float *a, *b, *c;

    int totalFLOPS = 2 * m * n * k;
    a = (float *)_mm_malloc((size_t)m * k * sizeof(float), 64);
    b = (float *)_mm_malloc((size_t)k * n * sizeof(float), 64);
    c = (float *)_mm_malloc((size_t)m * n * sizeof(float), 64);
     //initialize A
    for (int i = 0; i != k * m; ++i){
        a[i] = ((double) rand())/ ((double) RAND_MAX);
    }
    //initialize B
    for (int i = 0; i != k * n; ++i){
        b[i] = ((double) rand())/ ((double) RAND_MAX);
    }
    //initialize C
    for (int i = 0; i != m * n; ++i){
        c[i] = 0.0;
    }
    printf("MMM: \t%d \t %d\t %d\t  %d\t", m, n, k, totalFLOPS);
    
    sum = 0; 
    for (int runs = 0; runs != RUNS; ++runs){
        t0 = rdtsc();
        kernel_4x48(k, a, b, c, n);
        t1 = rdtsc();
        sum += (t1 - t0);
    }
    volatile float sink = c[0];
    (void) sink;

    printf(" %lf\t %lf\n", (double)(sum/(RUNS)), ((double)totalFLOPS)/(((double)  sum/(RUNS))));

    free(a);
    free(b);
    free(c);
}

int main(){
    // SILU
    test_SILU(32, 64);
    // Since is 4x48 kernel, m = 4, n = 48, k is the only thing we can change
    test_MMM(4, 256, 48);
    
    return 0;
}


