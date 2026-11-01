#ifndef MEMORY_KERNELS_TIMER_H
#define MEMORY_KERNELS_TIMER_H
#include <errno.h>
#include <linux/perf_event.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#include <x86intrin.h>
typedef struct { int fd;uint64_t cycles,instructions,enabled,running; } memory_counter;
static uint64_t memory_now(void) {
    struct timespec t;if(clock_gettime(CLOCK_MONOTONIC_RAW,&t)) abort();
    return (uint64_t)t.tv_sec*1000000000ull+(uint64_t)t.tv_nsec;
}
static inline uint64_t memory_tsc(void) {
    unsigned aux;_mm_lfence();uint64_t t=__rdtscp(&aux);_mm_lfence();return t;
}
static memory_counter memory_counter_open(void) {
    memory_counter c={.fd=-1};struct perf_event_attr a;
    memset(&a,0,sizeof(a));a.type=PERF_TYPE_HARDWARE;a.size=sizeof(a);
    a.config=PERF_COUNT_HW_CPU_CYCLES;a.exclude_kernel=1;a.exclude_hv=1;
    a.read_format=PERF_FORMAT_GROUP|PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
    c.fd=(int)syscall(__NR_perf_event_open,&a,0,-1,-1,0);
    if(c.fd<0) {fprintf(stderr,"perf cycles unavailable: %s\n",strerror(errno));return c;}
    a.config=PERF_COUNT_HW_INSTRUCTIONS;
    int ins=(int)syscall(__NR_perf_event_open,&a,0,-1,c.fd,0);
    if(ins<0) {fprintf(stderr,"perf instruction group unavailable: %s\n",strerror(errno));close(c.fd);c.fd=-1;}
    /* Keep the member fd open until process exit; group reads contain both. */
    return c;
}
static memory_counter memory_counter_read(memory_counter c) {
    uint64_t v[5];
    if(c.fd<0 || read(c.fd,v,sizeof(v))!=(ssize_t)sizeof(v) || v[0]!=2) {c.fd=-1;return c;}
    c.enabled=v[1];c.running=v[2];c.cycles=v[3];c.instructions=v[4];return c;
}
static double memory_cycles(memory_counter a,memory_counter b) {
    if(a.fd<0 || b.fd<0 || b.running<=a.running ||
       (double)(b.running-a.running)/(double)(b.enabled-a.enabled)<0.99) return -1;
    return (double)(b.cycles-a.cycles);
}
#endif
