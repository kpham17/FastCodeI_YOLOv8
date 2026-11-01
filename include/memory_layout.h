#ifndef MEMORY_LAYOUT_H
#define MEMORY_LAYOUT_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Dense FP32 [N][C/16][H][W][16]. No padding; C must be a positive
 * multiple of 16. All sizes/capacities are FLOAT elements. A channel slice
 * starts at image_base+c0*H*W, c0%16==0. Slice each batch separately. */
typedef enum {
    MEMORY_OK=0, MEMORY_BAD_ARGUMENT=-1, MEMORY_CAPACITY=-2, MEMORY_OVERLAP=-3,
    MEMORY_UNSUPPORTED_ISA=-4, MEMORY_OVERFLOW=-5
} memory_status;
/* Select ZMM for L1-sized work, AVX2 for larger streams on ece020.
 * Define MEMORY_PORTABLE at build time to use the portable implementation. */
static inline size_t tensor_index(size_t c, size_t y, size_t x,
                                  size_t h, size_t w) {
    return (((c/16)*h+y)*w+x)*16+c%16;
}
#ifdef __cplusplus
}
#endif
#endif
