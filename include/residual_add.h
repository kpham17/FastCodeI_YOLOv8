#ifndef RESIDUAL_ADD_H
#define RESIDUAL_ADD_H
#include "memory_layout.h"
#ifdef __cplusplus
extern "C" {
#endif
/* One image: dst=conv+shortcut in [C/16][H][W][16] order. H,W positive,
 * C positive/divisible by 16, all capacities in floats. dst==conv supported;
 * any other dst/conv overlap and all dst/shortcut overlap rejected. Shortcut
 * preserved. A block-aligned C2f half starts at cv1+C*H*W (not cv1+C).
 * Zero sizes/invalid calls return errors without writing. Float alignment
 * sufficient; strict FP32 addition, no allocation, no conversion. */
int residual_add(float *dst,size_t dst_capacity,
                         const float *conv,size_t conv_capacity,
                         const float *shortcut,size_t shortcut_capacity,
                         size_t height,size_t width,size_t channels);
#ifdef __cplusplus
}
#endif
#endif
