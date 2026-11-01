#ifndef FUSED_UPCONCAT_H
#define FUSED_UPCONCAT_H
#include "memory_layout.h"
#ifdef __cplusplus
extern "C" {
#endif
/* One image: up [Cu/16][H][W][16], skip [Cs/16][2H][2W][16],
 * out [(Cu+Cs)/16][2H][2W][16], up channels first. Positive dimensions,
 * positive Cu,Cs divisible by 16. All capacities in floats. Output disjoint
 * from both input spans. Float alignment sufficient. Invalid calls do not
 * write; zero sizes are errors. No intermediate tensor or allocation. */
int fused_upconcat_f32(const float *up,size_t up_capacity,
                              const float *skip,size_t skip_capacity,
                              float *out,size_t out_capacity,size_t up_h,size_t up_w,
                              size_t up_channels,size_t skip_channels);
int fused_upconcat_a(const float *up,size_t up_capacity,
                            const float *skip,size_t skip_capacity,
                            float *out,size_t out_capacity,size_t up_h,size_t up_w);
int fused_upconcat_b(const float *up,size_t up_capacity,
                            const float *skip,size_t skip_capacity,
                            float *out,size_t out_capacity,size_t up_h,size_t up_w);
#ifdef __cplusplus
}
#endif
#endif
