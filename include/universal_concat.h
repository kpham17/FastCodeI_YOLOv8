#ifndef UNIVERSAL_CONCAT_H
#define UNIVERSAL_CONCAT_H
#include "memory_layout.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    const float *data;
    size_t channels;
    size_t capacity; /* Readable float elements remaining from data. */
} universal_concat_view;
/* Positive N,H,W,K and channels divisible by 16. Inputs dense per entire
 * batch; output [N][sum(C)/16][H][W][16], preserving logical channel order.
 * Inputs may overlap each other. Output must not overlap any input span or
 * the descriptor array. Valid float objects need no SIMD alignment.
 * All invalid calls return an error before writing output. No allocation.
 */
int universal_concat_f32(const universal_concat_view *inputs,size_t input_count,
                                float *output,size_t output_capacity,
                                size_t batch,size_t height,size_t width);
#ifdef __cplusplus
}
#endif
#endif
