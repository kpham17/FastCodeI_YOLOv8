#ifndef FUSED_UPCONCAT_H
#define FUSED_UPCONCAT_H

#include <stddef.h>

/* FP32, contiguous NHWC. Inputs: up[H][W][Cu], skip[2H][2W][Cs].
 * Output: out[2H][2W][Cu+Cs], channels ordered [upsampled, skip].
 * All buffers must be large enough and out must not overlap either input.
 * The functions allocate nothing and preserve all FP32 bit patterns.
 * Requires an AVX2-capable CPU (the target is ece020). */

/* YOLOv8n layers 10+11: Cu=256, Cs=128, output channels=384. */
void fused_upconcat_a(const float *up, const float *skip, float *out,
                      size_t h, size_t w);

/* YOLOv8n layers 13+14: Cu=128, Cs=64, output channels=192. */
void fused_upconcat_b(const float *up, const float *skip, float *out,
                      size_t h, size_t w);

#endif
