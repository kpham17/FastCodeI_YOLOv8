#ifndef RESIDUAL_ADD_H
#define RESIDUAL_ADD_H
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* FP32 NHWC, one thread, no allocation. dst and conv are dense [pixels,C].
 * dst == conv is supported; other dst/conv overlap is unsupported. Shortcut
 * is read-only and must not overlap dst. Pointers may be unaligned to SIMD
 * width, but must point to valid float objects. Either zero size is a no-op.
 * Caller supplies valid sizes/objects; all pointer arithmetic must fit them.
 */

/* Dense shortcut: dst[p*C+c] = conv[p*C+c] + shortcut[p*C+c].
 * Each buffer contains at least pixels*C floats. */
void residual_add_contiguous(float *dst, const float *conv,
                             const float *shortcut, size_t pixels, size_t C);

/* View shortcut: dst[p*C+c] = conv[p*C+c] + shortcut[p*pitch+c].
 * pitch is in FLOATS and must be >= C. For pixels>0, shortcut contains at
 * least (pixels-1)*pitch+C floats from its supplied pointer. For C2f chunk B,
 * pass shortcut=cv1_output+C and pitch=2*C; no chunk copy is needed. */
void residual_add_strided(float *dst, const float *conv,
                          const float *shortcut, size_t pixels,
                          size_t C, size_t pitch);

#ifdef __cplusplus
}
#endif
#endif
