#ifndef CHUNK_NHWC_H
#define CHUNK_NHWC_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Split dense FP32 [pixels,2*C] into dense [pixels,C] halves, preserving bits.
 * A NULL destination omits that half. Either zero size is a no-op.
 * Caller supplies valid sizes/storage; input and non-NULL outputs are disjoint.
 * No allocation. Pointers may be unaligned to SIMD width, but float-aligned.
 * Target: ece020. One output requires AVX2; both outputs require AVX-512F.
 */
void chunk2_nhwc_f32(float *dst0, float *dst1, const float *src,
                     size_t pixels, size_t C);
#ifdef __cplusplus
}
#endif
#endif
