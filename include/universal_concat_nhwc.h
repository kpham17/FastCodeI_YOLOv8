#ifndef UNIVERSAL_CONCAT_NHWC_H
#define UNIVERSAL_CONCAT_NHWC_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    UC_OK = 0,
    UC_BAD_ARGUMENT = -1,
    UC_INSUFFICIENT_CAPACITY = -2,
    UC_OUTPUT_OVERLAP = -3,
    UC_OVERFLOW = -5
} uc_status;

/* Sizes are measured in float elements, not bytes. Capacity starts at data,
 * so an offset view must report only the elements remaining after its offset.
 * Dense input: pixel_stride == channels. A view into a larger NHWC tensor
 * uses that tensor's channel count as pixel_stride.
 */
typedef struct {
    const float *data;
    size_t channels;
    size_t pixel_stride;
    size_t capacity;
} uc_input_view;

/* Concatenate channels into dense [batch,height,width,sum(channels)] output,
 * preserving input order and every FP32 bit pattern. Dimensions, input_count
 * and channels must be positive; each pixel_stride must be >= channels.
 *
 * Required elements: output = N*sum(channels), input i =
 * (N-1)*inputs[i].pixel_stride + inputs[i].channels, where N=batch*height*width.
 * Row/batch padding is not represented separately from pixel_stride.
 *
 * Caller supplies valid float-aligned objects and keeps inputs/descriptors
 * alive and unchanged until return. Inputs may overlap each other. Output
 * must not overlap any input's bounding span (including stride gaps) or the
 * descriptor array. Disjoint ranges of one allocation are allowed.
 *
 * Invalid calls never write output. Cached AVX2 is selected automatically
 * when CPU/OS support it; otherwise memcpy is used. No heap allocation,
 * threading, floating-point arithmetic or intermediate tensor is performed.
 */
int universal_concat_nhwc_f32(const uc_input_view *inputs, size_t input_count, float *output,
                              size_t output_capacity, size_t batch, size_t height, size_t width);

/* Human-readable text for a returned status; no allocation. */
const char *uc_status_string(int status);

#ifdef __cplusplus
}
#endif
#endif
