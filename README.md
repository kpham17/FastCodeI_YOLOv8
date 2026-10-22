# FastCodeI_YOLOv8
YOLOv8 implementation

## Kernel benchmark

Run `make benchmark` to benchmark 3x3 convolution at strides 1 and 2, 1x1
convolution, and the `fast_SILU` kernel. The benchmark uses three input shapes
(`16x32x8`, `32x48x16`, and `64x64x32`) with output channel counts 24, 48, and
72, respectively.

Output reports median TSC cycles per call, nominal FLOPs per call, and
FLOPs per TSC cycle. Convolution FLOPs count one multiply and one add per
multiply-accumulate, including padded 3x3 positions. SiLU FLOPs count four
floating-point operations per element, including its approximate reciprocal;
integer conversion and bit operations are not counted. Convolution timings
include the packing and temporary allocations performed inside each kernel;
input setup and output cleanup are excluded. Each result is the median of
seven samples of five calls.
