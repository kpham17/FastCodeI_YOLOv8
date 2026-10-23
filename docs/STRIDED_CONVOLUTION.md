# Channel-strided 3x3 convolution

`conv_3x3_strided` in `include/conv2D.h` consumes a channel slice of an NHWC
parent tensor without allocating and copying that slice first. It uses the
current AVX-512 4x48/4xR microkernels and the existing packed-weight format.
The original `conv_3x3` and `conv_1x1` routines keep their existing behavior.

For a parent `[H][W][2*C]`, a C2f split is represented by the same parent
pointer, a physical pixel stride of `2*C`, and a channel offset of `0` or `C`.
These stride and offset values count **floats**, not bytes.

```c
// raw_kernel has shape [kernel_width=3][kernel_height=3][C][OC].
float *packed = kernel_pack(raw_kernel, 9 * C, OC);
float *first = conv_3x3_strided(parent, H, W, C, packed, OC,
                               1, 2 * C, 0);
float *second = conv_3x3_strided(parent, H, W, C, packed, OC,
                                1, 2 * C, C);
// Check allocations before use. Outputs are dense [H][W][OC] at stride 1.
_mm_free(first);
_mm_free(second);
_mm_free(packed);
```

The final two parameters are `input_pixel_stride` and `channel_offset`.
The earlier `stride` parameter remains the **spatial convolution stride**.
For example, a spatial stride of 2 produces `ceil(H/2) * ceil(W/2)` pixels.
Padding is always one pixel on each side, as in `conv_3x3`.

Requirements:

- `H`, `W`, and spatial stride are positive.
- `C` is positive and divisible by 8; `OC` is positive and divisible by 16.
- `0 <= channel_offset <= input_pixel_stride - C`.
- The parent tensor is dense NHWC with `input_pixel_stride` floats per pixel.
- Weights are packed with `kernel_pack(raw_kernel, 9*C, OC)`, using only the
  selected input channels. The packed-weight allocation is 64-byte aligned.
- Input addresses may be unaligned. Output is contiguous and 64-byte aligned.
- Invalid arguments, unsupported dimensions, and allocation failures return
  `NULL`. Callers own the returned allocation and release it with `_mm_free`.

The routine handles widths that do not fill a four-output tile, including
stride-2 outputs. Extra zero halo columns protect the input window reads;
an aligned temporary output tile prevents writes past the last output pixel.

This removes the **separate chunk copy**. The convolution still gathers the
selected channels into its row window and packs the four-pixel A panel.
It does not guarantee a speedup for every shape or remove all memory traffic.

## Verification

```sh
make test
clang -std=c11 -O2 -g -march=native -fgnu89-inline -Wno-pass-failed \
    -fsanitize=address,undefined -fno-omit-frame-pointer \
    tests/test_conv_strided.c -lm \
    -o /tmp/test_conv_strided_sanitize.x
ASAN_OPTIONS=detect_leaks=1 /tmp/test_conv_strided_sanitize.x
```

The new correctness test checks independent double-precision scalar results,
both C2f chunk halves, arbitrary unaligned offsets and pixel spacing, padding,
spatial strides 1, 2, 3, 4 and 7, 16/32-channel output remainders, short output
tiles, input immutability, invalid arguments and overflow rejection, and C2f shapes for
640, 1024, and 1280 image sizes. Shapes supported by the original convolution
also receive a bit-for-bit comparison against copying the chunk first.
The Clang sanitizer command uses GNU inline semantics for the existing
microkernel headers; the default GCC build remains unchanged. Clang's bundled
sanitizer runtime is used because the school GCC installation lacks libasan.

## Timing

```sh
make benchmark_conv_strided.x
taskset -c 2 ./benchmark_conv_strided.x 31
```

The optional argument is an odd sample count from 3 through 1001. The benchmark
compares dense convolution alone, copying a chunk into a reusable dense buffer
plus convolution, and convolution directly from the strided parent view.
It checks output equality before timing, warms each path, rotates measurement
order, and reports median microseconds. Weight packing and initial buffer
allocation are outside the measurement. Per-call output/scratch allocation is
included for all paths, and copying is included only for copy-plus-convolution.
Output freeing is outside the measurement for all paths. These are warm-cache,
single-thread convolution measurements, rather than end-to-end YOLOv8 timings.

### School-machine result, October 22, 2026

Intel Xeon Silver 4208, GCC `-std=c11 -march=native -O3 -mavx2 -mfma`,
CPU 2, 31 samples per path, second C2f chunk, spatial stride 1. Times are
medians in microseconds. The ratio is copy-plus-convolution divided by strided
convolution; a ratio above 1 means the strided path was faster in this run.

| H = W | C = OC | Copy + conv | Strided conv | Ratio |
|---:|---:|---:|---:|---:|
| 160 | 16 | 4806.888 | 4689.060 | 1.0251 |
| 80 | 32 | 2800.064 | 2730.793 | 1.0254 |
| 40 | 64 | 2724.540 | 2812.391 | 0.9688 |
| 20 | 128 | 2512.605 | 2511.928 | 1.0003 |
| 256 | 16 | 10594.321 | 10194.005 | 1.0393 |
| 128 | 32 | 7603.400 | 7381.158 | 1.0301 |
| 64 | 64 | 6705.604 | 6850.650 | 0.9788 |
| 32 | 128 | 5893.298 | 5904.120 | 0.9982 |
| 320 | 16 | 15863.858 | 15107.717 | 1.0500 |
| 160 | 32 | 11208.530 | 11289.672 | 0.9928 |
| 80 | 64 | 9807.006 | 9412.168 | 1.0419 |
| 40 | 128 | 9176.764 | 8855.909 | 1.0362 |

The separate chunk copy is eliminated; measured runtime remains dependent on
shape. The ratios range from 0.9688 to 1.0500 in this warm-cache run.
Verification passed: the existing suite, 160 scalar-reference cases, 12
argument-rejection cases, and Clang AddressSanitizer/UndefinedBehaviorSanitizer
with leak detection.
