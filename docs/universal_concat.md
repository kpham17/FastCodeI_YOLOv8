# Universal NHWC concatenation

A checked C11 function that joins FP32 tensors along the channel axis into
one dense `[B,H,W,sum(Ci)]` output. Any valid positive dimensions, input
count, channel widths and per-input pixel strides are supported.

The implementation uses cached AVX2 on supported CPUs and a memcpy fallback.
It preserves every 32-bit pattern, including NaNs, and performs no allocation,
threading, floating-point arithmetic or intermediate tensor materialization.
The entire production implementation is **one header and one C file**.

## Build and use

```sh
make concat                    # kernels/libuniversal_concat.a
make check-concat              # byte oracle and C++ linkage
make check-concat-portable     # fallback with AVX2 disabled
make check-concat-sanitize CC=clang
```

GCC/Clang are supported. The default flags tune scheduling for the measured
Xeon Silver 4208 without globally enabling AVX2; SIMD is enabled only inside
runtime-dispatched functions. Override `CONCAT_CFLAGS` for another target, for example
`make concat CONCAT_CFLAGS='-O3 -std=c11 -Wall -Wextra -Wpedantic'`.

Compile `kernels/universal_concat_nhwc.c` and include
`include/universal_concat_nhwc.h`. No private header or reference source is
required. For example, two dense inputs with channels 32 and 16:

```c
#include "universal_concat_nhwc.h"

uc_input_view inputs[] = {
    {a, 32, 32, pixels * 32},
    {b, 16, 16, pixels * 16}
};
int status = universal_concat_nhwc_f32(
    inputs, 2, output, pixels * 48, batch, height, width);
/* Require status == UC_OK before consuming output. */
```

Here `pixels = batch*height*width`; `a`, `b` and `output` are caller-owned
float arrays with the stated capacities. Output channel order follows input
order. 

All strides and capacities are **float elements**, not bytes. For N pixels,
input i needs `(N-1)*pixel_stride + channels` elements; output needs
`N*sum(channels)`. For an offset view into another tensor, capacity starts at
the offset pointer and pixel_stride is the parent tensor's channel count.
Separate row/batch padding is not represented.

Arguments and arithmetic are checked before any output write. Inputs can
share storage; output must not overlap input bounding spans, stride gaps or
the descriptor array. Buffers must be valid float-aligned objects kept alive
through the call. Zero dimensions/count/channels are invalid. Use
`uc_status_string(status)` for an error description. The public header
contains the full contract.

## Copy schedules and coverage

The entry point covers the supplied inventory's 11 standalone joins: eight
C2f joins, the four-input SPPF join, and neck layers 17/20. Dimensions remain
runtime values; there are no separate implementations for different images.
Tests cover all 33 site/resolution combinations at 640/1024/2048, plus arbitrary
shapes, batches, strides, input counts, offset views, bit patterns, guards,
exact allocation tails, inaccessible guard pages and reject-before-write cases.

One AVX2 copy primitive serves both the generic loop and a shared constant-width
schedule. Exact dense C2f tuples `[2*C,C]` / `[2*C,C,C]` with C=16,32,64, and the
neck tuple `[64,128]`, use that schedule. C=16 takes the fixed schedule below 32768 total pixels;
paired tests favored the generic loop for its larger measured tensors. This
cutoff separates the measured workloads and is target tuning, not a claim about
every possible intermediate size. The first C2f input already contains
both adjacent cv1 splits. Everything else uses the generic per-input-stride
loop, including SPPF and the `[128,256]` neck tuple. A single dense input uses
one memcpy. Runtime CPU detection guards AVX2; other CPUs use portable memcpy.

## Measured simplification on ece020

The reference is the committed implementation at `9191c93004626f07673f5788724dc922fbad3672`.
The reviewed source decreases from **416 to 199 lines** including comments;
GCC object text decreases from **23891 to 4108 bytes (82.8%)**. Validation and the
public header remain unchanged. Expanded per-width routines, four-pixel
unrolling, specialized fallback copies, and the AVX-512 feature-bit policy are
replaced by shared routines. The retained SIMD implementation uses cached AVX2.

23 alternative copy implementations were screened. Final comparisons of the
144-, 165- and 188-line experimental candidates used three independent paired
processes on ece020 (Xeon Silver 4208, GCC 8.5, CPU2; sibling CPU18 idle at setup).
The final 199-line source adds explanatory comments and a C=16 pixel-count cutoff.
It was separately compared to the committed baseline across the full matrix.
The integrated object has identical `.text` bytes to that final measured source.

Each process ran both implementations on the same buffers in randomized order,
21 rounds per case: seven calls per warm batch, or one call after a 64 MiB
cache-eviction proxy. Each time below sums the 11 site medians within a process,
then takes the median of the three process sums. Times are milliseconds:

| Input | Mode | Current commit | Review candidate | Change |
|---:|---|---:|---:|---:|
|640|warm|1.705904|1.685077|-1.22%|
|640|cold|2.083249|2.071782|-0.55%|
|1024|warm|5.418057|5.356123|-1.14%|
|1024|cold|5.889115|5.821818|-1.14%|
|2048|warm|29.123771|28.782721|-1.17%|
|2048|cold|29.969255|29.603859|-1.22%|

All 396 paired result rows pass bitwise verification. Across per-case medians
from the three processes, the largest increase is 0.86%; the identical-code
control exhibited per-case increases up to 1.49%. This supports no material
regression on this matrix, rather than a claim of a universally optimal kernel.

The remaining families were then removed individually, comparing each deletion
to the compact candidate on affected cases: 17 rounds, five warm calls or one
cold call, three independent processes. The largest median slowdowns were:

| Removed family | Affected example | Slowdown |
|---|---|---:|
|C2f C=16|640, site 2, warm|3.03%|
|C2f C=32|1024, site 15, warm|1.84%|
|C2f C=64|640, site 12, warm|3.58%|
|Neck [64,128]|640, site 17, warm|2.87%|

These measurements justify keeping those four families. Still shorter generic
and two-input candidates showed repeatable slowdowns in individual production
cases. Experimental variants and detailed results are archived separately from
this production tree.

The final GCC native, forced-portable, Clang optimized and Clang ASan/UBSan
suites each pass 313 cases / 172412 assertions. C++ linkage, `make -B all kernels`
and the full repository `make test` target pass. Assembly uses YMM registers
without ZMM instructions or streaming stores.

The two fused upsample+concat sites retain their separate kernels. This function
copies FP32 NHWC channels; it does not perform upsampling or convolution.
The timings are standalone single-core measurements, not complete inference
latency or a percentage of the machine's hardware peak.

## Reproduce a standalone measurement

```sh
make bench-concat
# Choose a quiet logical CPU and inspect its SMT sibling before measuring.
taskset -c 2 ./bench_universal_concat.x --rounds 21 --calls 7 > concat_warm.csv
taskset -c 2 ./bench_universal_concat.x --cold --rounds 21 --calls 1 > concat_cold.csv
```

Allocation, generation, verification and eviction are outside timing. Warm
batches have an untimed restore call; cold mode is an eviction proxy. CSV
records median/p10/p90 and useful bandwidth; stderr records host and affinity.
Useful traffic is twice output bytes, not measured DRAM traffic. `--image N`
accepts inventory images divisible by 32; the API accepts arbitrary positive
dimensions. `--site N` selects one inventory site.
