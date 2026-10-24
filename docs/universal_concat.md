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

The implementation selector was removed: existing calls should drop their
final `UC_AUTO` argument. The function always chooses the measured production
policy, with no AVX-512 or streaming-store implementation to configure.

## YOLOv8n coverage and performance

One entry point covers all 11 standalone feature-map joins in the supplied
YOLOv8n inventory: eight C2f joins, SPPF's four-input join, and neck sites 17/20.
The tests cover all 33 site/resolution combinations at 640/1024/2048, plus odd
shapes, batches, arbitrary strides/counts, offset views, bit patterns, buffer
guards, exact allocation tails and reject-before-write cases. Unix builds
also test vector tails against inaccessible guard pages.

C2f's dense `[cv1 pair, bottleneck1, ...]` representation retains the measured
fixed-layout scheduling. Large dense `[128,64]` and `[128,64,64]` joins with
N>=16384 select sequential AVX2 on AVX-512-capable CPUs, preserving the
previous target policy. That feature-bit check qualifies the policy; the
kernel contains no AVX-512 instructions. Other valid layouts use the generic
AVX2 copy loop. This is measured target tuning, not proof of optimality on
every CPU or every possible shape.

The two fused upsample+concat sites keep their existing separate kernels.
This function does not perform upsampling, convolution, layout conversion,
other data types or concatenation along non-channel axes. It has not been
integrated into a complete inference run.

Cleanup was checked against the previous production default at commit
`a9042b0` on ece020 (Intel Xeon Silver 4208, GCC 8.5, CPU2). Both kernels ran
on the same buffers in randomized order: 21 rounds, seven calls per warm
batch or one call per 64MiB eviction-proxy batch, three independent processes.
Each number sums eleven site medians within a process, then takes the median
of the three process sums. Times are milliseconds:

| Input | Mode | Previous default | Cleaned implementation |
|---:|---|---:|---:|
|640|warm|1.722058|1.716136|
|640|cold|2.052611|2.045481|
|1024|warm|5.500214|5.496288|
|1024|cold|6.045792|6.049171|
|2048|warm|29.356939|29.357463|
|2048|cold|30.235198|30.276017|

Aggregate differences are within 0.4%, supporting no material performance
regression on this matrix. All 396 comparison rows pass bitwise verification.
Native, portable and Clang ASan/UBSan each pass 313 cases; C++ linkage and
integration examples pass. GCC's object text size falls from 47440 to 23891
bytes (49.6%); assembly contains no ZMM or non-temporal stores. Shared-host
variability remains; these are standalone single-core measurements, not
full-inference times or a hardware-peak claim.

The baseline identifier `a9042b0` refers to the standalone development snapshot;
detailed validation data are archived separately. Development reports,
reference copies and alternate implementations are excluded from this tree.

## Reproduce a standalone measurement

```sh
make bench-concat
# Choose a quiet logical CPU and check its SMT sibling before measuring.
taskset -c 2 ./bench_universal_concat.x --rounds 21 --calls 7 > benchmarks/concat_warm.csv
taskset -c 2 ./bench_universal_concat.x --cold --rounds 21 --calls 1 > benchmarks/concat_cold.csv
```

The benchmark checks every timed batch against an independent byte oracle.
Allocation, input generation, verification and 64MiB eviction are outside the
timer. Warm batches have an untimed restore call; cold mode is an eviction
proxy, not a guarantee that every cache line is cold. CSV includes per-site
median/p10/p90 and useful bandwidth; metadata on stderr records host/affinity.
Useful traffic is twice output bytes, not measured DRAM traffic or a machine
peak. `--image N` accepts inventory images divisible by 32; the API itself
accepts arbitrary positive dimensions. `--site N` selects one inventory site.
