# Materialized NHWC channel chunks

`chunk2_nhwc_f32` splits a dense FP32 parent `[pixels,2*C]` into dense
`[pixels,C]` halves. Pass `pixels=H*W` (or `batch*H*W`).

```c
#include "chunk_nhwc.h"

/* Allocate/reuse output buffers with pixels*C floats each. */
chunk2_nhwc_f32(a, b, parent, pixels, C);    /* both halves */
chunk2_nhwc_f32(NULL, b, parent, pixels, C); /* only B */
chunk2_nhwc_f32(a, NULL, parent, pixels, C); /* only A */
```

The caller supplies valid disjoint input/output storage. Either zero size,
or both outputs NULL, is a no-op. No heap allocation, scratch tensor,
threading, or floating-point arithmetic occurs. Float-aligned pointers may
be unaligned to SIMD width; copied bits, including NaNs, are unchanged.
The fixed ece020 target requires AVX2 for one half and AVX-512F for both.
There is no runtime CPU compatibility dispatcher.

The implementation uses cached unaligned integer moves, constant-width
specialization for C=16/32/64/128, channel-loop unrolling, and strength
reduction. One-half copying uses AVX2; both halves use one AVX-512 pass
through the parent, except C=128 reuses the AVX2 half-copy routine twice.
Other widths use the same bounded loops and exact
memcpy tails. No per-resolution tuning or streaming-store branches remain.

Build/test from the school repository:

```sh
make chunk check-chunk
make check-chunk-sanitize CC=clang
make bench-chunk
make bench-chunk-consumers
```

Chunk benchmarks pin CPU 2 on socket 0 and use one thread. Check that its
SMT sibling CPU 18 is quiet. The kernel benchmark covers 12 model shapes,
A/B/both modes, bitwise checks, 15 warmed batches, wall time and core cycles.
Allocation/initialization is excluded from timing; the final store fence
is included. Compile with the supplied target-tuning flags to avoid GCC
splitting unaligned vector accesses into smaller instructions.

The consumer benchmark compares prepared-dense input, chunk-copy plus a
dense consumer, and the existing zero-copy view consumer. Copying is timed
in the second path. Dense/view results are checked bitwise. Convolution's
existing allocation/packing is included in every convolution path, while
output freeing and weight setup are outside timing.

Materialization adds a read and write per copied value. For S bytes in
each half, one-half logical traffic is 2S and both-half logical traffic is
4S. The optimistic store-throughput bounds for the selected instructions
are S/32 cycles for one half, 2S/64 cycles for the AVX-512 both-half path,
and 2S/32 cycles for the C=128 AVX2 both-half path. The machine's widest
ideal both-half bound remains 2S/64. These bounds assume independent
operations and available data; they are not cache or DRAM throughput claims.

Choose materialization using total copy-plus-consumer time, including all
uses of the chunk. The existing zero-copy convolution/add APIs are retained;
the presence of a faster copy kernel does not prove a full C2f/model speedup.

On ece020, three process runs over 36 copy cases gave a 1.15x geometric
mean speedup versus per-pixel memcpy (range 0.93x--1.40x). Some both-output
cases had substantial process-to-process variation; no universal fastest
claim is justified. Extra pixel blocking and streaming stores did not give
a consistent overall benefit, so they are excluded from the production API.
Copying B before residual addition took 1.70x--2.47x the existing view-path
time over the 12 tested shapes. Convolution plus copying ranged from 0.98x
to 1.09x the view-path time, with small apparent wins within measurement
variation. These are individual consumer tests, not a full-model result.
