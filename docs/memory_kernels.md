# Memory kernels

The three production implementations are `universal_concat.c`,
`fused_upconcat.c`, and `residual_add.c`, with matching headers in `include`.
They use FP32 physical layout `[N][C/16][H][W][16]`, with positive channel
counts divisible by 16 and no padding. Layout details belong in the contract,
not in file or function names. The old NHWC implementations are removed.

## Interfaces

Include `universal_concat.h`, `fused_upconcat.h`, or `residual_add.h`.
All capacities are in floats, and all checked calls return `MEMORY_OK` on
success or a status defined in `memory_layout.h`. Invalid arguments,
overflow, short capacities, and prohibited overlap are rejected before writes.
Float alignment is sufficient. Kernels allocate nothing and convert nothing.

- `universal_concat_f32(views,k,out,out_capacity,N,H,W)` concatenates dense
  inputs separately within each image. `universal_concat_view` contains
  `data`, `channels`, and `capacity`. Output channel order follows view order.
- `fused_upconcat_f32(up,up_capacity,skip,skip_capacity,out,out_capacity,H,W,Cu,Cs)`
  performs 2x nearest upsampling and concatenates skip channel planes.
  `fused_upconcat_a` and `fused_upconcat_b` accept the same buffers/capacities
  and H,W, with channels fixed to 256/128 and 128/64 respectively.
- `residual_add(dst,dst_capacity,conv,conv_capacity,shortcut,shortcut_capacity,H,W,C)`
  adds dense blocked spans. `dst==conv` is supported; shortcut is preserved.
  A C2f second half begins at `cv1 + C*H*W` floats. Slice each image separately.

These replace the old interfaces; NHWC pixel strides are not accepted.
Callers must supply the documented physical layout. Convolution, chunk and
other teammates' kernels retain their own existing contracts; this change
does not make the complete model graph use the new layout automatically.

## Build and check on ece020

Use scratch for generated files because the AFS account is near quota:

```sh
build=/scratch/zihengli-memory-replace-20261031/afs-build
make MEMORY_BUILD="$build" memory-kernels
make MEMORY_BUILD="$build" check-memory
make MEMORY_BUILD="$build" check-memory-portable
make MEMORY_BUILD="$build" check-memory-sanitize
make MEMORY_BUILD="$build" bench-memory
taskset -c 5 "$build/bench_memory_kernels.x" warm 1 > /scratch/memory-kernels.csv
taskset -c 5 "$build/bench_l1_store.x" 1 > /scratch/memory-store.csv
```

The static library is `$build/libmemory_kernels.a`. Native code selects
AVX-512 for L1-sized operations and AVX2 for larger streams, with a portable
fallback. Define `MEMORY_PORTABLE` when compiling to force that fallback.
The older functional Make targets `concat`, `residual`, `upconcat` and their
checks build/use this same implementation. `bench-residual` runs the residual
benchmark; `benchmark-upconcat` runs three fused benchmark processes.
`MEMORY_BENCH_KERNEL=concat|upconcat|add` selects one operation.
`make test` includes the memory
suite alongside the existing unrelated repository tests.

The 120-case suite compares unpacked logical results with independent
scalar oracles; it does not depend on a second production implementation.
It covers production shapes, batching, copy bit patterns, C2f slices,
in-place add, unaligned objects, guard pages, capacities, overflow and overlap.
The C++ check verifies public headers and linking.

The benchmark validates scalar logical reference results before timing.
Allocation, packing, reference work and resets are outside timing. Each
case warms for at least 10 ms; perf core cycles are distinct from TSC ticks.
`MEMORY_BENCH_FILTER=L1` runs only the small cases. Tensor size below 32 KiB
does not guarantee conflict-free L1 residency or one output store per cycle.
The store-only calibration is a hardware reference, not a kernel result.
The chunk consumer benchmark retains its NHWC convolution cases; its old
residual comparison was removed because it assumed the deleted NHWC API.

The pre-replacement tree and historical old/new performance evidence are
preserved on scratch, outside this checkout. See `memory_kernel_results.md`.

## Warm L1 and the 90% test

Run `make MEMORY_BUILD="$build" MEMORY_CPU=6 bench-memory-l1` after choosing
an idle core and checking its SMT sibling. The target brackets three
processes of five warmed rounds with store-only calibration, writes CSVs
under `$build`, and reports the 90% decision for each complete checked call.
It rejects incomplete, duplicate, non-finite, or inconsistent samples.
All selected tensor footprints are 832 bytes to 16 KiB, with reproducible
page-aligned allocation. Larger small cases are excluded from this target.

The fused SIMD implementation flattens width-one inputs because each source
pixel maps to four consecutive output vectors. Heights and channel counts
remain generic. See [measurement and diagnosis](memory_l1_diagnosis.md) for
actual SIMD evidence, warmed-cache PMU results, complete-call versus body
measurements, and remaining failures.
