# SIMD, warm L1, and the 90% target

**Yes, the kernels execute SIMD. No, the complete checked calls do not yet
reach 90% of the store-only reference.** This report keeps those facts separate.
Measurements ran on ece020's Xeon Silver 4208 using GCC 8.5.0, pinned to CPU6
(SMT sibling CPU22), with at least 10 ms warming before every sample.
No code was pushed or committed.

## SIMD is verified in the compiled and executed code

Freshly compiled concat and fused bodies contain `vmovups zmm` loads/stores:
each moves 16 FP32 values. Residual contains `vaddps zmm` and ZMM loads/stores.
The 12 KiB residual call retires **64 packed 512-bit single-precision arithmetic
instructions and zero scalar single-precision arithmetic instructions** per
call, matching its 1,024 elements. This is runtime counter evidence, not just
the presence of an unused SIMD function in the binary. Copy and fused kernels
perform no arithmetic, so their zero FP-arithmetic counts are expected.
Their assembly and retired-store counts establish the full-width copy path.

The event definitions are from [Intel's Cascade Lake event documentation](https://perfmon-events.intel.com/platforms/cascadelakex/core-events/core/).
The FP-arithmetic profile sets DAZ/FTZ as required by Intel for that event;
all benchmark operands are normal finite values or zero. Other timing/cache
profiles and production correctness tests retain their usual FP environment.

## Tiny sets really are warm

Tests include a generic fused operation with only **832 tensor bytes**
(64 B up, 256 B skip, 512 B output), a 1.5 KiB add, and a 2 KiB concat.
Real fused stages retain their production channel counts: A needs 9 KiB
at 1x1, and B needs 4.5 KiB. The largest selected tensor footprint is 16 KiB,
half the 32 KiB L1 capacity. Alignment is controlled for reproducibility;
the public kernels still accept ordinary float-aligned pointers.

Three processes and three diagnostic rounds per profile measured the current
pre-optimization checked functions as follows:

| Operation | Tensor bytes | L1 load-hit rate | L2 RFO requests/call |
|---|---:|---:|---:|
| Generic fused 1x1, Cu=Cs=16 | 832 | 99.99958% | 0.000128 |
| Residual 1x8,C=16 | 1,536 | 99.99965% | 0.000087 |
| Concat, two C16 inputs, 1x8 | 2,048 | 99.99948% | 0.000145 |
| Fused A, 1x1 | 9,216 | 99.99938% | 0.001093 |
| Fused B, 1x1 | 4,608 | 99.99955% | 0.000286 |
| Residual 8x8,C=16 | 12,288 | 99.99933% | 0.000627 |
| Concat, two C16 inputs, 8x8 | 16,384 | 99.99854% | 0.002894 |

RFO counts check write ownership traffic beyond L1; load hits alone would not
prove output residency. These negligible miss/ownership rates cannot explain
the large throughput gap. Allocation, packing, reference computation, output
validation and input reset are outside the timed call. No cold-cache timing,
non-temporal stores, padding stores, or work-eliding kernel is used.

## Why complete calls are slow even with tiny warm data

The public APIs perform capacity/overflow/overlap checks, dispatch, call setup,
traversal and return. These operations do not shrink in proportion to the data.
The 832-byte fused case writes only eight useful output vectors, but the old
complete call retires 37 stores including bookkeeping. The tiny residual has
eight useful output stores and 16 total retired stores. A smaller tensor
therefore exposes more overhead per useful output, even with near-perfect L1
hits. Extra stack stores also consume the store pipeline.

For 16 KiB concat, the exact production copy bodies reach **0.972 useful
64-byte output stores/core-cycle**, while the complete call reaches 0.630.
A repeated copy control reaches 0.998. That identifies the complete-call
overhead rather than missing SIMD as the main gap for this fixture.
Relative buffer offsets also matter: changing only offsets produces false
address dependencies despite L1 hits. The raw offset-control CSVs preserve
those experiments; alignment alone is not a universal throughput guarantee.

A new tiny-set placement experiment keeps inputs, sizes, instructions and
warming identical, and shifts only the output's page offset from 0 to 2,048 B:

| Repeated SIMD control | Output offset 0 | Output offset 2,048 | False-address detections/call, before/after |
|---|---:|---:|---:|
| 2 KiB concat | 0.702 | 0.999 | 16.0 / 0.0 |
| 4 KiB concat | 0.702 | 0.998 | 32.0 / 0.0 |
| 1.5 KiB residual add | 0.413 | 0.694 | 15.9 / 0.0 |

L1 hits remain essentially perfect. This isolates false address dependencies
as another reason tiny warm SIMD streams can be slow. These are explicitly
repeated-loop controls, not public-call results: the 2 KiB complete concat
call stays near 0.148 with either placement. Its setup costs dominate after
the memory-dependency issue is removed. The public benchmark records actual
page offsets and does not substitute the fastest control for the kernel.

## Fused traversal: a measured fix

The original SIMD code traversed channel blocks, rows and columns even when
W=1. In that case each source pixel's four output vectors are consecutive.
The implementation now uses one flat loop for W=1, with the same four stores
and the same real skip copy. It works for arbitrary supported channel counts
and heights, without a per-stage assembly implementation or a new API.

Three-process, five-round before/after measurements give these medians:

| Fused fixture | Complete call before | Complete call after | Core-cycle reduction |
|---|---:|---:|---:|
| A, 1x1, 9 KiB | 0.460 | 0.596 | 22.9% |
| B, 1x1, 4.5 KiB | 0.324 | 0.415 | 21.9% |
| Generic 1x1, 832 B | 0.085 | 0.096 | 11.3% |

The reductions above are in core cycles, not elapsed time. In the same
samples, median elapsed times changed from 93.8 to 77.6 ns (A), 72.4 to
53.7 ns (B), and 46.1 to 39.8 ns (generic); clock variation makes the ratios
different. These are measured call times on the shared machine.

Rates are useful 64-byte output units per core cycle. The new exact production
bodies, including skip copy but excluding public validation, have medians
0.945 for A and 0.918 for B. Individual process medians vary (A 0.794–0.949,
B 0.859–0.939); these are not guarantees that every run exceeds 90%.
The full checked functions remain below the 90% target.
The general W>1 traversal and all public contracts remain unchanged.

## Residual: the second read stream is the remaining bottleneck

For the same warmed 12 KiB buffers and normal operands:

| Work measured | Useful output stores/core-cycle |
|---|---:|
| Complete checked residual call | 0.535 |
| Exact production SIMD body | 0.679 |
| Low-overhead two-input add loop | 0.701 |
| Two loads + one store, with the add removed | 0.702 |
| One load + register-constant add + one store | 0.997 |
| Store-only reference | about 0.998 |

Removing FP addition does not fix the two-load/one-store rate. Removing one
read stream does. The bottleneck is therefore mixed memory execution for
these streams, not scalar arithmetic or cold cache. The one-load control
changes the operation and touches fewer bytes; it is explicitly not a valid
replacement residual kernel. The two-load/no-add control retains both input
streams and is the stronger test excluding FP addition as the principal cause.

Intel lists 192 B/cycle maximum and approximately 133 B/cycle sustained L1
bandwidth for this core family in its [optimization manual](https://cdrdv2-public.intel.com/814198/248966-Optimization-Reference-Manual-V1-049.pdf).
Residual requires 192 bytes per 64-byte output (two reads and one write);
0.70 outputs/cycle corresponds to about 134 B/cycle. The observation is
consistent with that sustained figure, but does not prove a universal hard
ceiling or uniquely identify the internal cache resource. The 90% comparison
continues to use the measured store-only reference; it is not redefined to
make residual pass.

## Reproduce and inspect every 90% failure

```sh
build=/scratch/zihengli-memory-l1-90-20261031/afs-build
make MEMORY_BUILD="$build" MEMORY_CPU=6 bench-memory-l1
```

Check the chosen physical core and its SMT sibling are otherwise idle.
The target runs three processes of five warmed rounds, brackets them with
store calibration, writes CSVs under `$build`, and prints `YES` or `NO` for
90%. It validates complete unique samples, expected cases/footprints, and
finite positive measurements. It refuses a store-reference median below 0.95.
The complete-call results are intentionally not replaced by isolated loop
or store-only results. All selected complete calls currently print `NO`.

Native, portable, C++ and ASan/UBSan verification passed 120 cases. New tests
cover W=1 with generic channel counts, height>1, guards, unaligned buffers,
and a footprint large enough to select AVX2. The diagnostic controls are
outside the production API and outside the AFS source tree.

The shared machine was not reserved. Sibling CPU22 averaged 0.43% busy during
the main campaign and 1.31% during the before/after campaign, with brief
100 ms samples as high as 60%/70%. Store-calibration medians stayed between
0.9966 and 0.9987 in the main campaign. Some variation remains; all raw runs
and ranges are retained rather than selecting only the fastest samples.

Full assembly, input source hashes, exported-body provenance, per-event CSVs,
SMT samples, scripts and before/after results are under:
`/scratch/zihengli-memory-l1-90-20261031`.
