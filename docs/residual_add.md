# Standalone FP32 NHWC residual addition

Production is **kernels/residual_add.c + include/residual_add.h**. GCC 8.5 on
ece020 (Xeon Silver 4208) is the measured target. The kernel uses one CPU thread,
no allocations, and preserves the shortcut for C2f's final concatenation.

```c
#include "residual_add.h"
/* Dense shortcut, later bottleneck output. */
residual_add_contiguous(conv_out, conv_out, shortcut, H*W, C);
/* Zero-copy chunk B from an NHWC cv1 output with 2*C channels. */
residual_add_strided(conv_out, conv_out, cv1_output+C, H*W, C, 2*C);
```

`pixels=H*W` and `pitch` is in **float elements**. Output and convolution input
are dense. `dst==conv` is allowed; other partial dst/conv overlap is unsupported.
Shortcut must not overlap output. Either zero dimension returns without reading
pointers. Generic valid pitches >= C, channel tails and SIMD-unaligned buffers
are supported. The caller supplies valid object spans and representable sizes.

Dense uses AVX2 except for selected small AVX-512 shapes. Aligned in-place 2048
P3/P4 use an AVX2 pixel loop. Strided chunk B uses shared 128-output assembly
tiles with fixed shortcut displacements and optional 4 KiB prefetch; 2048 P5
uses an interleaved AVX2 loop. Runtime ISA checks protect these paths. Keep GCC
`noipa`: specialization changed the measured pixel-loop schedule. All loads and
stores support SIMD misalignment; 64-byte alignment gates only a tuning choice.

The repeated strided cases are expressed once with macros. Compiled `.text`
is checked against the previously measured kernel before accepting source-only
simplification. Historical variants and experiment generators are not needed
to build this implementation and are kept outside the repository.

## Build and check

```sh
make residual                       # kernels/libresidual_add.a
make check-residual                 # C oracle/guards and C++ linkage, CPU 0
make bench_residual_add.x
taskset -c 0 ./bench_residual_add.x > /tmp/residual-normal.csv
taskset -c 0 ./bench_residual_add.x --reverse > /tmp/residual-reverse.csv
make check-residual-sanitize CC=clang
make clean
```

`RESIDUAL_CFLAGS` deliberately avoids global `-march`/AVX flags and disables
fast-math and FP contraction. Change the test CPU with `RESIDUAL_CPU=2`. GCC's
ASan runtime is missing on ece020; Clang's installed runtime can check the same
source, while ignoring GCC-only optimization attributes. Sanitizers do not
instrument inline assembly; independent guard-page tests check its tails.

The C suite covers 312 model/generic configurations and 300 exact-span guarded
configurations, each in-place and out-of-place, plus empty null calls. The
shifted cases also shift the shortcut. All 12 model shapes are checked with
both dense and chunk-B layouts; the C++ smoke test checks both exported symbols.

The standalone benchmark validates each output against a scalar oracle, then
reports nine-batch median elapsed time for each of 24 shape/layout cells. Each
batch targets 8 ms and resets the in-place output outside timing. `--reverse`
reverses cell traversal. GB/s counts 12 useful bytes/add, not DRAM-bus traffic.
These are warmed standalone timings, not an integrated inference measurement.

To reproduce ten independent consistency runs, keep raw artifacts outside Git:

```sh
out=$(mktemp -d /tmp/residual-check-XXXXXX)
make residual test_residual_add.x test_residual_add_cpp.x bench_residual_add.x
for i in $(seq 1 10); do
    make check-residual > "$out/test-$i.log" || exit
    if [ $((i % 2)) -eq 0 ]; then order=--reverse; else order=; fi
    taskset -c 0 ./bench_residual_add.x $order > "$out/run-$i.csv" || exit
done
make clean
```

## Ten-run consistency check (ece020, 2026-10-24)

Ten independent processes were run **after integration into this checkout**, pinned to CPU 0. Each run passed all 612 C configurations in both output modes and the C++ test, then validated and timed all 24 model shape/layout cells. Runs alternated forward/reverse traversal. No correctness failures occurred.

Values below summarize the ten per-process median latencies. CV is population standard deviation divided by the mean; min/max describe observed spread, not confidence bounds. Shared-machine load and cache/frequency transitions remain sources of variation.

| Input | Stage | Layout | Min (us) | Median (us) | Max (us) | CV |
|---:|:---:|:---|---:|---:|---:|---:|
| 640 | P2 | contiguous | 151.10 | 156.50 | 163.18 | 2.72% |
| 640 | P2 | strided | 157.27 | 161.14 | 173.92 | 2.74% |
| 640 | P3 | contiguous | 69.06 | 70.95 | 71.93 | 1.33% |
| 640 | P3 | strided | 66.16 | 67.67 | 69.61 | 1.42% |
| 640 | P4 | contiguous | 11.99 | 14.28 | 17.99 | 14.24% |
| 640 | P4 | strided | 18.37 | 19.63 | 20.88 | 3.71% |
| 640 | P5 | contiguous | 7.33 | 7.43 | 7.71 | 1.44% |
| 640 | P5 | strided | 5.54 | 5.62 | 5.82 | 1.69% |
| 1024 | P2 | contiguous | 409.41 | 426.26 | 430.71 | 1.87% |
| 1024 | P2 | strided | 540.72 | 542.51 | 569.02 | 1.49% |
| 1024 | P3 | contiguous | 194.13 | 195.16 | 220.10 | 3.98% |
| 1024 | P3 | strided | 217.18 | 219.89 | 222.00 | 0.65% |
| 1024 | P4 | contiguous | 96.81 | 97.28 | 104.91 | 2.36% |
| 1024 | P4 | strided | 93.39 | 95.73 | 97.62 | 1.18% |
| 1024 | P5 | contiguous | 26.40 | 29.46 | 30.89 | 4.82% |
| 1024 | P5 | strided | 34.35 | 35.69 | 38.28 | 2.83% |
| 2048 | P2 | contiguous | 2885.76 | 2891.75 | 2922.09 | 0.37% |
| 2048 | P2 | strided | 3795.46 | 3806.18 | 3861.93 | 0.56% |
| 2048 | P3 | contiguous | 1130.68 | 1155.45 | 1178.12 | 1.41% |
| 2048 | P3 | strided | 1456.85 | 1475.80 | 1534.69 | 1.51% |
| 2048 | P4 | contiguous | 421.20 | 425.63 | 433.75 | 0.84% |
| 2048 | P4 | strided | 525.37 | 534.76 | 562.48 | 1.94% |
| 2048 | P5 | contiguous | 194.32 | 196.00 | 203.44 | 1.38% |
| 2048 | P5 | strided | 209.52 | 226.87 | 238.70 | 5.21% |

The largest timing variation is **640 P4 contiguous**, 11.99-17.99 us (14.24% CV). Correctness is repeatable, but this small cell's latency is less stable than the larger workloads. No frequency/counter evidence was collected to assign a specific cause; the full range is retained rather than discarding runs.

Traversal order is associated with visible differences in two cells:

| Cell | Forward median (us), five runs | Reverse median (us), five runs |
|:---|---:|---:|
| 640 P4 contiguous | 15.16 | 12.28 |
| 2048 P5 strided | 210.12 | 234.63 |

The check therefore establishes stable correctness and records the timing sensitivity, rather than asserting identical latency across every scheduling order.

Estimated six-call sums (P2 view, P3 view+dense, P4 view+dense, P5 view) are sums of standalone cell medians within each process, not integrated C2f timings:

| Input | Min (us) | Median (us) | Max (us) | CV |
|---:|---:|---:|---:|---:|
| 640 | 334.73 | 341.30 | 349.50 | 1.24% |
| 1024 | 1183.15 | 1191.44 | 1217.13 | 0.95% |
| 2048 | 7592.65 | 7638.73 | 7679.53 | 0.33% |

Production C: **233 lines including comments**, 9,903 bytes. Its GCC 8.5 `.text` is byte-identical to the previously measured kernel; removing more source through a generic intrinsic-only replacement lost performance in the earlier experiments. The explicit C=64 prefetch pair preserves GCC's original instruction schedule.

Source SHA-256: `984f2187fa6ed946a6bed4e1e519d507a9fb6abfa2f52b5c1cc3fcd375c82380`.

Full repository `make test` and the Clang ASan/UBSan residual suite were also run. Detailed raw evidence is kept outside the Git checkout at `~/residual_add_repo_checks_20261024_i7RLVp/`. Generated objects, libraries and executables are removed with `make clean`; the checkout contains source, tests, benchmark and documentation changes only. No commit or push was performed.
