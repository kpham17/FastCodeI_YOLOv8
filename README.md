# FastCodeI_YOLOv8
YOLOv8 implementation

## Batchnorm and maxpooling skeletons

The C APIs use CHW16 tensors stored as `[N][ceil(C / 16)][H][W][16]`.
The `C` argument remains the logical channel count; the final block is
zero-padded in storage when `C` is not a multiple of 16:

- [`include/batchnorm.h`](include/batchnorm.h)
- [`include/maxpool.h`](include/maxpool.h)

Build the shared library with:

```sh
make kernels
```

After implementing the two functions, compare runtime and analytic FLOP
estimates against PyTorch:

```sh
.venv/bin/python benchmarks/compare_kernels.py --library kernels/libfastkernels.so \
    --cpu 0
```

The default `1x16x16x16` workload keeps both CHW16 input/output tensors and
the parameter vectors in L1 cache. It warms the kernels 100 times, then takes
1000 timed calls on the same buffers; use `--cpu` to pin execution to one
logical CPU and reduce scheduler noise. Customize the workload with
`--shape N H W C`, `--warmups`, and `--repeats`. Larger workloads measure
streaming behavior rather than the near-ideal cached load/store throughput
targeted by this benchmark.

For native cycle measurements without Python/`ctypes` call overhead, use:

```sh
make benchmark-kernels KERNEL_CPU=0
```

This pins the process to one logical CPU, warms a `1x14x16x16` CHW16 working
set, and measures the 5x5 16-channel 5x5 stride-1 maxpool tile (without
boundary fallback) and batchnorm in separate repeated loops. The tile
benchmark isolates the vectorized load/reduction/store path; full API
correctness and boundary behavior are covered by the numerical checks.
Batchnorm preparation is performed once before timing, so its reported
FLOP/cycle values use a 32 FLOP/core-cycle ceiling for a 16-lane FMA
(2 FLOPs per lane) and describe the steady-state load/FMA/store loop rather than
allocation, square-root, and parameter setup overhead. If permitted, the
benchmark also reports `perf_event_open` core cycles and reference cycles;
the latter is the reproducible denominator when frequency tracking is
unreliable, while the former reports actual core execution efficiency.
Maxpool is reported in comparison-equivalents/cycle rather than arithmetic
FLOPs/cycle because its reduction uses floating-point comparisons. For the
5x5 tile, the work count is `25 outputs * 16 lanes * 24 comparisons =
9600 comparison-equivalents`. With one AVX-512 max functional unit, one
512-bit `vmaxps` per core cycle gives a 16
comparison-equivalent/core-cycle peak. The benchmark also reports the
comparison-equivalent count per TSC cycle; this is an effective FLOP-like
number, not literal arithmetic FLOPs, and its peak varies with the
core-frequency/TSC ratio.
The benchmark also compares the intrinsic 8-way unrolled path with an
experimental inline-assembly no-copy path. The no-copy path loads each input
directly into its result register before the FMA; keeping multiple copies of
the scale vector is intentionally not used because those copies consume
registers and add instructions without reducing memory traffic.

For a C++ LibTorch comparison without Python dispatch, use:

```sh
make benchmark-libtorch KERNEL_CPU=0 KERNEL_REPEATS=100000
```

This uses pre-created CPU tensors, one intra-op/inter-op thread, 1000 warmup
calls, and reports LibTorch batchnorm FLOPs/TSC-cycle and utilization against
the 32 FLOP/core-cycle ceiling. LibTorch maxpool is reported as comparisons
per TSC-cycle. The custom batchnorm prepared path (reported as
`custom_batchnorm_unrolled`) is measured in the same process and shape for
direct comparison. The public custom API uses this same unrolled apply path.
The benchmark also reports `custom_batchnorm_full`, which includes the custom
API's scale/bias preparation, allocation, and free overhead so it can be
compared fairly with LibTorch's returned-tensor allocation overhead.
The maxpool rows use the same 5x5, stride-1, padding-2 operator on both
implementations. Their work rate is reported in comparison-equivalents per
TSC cycle, including LibTorch's returned-tensor allocation and the custom
public API overhead.

The maxpool implementation uses an AVX-512 channel-vector tile for 5x5,
stride-1, padding-2 pooling. It processes a 9x9 input footprint as a 5x5
output tile with 16-channel ZMM vectors; partial channel blocks and boundary
tiles use the scalar path.

## Channel-strided convolution

Consume C2f channel chunks without a separate chunk-copy pass using
`conv_3x3_strided` in `include/conv2D.h`. See
[STRIDED_CONVOLUTION.md](docs/STRIDED_CONVOLUTION.md) for the API,
correctness tests, and a copy-plus-convolution benchmark.

## Memory kernels

The production FP32 memory kernels use physical `[N][C/16][H][W][16]`
storage, with positive channel counts divisible by 16:

- `universal_concat_f32` in `include/universal_concat.h` joins channel planes.
- `fused_upconcat_f32`, `fused_upconcat_a`, and `fused_upconcat_b` in
  `include/fused_upconcat.h` combine 2x nearest upsampling and skip concatenation.
- `residual_add` in `include/residual_add.h` adds dense blocked spans and
  supports in-place output while preserving the shortcut.

These replace the original NHWC memory-kernel APIs. The functions check
capacities and invalid arguments before writing. For a C2f half, pass
`cv1 + C*H*W` floats. Other kernels retain their existing layout contracts.
See the [API, build and verification guide](docs/memory_kernels.md).

Build and test with `make memory-kernels check-memory`. On AFS, set
`MEMORY_BUILD=/scratch/zihengli-memory-replace-20261031/afs-build`
to keep generated files outside the nearly full account volume.

Materialized channel splitting is available through `chunk2_nhwc_f32` in `include/chunk_nhwc.h`; either output may be omitted. See [usage and measurement notes](docs/chunk_nhwc.md). Build and verify with `make chunk check-chunk`; benchmark with `make bench-chunk` (one thread, CPU 2). Existing zero-copy model paths remain the default.
