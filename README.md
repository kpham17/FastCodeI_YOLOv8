# FastCodeI_YOLOv8
YOLOv8 implementation

## Batchnorm and maxpooling skeletons

The C APIs use NHWC tensors:

- [`include/batchnorm.h`](include/batchnorm.h)
- [`include/maxpool.h`](include/maxpool.h)

The current implementations are explicit skeletons that return an error until
the kernels are designed. Build the shared library with:

```sh
make kernels
```

After implementing the two functions, compare runtime and analytic FLOP
estimates against PyTorch:

```sh
.venv/bin/python benchmarks/compare_kernels.py \
    --library kernels/libfastkernels.so
```

The benchmark uses batchnorm inference with running statistics and 2x2
maxpooling with stride 2. Customize the workload with `--shape N H W C`,
`--warmups`, and `--repeats`.

The NHWC maxpool API keeps the optimized AVX-512 tile restricted to `C=1`;
interleaved NHWC data with `C=3` must not be passed to that tile because its
16-float loads would mix channels and spatial positions. For performance
experiments that intentionally use planar storage, call
`fast_maxpool2d_planar`, which applies the optimized single-channel tile to
each `[H][W]` plane. Any NHWC-to-planar conversion should be measured
separately or included explicitly when reporting end-to-end performance.

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
