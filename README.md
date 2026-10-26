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

## Fused NHWC upsample and concatenate

Combine 2x nearest-neighbor upsampling and skip concatenation with
`fused_upconcat_a` and `fused_upconcat_b`. The current VL_MIXED implementation
uses 256-bit moves and AVX-512VL's extra vector registers. It requires
AVX-512F and AVX-512VL, as supported by ece020.

Both stages use QUAD traversal. Stage B interleaves each pixel's upsampled
and skip stores; stage A interleaves its second 128-channel block and skips.
See the [API and testing guide](docs/nhwc_avx2_upconcat.md).
The [earlier AVX2 measurements](docs/nhwc_avx2_ece020_20261022.md) are historical.

## Universal NHWC channel concatenation

Join any positive number of FP32 NHWC inputs with
`universal_concat_nhwc_f32` in `include/universal_concat_nhwc.h`.
The checked implementation uses cached AVX2 with a portable fallback and
covers all eleven standalone YOLOv8n joins at 640/1024/2048 input sizes.
Build with `make concat`; run `make check-concat` and
`make check-concat-portable`. See the [API, benchmark and integration guide](docs/universal_concat.md).

## Standalone bottleneck residual addition

Add FP32 NHWC convolution output and a dense or strided shortcut with
`residual_add_contiguous` / `residual_add_strided`. In-place output is supported;
the shortcut is preserved for C2f concatenation. Production is one C file and
one header. Build with `make residual`; test with `make check-residual`.
See the [API and consistency guide](docs/residual_add.md).

Materialized channel splitting is available through `chunk2_nhwc_f32` in `include/chunk_nhwc.h`; either output may be omitted. See [usage and measurement notes](docs/chunk_nhwc.md). Build and verify with `make chunk check-chunk`; benchmark with `make bench-chunk` (one thread, CPU 2). Existing zero-copy model paths remain the default.
