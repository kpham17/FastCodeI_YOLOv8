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

## Channel-strided convolution

Consume C2f channel chunks without a separate chunk-copy pass using
`conv_3x3_strided` in `include/conv2D.h`. See
[STRIDED_CONVOLUTION.md](docs/STRIDED_CONVOLUTION.md) for the API,
correctness tests, and a copy-plus-convolution benchmark.

## Fused NHWC AVX2 upsample and concatenate

Combine 2x nearest-neighbor upsampling and skip concatenation in one AVX2 pass
with `fused_upconcat_a` and `fused_upconcat_b`. See the
[API and testing guide](docs/nhwc_avx2_upconcat.md) and
[ece020 benchmark results](docs/nhwc_avx2_ece020_20261022.md).
