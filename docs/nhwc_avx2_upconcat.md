# Fused NHWC upsample and concatenate: VL_MIXED

The kernel combines 2x nearest-neighbor upsampling and channel concatenation.
It uses **256-bit vector moves** and **AVX-512VL's extra YMM registers**.
AVX-512F and AVX-512VL are required; there is no AVX2 fallback or CPU dispatcher.

## API and store order

All buffers are contiguous FP32 NHWC. Channels are `[upsampled, skip]`.
The caller allocates the output; the kernel allocates nothing.

| Function | Up channels | Skip channels | Output channels | YOLOv8n layers |
|---|---:|---:|---:|---|
| `fused_upconcat_a` | 256 | 128 | 384 | 10+11 |
| `fused_upconcat_b` | 128 | 64 | 192 | 13+14 |

Each call processes one image. Both stages use QUAD traversal: a source pixel
supplies a 2x2 group of output pixels. Every source value is loaded once and
used for four output stores.

- **Stage A:** copy the first 128 channels to all four pixels. Keep the second
  128 channels in registers while interleaving their stores with each pixel's
  skip copy. This is partial interleaving.
- **Stage B:** keep all 128 source channels in registers and write
  `A up, A skip, B up, B skip, C up, C skip, D up, D skip`.

The source block occupies 16 YMM registers. The additional registers let skip
copies run while source data remains live. Empty compiler barriers keep the
generated store groups in order; they are not CPU fences.

```c
#include "fused_upconcat.h"

// up: [H][W][256], skip: [2*H][2*W][128]
// out: [2*H][2*W][384], allocated by the caller
fused_upconcat_a(up, skip, out, H, W);
```

Buffers must be large enough, and output must not overlap either input.
Unaligned addresses are supported. Integer moves preserve all FP32 bit
patterns, including NaN payloads and signed zero. The public API and output
layout are unchanged from the earlier AVX2 implementation.

## Build and test

Run from the repository root on a compatible x86 CPU:

```sh
make kernels
make test
make check-upconcat PYTHON=/path/to/python-with-torch
make benchmark-upconcat PYTHON=/path/to/python-with-torch
```

The fused kernel has its own `UPCONCAT_CFLAGS`, independent of convolution:

```text
-O3 -std=c11 -march=native -mavx512f -mavx512vl -mprefer-vector-width=256
```

The Makefile also enables compiler warnings. This build is tuned to the
compilation host; build and run on ece020 or an equivalent target. The verified
compiler is GCC 8.5.0.

`make test` needs Python and GNU `objdump`, but does not need PyTorch.
Its 48 native cases cover both stages, the tested image sizes, empty and
rectangular shapes, unaligned addresses, raw floating-point bits, unchanged
inputs and guard bytes. The assembly check verifies high YMM register use,
absence of ZMM instructions and absence of vector stack spills. It can also
be run separately:

```sh
make check-upconcat-isa
```

The PyTorch checks and benchmark require Linux, Python 3.9 or later, CPU
PyTorch, an available CPU 0 and one thread. `PYTHON` defaults to `python3` and
can select an existing environment. To create an environment matching the
earlier PyTorch 2.5.1 CPU setup:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r benchmarks/requirements_upconcat.txt
make check-upconcat PYTHON=.venv/bin/python
make benchmark-upconcat PYTHON=.venv/bin/python
```

PyTorch uses NCHW dimension labels with `channels_last` storage, giving the C
kernel NHWC bytes without a timed layout conversion. Its 12 checks compare all
output bits for 640, 1024 and 2048 image sizes, both stages and two alignments.

## Benchmark outputs

The benchmark labels the kernel `VL_MIXED` and writes `run-1.csv`, `run-2.csv`,
`run-3.csv` and `result.md` under `benchmarks/results/nhwc_vl/`. Earlier AVX2
results remain separate. Generated results and binaries are ignored by Git.
`make clean` removes build outputs and keeps measurements.

Each of three processes uses 21 rounds per case, four warmups and five timed
calls per round, with randomized case and method order. It compares PyTorch
nearest upsampling plus concatenate with the fused C kernel. Buffers are
preallocated; PyTorch also uses a preallocated upsample intermediate. These
times cover the two operations, not the whole YOLOv8n model.

The [October 22 report](nhwc_avx2_ece020_20261022.md) and the
[earlier 512-bit comparison](AVX2_AVX512_comparison.md) describe older kernels.
Their measurement tables do not describe the current VL_MIXED implementation.
