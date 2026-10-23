# Fused NHWC AVX2 upsample and concatenate

Imported from the supplied `nhwc_avx2_code.zip`. The C implementation and public
header are preserved byte for byte. The Python harness keeps the supplied
correctness checks and timing protocol, with paths adapted to this repository.

| File | Purpose |
|---|---|
| `include/fused_upconcat.h` | Public C API and buffer contract |
| `kernels/fused_upconcat.c` | AVX2 implementation |
| `tests/test_fused_upconcat.c` | Native bit-pattern and buffer-boundary checks |
| `benchmarks/compare_fused_upconcat.py` | PyTorch checks and three-run benchmark |
| `docs/nhwc_avx2_ece020_20261022.md` | Verified ece020 measurements |

The kernel performs 2x nearest-neighbor upsampling and channel concatenation in
one pass. All buffers use contiguous FP32 NHWC storage. Channels are ordered
`[upsampled, skip]`. The caller allocates the output; the kernel allocates nothing.

- `fused_upconcat_a`: 256 upsample channels + 128 skip channels; output 384.
  This corresponds to YOLOv8n layers 10+11.
- `fused_upconcat_b`: 128 upsample channels + 64 skip channels; output 192.
  This corresponds to YOLOv8n layers 13+14.

```c
#include "fused_upconcat.h"

// up: [H][W][256], skip: [2*H][2*W][128]
// out: [2*H][2*W][384], allocated by the caller
fused_upconcat_a(up, skip, out, H, W);
```

Buffers must be large enough; output must not overlap either input. The API
processes one image per call. Unaligned addresses are supported. Integer vector
moves preserve all FP32 bit patterns, including NaN payloads and signed zero.
An AVX2-capable x86 CPU is required.

## Build and test

Run commands from the repository root:

```sh
make kernels                    # Includes kernels/libfused_upconcat.so
make test                       # Includes native AVX2 checks; no PyTorch needed
make check-upconcat PYTHON=/path/to/python-with-torch
make benchmark-upconcat PYTHON=/path/to/python-with-torch
```

The AVX2 kernel uses its own `UPCONCAT_CFLAGS`, preserving the archive's
`-march=native -mno-avx512f -mno-avx512vl` settings. Existing convolution flags
remain independent. To build a portable AVX2 binary rather than one tuned to
the compilation host, override the flags explicitly:

```sh
make kernels/libfused_upconcat.so UPCONCAT_CFLAGS='-O3 -std=c11 -mavx2 -Wall -Wextra -Wpedantic'
```

The Python check/benchmark requires Linux, Python 3.9 or later, CPU PyTorch,
an available CPU 0, and one thread. The Python interpreter defaults to `python3`;
`PYTHON` selects an existing environment. An optional environment matching the
supplied PyTorch 2.5.1 CPU measurements can be created with:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r benchmarks/requirements_upconcat.txt
make check-upconcat PYTHON=.venv/bin/python
make benchmark-upconcat PYTHON=.venv/bin/python
```

PyTorch uses NCHW dimension labels with `channels_last`
storage, so the C implementation receives NHWC bytes without a layout conversion.

The native checks cover both kernels, rectangular and empty dimensions, raw
floating-point bit patterns, unaligned addresses, unchanged inputs and guard
bytes. The supplied Python checks additionally compare all output bits against
PyTorch for 640, 1024 and 2048 image sizes and both YOLO stages.

## Benchmark outputs

`make benchmark-upconcat` runs the supplied protocol in three separate
processes, then writes `run-1.csv`, `run-2.csv`, `run-3.csv` and `result.md`
under `benchmarks/results/nhwc_avx2/`. Generated outputs and compiled binaries
are ignored by Git. `make clean` removes build outputs and keeps measurements.

Timing uses 21 rounds per case, four warmups and five timed calls per round,
with randomized case/path order. It compares preallocated PyTorch nearest
upsampling plus concatenate against the fused AVX2 function. Both paths use
preallocated output; PyTorch also uses a preallocated upsample intermediate.
These timings cover the two operations, rather than the complete model.

A fresh verification of this integration is recorded in
[the ece020 report](nhwc_avx2_ece020_20261022.md).
