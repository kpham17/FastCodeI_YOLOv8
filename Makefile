CC ?= gcc
CFLAGS ?= -std=c11 -march=native -O3  -mavx2 -mfma 
LDLIBS ?= -lm
PYTHON ?= python3
# VL_MIXED uses 256-bit moves and AVX-512VL's extra YMM registers.
UPCONCAT_CFLAGS ?= -O3 -std=c11 -march=native -mavx512f -mavx512vl -mprefer-vector-width=256 -Wall -Wextra -Wpedantic
UPCONCAT_LIB = kernels/libfused_upconcat.so
UPCONCAT_SRC = kernels/fused_upconcat.c
UPCONCAT_HEADERS = include/fused_upconcat.h

TEST_BIN = test_correctness.x test_conv_strided.x test_fused_upconcat.x benchmark.x
HEADERS = include/conv2D.h include/silu.h include/mmm_4x48.h
KERNEL_LIB = kernels/libfastkernels.so
KERNEL_SRC = kernels/batchnorm_maxpool_skeleton.c
KERNEL_HEADERS = include/batchnorm.h include/maxpool.h

.PHONY: all test kernels benchmark-strided check-upconcat check-upconcat-isa benchmark-upconcat clean

all: $(TEST_BIN) $(UPCONCAT_LIB)

test_correctness.x: tests/test_correctness.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

test_conv_strided.x: tests/test_conv_strided.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

benchmark_conv_strided.x: tests/benchmark_conv_strided.c include/conv2D.h include/mmm_4x48.h
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

benchmark-strided: benchmark_conv_strided.x
	./benchmark_conv_strided.x

benchmark.x: tests/benchmark.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

kernels: $(KERNEL_LIB) $(UPCONCAT_LIB)

$(UPCONCAT_LIB): $(UPCONCAT_SRC) $(UPCONCAT_HEADERS)
	$(CC) $(UPCONCAT_CFLAGS) -Iinclude -shared -fPIC $(UPCONCAT_SRC) -o $@

test_fused_upconcat.x: tests/test_fused_upconcat.c $(UPCONCAT_SRC) $(UPCONCAT_HEADERS)
	$(CC) $(UPCONCAT_CFLAGS) -Iinclude -o $@ tests/test_fused_upconcat.c $(UPCONCAT_SRC)

check-upconcat-isa: $(UPCONCAT_LIB)
	$(PYTHON) tests/check_fused_upconcat_isa.py

check-upconcat: $(UPCONCAT_LIB) check-upconcat-isa
	$(PYTHON) benchmarks/compare_fused_upconcat.py --check

benchmark-upconcat: $(UPCONCAT_LIB)
	$(PYTHON) benchmarks/compare_fused_upconcat.py --run 1
	$(PYTHON) benchmarks/compare_fused_upconcat.py --run 2
	$(PYTHON) benchmarks/compare_fused_upconcat.py --run 3
	$(PYTHON) benchmarks/compare_fused_upconcat.py --report

$(KERNEL_LIB): $(KERNEL_SRC) $(KERNEL_HEADERS)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $(KERNEL_SRC)

test: $(TEST_BIN) check-upconcat-isa
	./test_correctness.x
	./test_conv_strided.x
	./test_fused_upconcat.x
	./benchmark.x

clean:
	rm -f $(TEST_BIN) benchmark_conv_strided.x $(KERNEL_LIB) $(UPCONCAT_LIB)
