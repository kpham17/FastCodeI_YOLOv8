CC ?= gcc
CFLAGS ?= -std=c11 -march=native -O3 -mavx2 -mfma -mavx512f
LDLIBS ?= -lm
PYTHON ?= python3
TEST_BIN = test_correctness.x test_conv_strided.x benchmark.x
HEADERS = include/conv2D.h include/silu.h include/mmm_4x48.h
KERNEL_LIB = kernels/libfastkernels.so
KERNEL_SRC = kernels/batchnorm_maxpool_skeleton.c
KERNEL_HEADERS = include/batchnorm.h include/maxpool.h

.PHONY: all test kernels benchmark-strided check-upconcat benchmark-upconcat clean

all: $(TEST_BIN)

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

kernels: $(KERNEL_LIB)

$(KERNEL_LIB): $(KERNEL_SRC) $(KERNEL_HEADERS)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $(KERNEL_SRC)

test: $(TEST_BIN)
	./test_correctness.x
	./test_conv_strided.x
	./benchmark.x

clean:
	rm -f $(TEST_BIN) benchmark_conv_strided.x $(KERNEL_LIB) $(CHUNK_OBJ) $(CHUNK_LIB) $(CHUNK_TEST_BIN)

# Materialized NHWC channel halves; ece020 AVX2 / AVX-512F target.
CHUNK_CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -mtune=skylake-avx512
CHUNK_CPU ?= 2
CHUNK_SRC = kernels/chunk_nhwc.c
CHUNK_HEADER = include/chunk_nhwc.h
CHUNK_OBJ = kernels/chunk_nhwc.o
CHUNK_LIB = kernels/libchunk_nhwc.a
CHUNK_TEST_BIN = test_chunk_nhwc.x test_chunk_nhwc_cpp.x test_chunk_nhwc_sanitize.x bench_chunk_nhwc.x bench_chunk_consumers.x

.PHONY: chunk check-chunk check-chunk-sanitize bench-chunk bench-chunk-consumers
chunk: $(CHUNK_LIB)

$(CHUNK_OBJ): $(CHUNK_SRC) $(CHUNK_HEADER)
	$(CC) $(CHUNK_CFLAGS) -Iinclude -c $< -o $@

$(CHUNK_LIB): $(CHUNK_OBJ)
	$(AR) rcs $@ $<

test_chunk_nhwc.x: tests/test_chunk_nhwc.c $(CHUNK_HEADER) $(CHUNK_LIB)
	$(CC) $(CHUNK_CFLAGS) -Iinclude $< $(CHUNK_LIB) -o $@

test_chunk_nhwc_cpp.x: tests/test_chunk_nhwc_cpp.cpp $(CHUNK_HEADER) $(CHUNK_LIB)
	$(CXX) -std=c++11 -Wall -Wextra -Wpedantic -Iinclude $< $(CHUNK_LIB) -o $@

check-chunk: test_chunk_nhwc.x test_chunk_nhwc_cpp.x
	taskset -c $(CHUNK_CPU) ./test_chunk_nhwc.x
	taskset -c $(CHUNK_CPU) ./test_chunk_nhwc_cpp.x

bench_chunk_nhwc.x: benchmarks/bench_chunk_nhwc.c $(CHUNK_HEADER) $(CHUNK_LIB)
	$(CC) $(CHUNK_CFLAGS) -Iinclude $< $(CHUNK_LIB) -o $@

bench-chunk: bench_chunk_nhwc.x
	taskset -c $(CHUNK_CPU) ./bench_chunk_nhwc.x 1

bench_chunk_consumers.x: benchmarks/bench_chunk_consumers.c $(CHUNK_HEADER) $(CHUNK_LIB) include/conv2D.h include/mmm_4x48.h include/mmm_4x24.h include/silu.h
	$(CC) $(CFLAGS) -fno-fast-math -ffp-contract=off -Iinclude $< $(CHUNK_LIB) -o $@ $(LDLIBS)

bench-chunk-consumers: bench_chunk_consumers.x
	taskset -c $(CHUNK_CPU) ./bench_chunk_consumers.x 1

check-chunk-sanitize:
	$(CC) -std=c11 -O1 -g -mtune=skylake-avx512 -fno-omit-frame-pointer \
		-fsanitize=address,undefined -Iinclude $(CHUNK_SRC) tests/test_chunk_nhwc.c -o test_chunk_nhwc_sanitize.x
	ASAN_OPTIONS=detect_leaks=1 taskset -c $(CHUNK_CPU) ./test_chunk_nhwc_sanitize.x

all: $(CHUNK_LIB)
kernels: $(CHUNK_LIB)
test: check-chunk

include memory_kernels.mk
