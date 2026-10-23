CC ?= gcc
CFLAGS ?= -std=c11 -march=native -O3  -mavx2 -mfma 
LDLIBS ?= -lm

TEST_BIN = test_correctness.x test_conv_strided.x benchmark.x
HEADERS = include/conv2D.h include/silu.h include/mmm_4x48.h
KERNEL_LIB = kernels/libfastkernels.so
KERNEL_SRC = kernels/batchnorm_maxpool_skeleton.c
KERNEL_HEADERS = include/batchnorm.h include/maxpool.h

.PHONY: all test kernels benchmark-strided clean

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
	rm -f $(TEST_BIN) benchmark_conv_strided.x $(KERNEL_LIB)
