CC ?= gcc
CFLAGS ?= -std=c11 -march=native -O3  -mavx2 -mfma 
LDLIBS ?= -lm

TEST_BIN = test_correctness.x benchmark.x
HEADERS = include/conv2D_512.h include/silu.h include/mmm_4x48.h
HEADERS = include/conv2D.h include/silu.h include/mmm_4x24.h
KERNEL_LIB = kernels/libfastkernels.so
KERNEL_SRC = kernels/batchnorm_maxpool_skeleton.c
KERNEL_HEADERS = include/batchnorm.h include/maxpool.h

.PHONY: all test kernels clean

all: $(TEST_BIN)

test_correctness.x: tests/test_correctness.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

benchmark.x: tests/benchmark.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

kernels: $(KERNEL_LIB)

$(KERNEL_LIB): $(KERNEL_SRC) $(KERNEL_HEADERS)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $(KERNEL_SRC)

test: $(TEST_BIN)
	./test_correctness.x
	./benchmark.x

clean:
	rm -f $(TEST_BIN) $(KERNEL_LIB)