CC ?= gcc
CFLAGS ?= -O2 -std=c11 -mavx2 -mfma
LDLIBS ?= -lm

TEST_BIN = test_correctness.x
TEST_SRC = tests/test_correctness.c
HEADERS = include/conv2D.h include/silu.h include/mmm_4x24.h
KERNEL_LIB = kernels/libfastkernels.so
KERNEL_SRC = kernels/batchnorm_maxpool_skeleton.c
KERNEL_HEADERS = include/batchnorm.h include/maxpool.h

.PHONY: all test kernels clean

all: $(TEST_BIN)

$(TEST_BIN): $(TEST_SRC) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) $(LDLIBS)

kernels: $(KERNEL_LIB)

$(KERNEL_LIB): $(KERNEL_SRC) $(KERNEL_HEADERS)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $(KERNEL_SRC)

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -f $(TEST_BIN) $(KERNEL_LIB)