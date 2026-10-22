CC ?= gcc
CFLAGS ?= -std=c11 -march=native -O3  -mavx2 -mfma 
LDLIBS ?= -lm

TEST_BIN = test_correctness.x benchmark.x
HEADERS = include/conv2D_512.h include/silu.h include/mmm_4x48.h

.PHONY: all test clean

all: $(TEST_BIN)

test_correctness.x: tests/test_correctness.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

benchmark.x: tests/benchmark.c $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

test: $(TEST_BIN)
	./test_correctness.x
	./benchmark.x

clean:
	rm -f $(TEST_BIN)