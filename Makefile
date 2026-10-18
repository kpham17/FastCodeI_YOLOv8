CC ?= gcc
CFLAGS ?= -O2 -std=c11 -mavx2 -mfma
LDLIBS ?= -lm

TEST_BIN = test_correctness.x
TEST_SRC = tests/test_correctness.c
HEADERS = include/conv2D.h include/silu.h include/mmm_4x24.h

.PHONY: all test clean

all: $(TEST_BIN)

$(TEST_BIN): $(TEST_SRC) $(HEADERS)
	$(CC) $(CFLAGS) -o $@ $(TEST_SRC) $(LDLIBS)

test: $(TEST_BIN)
	./$(TEST_BIN)

clean:
	rm -f $(TEST_BIN)