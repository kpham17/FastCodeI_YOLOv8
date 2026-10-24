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
	rm -f $(TEST_BIN) benchmark_conv_strided.x $(KERNEL_LIB) $(UPCONCAT_LIB) $(CONCAT_OBJ) $(CONCAT_LIB) $(CONCAT_TEST_BIN) $(RESIDUAL_OBJ) $(RESIDUAL_LIB) $(RESIDUAL_TEST_BIN)

# Standalone concat keeps fallback code free of global -march/AVX flags.
CONCAT_CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -mtune=skylake-avx512
CONCAT_CPPFLAGS = -Iinclude -Itests
CONCAT_SRC = kernels/universal_concat_nhwc.c
CONCAT_HEADER = include/universal_concat_nhwc.h
CONCAT_WORKLOADS = tests/universal_concat_workloads.h
CONCAT_OBJ = kernels/universal_concat_nhwc.o
CONCAT_LIB = kernels/libuniversal_concat.a
CONCAT_TEST_BIN = test_universal_concat.x test_universal_concat_cpp.x test_universal_concat_portable.x test_universal_concat_sanitize.x bench_universal_concat.x

.PHONY: concat check-concat check-concat-portable check-concat-sanitize bench-concat
concat: $(CONCAT_LIB)

$(CONCAT_OBJ): $(CONCAT_SRC) $(CONCAT_HEADER)
	$(CC) $(CONCAT_CFLAGS) $(CONCAT_CPPFLAGS) -c $(CONCAT_SRC) -o $@

$(CONCAT_LIB): $(CONCAT_OBJ)
	$(AR) rcs $@ $<

test_universal_concat.x: tests/test_universal_concat.c $(CONCAT_WORKLOADS) $(CONCAT_HEADER) $(CONCAT_LIB)
	$(CC) $(CONCAT_CFLAGS) $(CONCAT_CPPFLAGS) $< $(CONCAT_LIB) -o $@

test_universal_concat_cpp.x: tests/test_universal_concat_cpp.cpp $(CONCAT_HEADER) $(CONCAT_LIB)
	$(CXX) -std=c++11 -Wall -Wextra -Wpedantic -Iinclude $< $(CONCAT_LIB) -o $@

check-concat: test_universal_concat.x test_universal_concat_cpp.x
	./test_universal_concat.x
	./test_universal_concat_cpp.x

check-concat-portable:
	$(CC) $(CONCAT_CFLAGS) $(CONCAT_CPPFLAGS) -DUC_DISABLE_AVX2 $(CONCAT_SRC) tests/test_universal_concat.c -o test_universal_concat_portable.x
	./test_universal_concat_portable.x

check-concat-sanitize:
	$(CC) -O1 -g -std=c11 -Wall -Wextra -fsanitize=address,undefined -fno-omit-frame-pointer $(CONCAT_CPPFLAGS) $(CONCAT_SRC) tests/test_universal_concat.c -o test_universal_concat_sanitize.x
	ASAN_OPTIONS=detect_leaks=1 ./test_universal_concat_sanitize.x

bench-concat: bench_universal_concat.x
bench_universal_concat.x: benchmarks/bench_universal_concat.c $(CONCAT_WORKLOADS) $(CONCAT_HEADER) $(CONCAT_LIB)
	$(CC) $(CONCAT_CFLAGS) $(CONCAT_CPPFLAGS) $< $(CONCAT_LIB) -o $@

# Extend the existing aggregates without changing their recipes.
all: $(CONCAT_LIB)
kernels: $(CONCAT_LIB)
test: check-concat

# Residual add: keep ISA checks valid by avoiding global -march/AVX flags.
RESIDUAL_CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off
RESIDUAL_CPU ?= 0
RESIDUAL_SRC = kernels/residual_add.c
RESIDUAL_HEADER = include/residual_add.h
RESIDUAL_OBJ = kernels/residual_add.o
RESIDUAL_LIB = kernels/libresidual_add.a
RESIDUAL_TEST_BIN = test_residual_add.x test_residual_add_cpp.x bench_residual_add.x test_residual_add_sanitize.x

.PHONY: residual check-residual bench-residual check-residual-sanitize
residual: $(RESIDUAL_LIB)

$(RESIDUAL_OBJ): $(RESIDUAL_SRC) $(RESIDUAL_HEADER)
	$(CC) $(RESIDUAL_CFLAGS) -Iinclude -c $(RESIDUAL_SRC) -o $@

$(RESIDUAL_LIB): $(RESIDUAL_OBJ)
	$(AR) rcs $@ $<

test_residual_add.x: tests/test_residual_add.c $(RESIDUAL_HEADER) $(RESIDUAL_LIB)
	$(CC) $(RESIDUAL_CFLAGS) -Iinclude $< $(RESIDUAL_LIB) -o $@ -lm

test_residual_add_cpp.x: tests/test_residual_add_cpp.cpp $(RESIDUAL_HEADER) $(RESIDUAL_LIB)
	$(CXX) -std=c++11 -Wall -Wextra -Wpedantic -Iinclude $< $(RESIDUAL_LIB) -o $@

check-residual: test_residual_add.x test_residual_add_cpp.x
	taskset -c $(RESIDUAL_CPU) ./test_residual_add.x
	taskset -c $(RESIDUAL_CPU) ./test_residual_add_cpp.x

bench_residual_add.x: benchmarks/bench_residual_add.c $(RESIDUAL_HEADER) $(RESIDUAL_LIB)
	$(CC) $(RESIDUAL_CFLAGS) -Iinclude $< $(RESIDUAL_LIB) -o $@

bench-residual: bench_residual_add.x
	taskset -c $(RESIDUAL_CPU) ./bench_residual_add.x

# ece020 lacks GCC's libasan; use CC=clang for this optional source check.
check-residual-sanitize:
	$(CC) -O1 -g -std=c11 -Wno-attributes -fno-omit-frame-pointer \
		-fsanitize=address,undefined -fno-fast-math -ffp-contract=off -Iinclude \
		$(RESIDUAL_SRC) tests/test_residual_add.c -o test_residual_add_sanitize.x -lm
	ASAN_OPTIONS=detect_leaks=0 taskset -c $(RESIDUAL_CPU) ./test_residual_add_sanitize.x

all: $(RESIDUAL_LIB)
kernels: $(RESIDUAL_LIB)
test: check-residual
