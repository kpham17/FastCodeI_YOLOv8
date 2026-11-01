# Functional memory kernels. On AFS, set MEMORY_BUILD to a scratch directory.
MEMORY_BUILD ?= build/memory
MEMORY_CPU ?= 0
MEMORY_CFLAGS ?= -O3 -std=c11 -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off -mtune=skylake-avx512
MEMORY_SRC = kernels/universal_concat.c kernels/fused_upconcat.c kernels/residual_add.c
MEMORY_HDR = include/memory_layout.h include/universal_concat.h include/fused_upconcat.h include/residual_add.h kernels/memory_kernels_internal.h
MEMORY_OBJ = $(patsubst kernels/%.c,$(MEMORY_BUILD)/%.o,$(MEMORY_SRC))
MEMORY_LIB = $(MEMORY_BUILD)/libmemory_kernels.a
MEMORY_INC = -Iinclude -Itests -Ibenchmarks
MEMORY_TEST = tests/test_memory_kernels.c tests/memory_kernel_fixtures.h tests/universal_concat_workloads.h
.PHONY: memory-kernels check-memory check-memory-portable check-memory-sanitize bench-memory clean-memory
$(MEMORY_BUILD):
	mkdir -p $@
$(MEMORY_OBJ): $(MEMORY_BUILD)/%.o: kernels/%.c $(MEMORY_HDR) | $(MEMORY_BUILD)
	$(CC) $(MEMORY_CFLAGS) $(MEMORY_INC) -c $< -o $@
$(MEMORY_LIB): $(MEMORY_OBJ)
	$(AR) rcs $@ $^
memory-kernels: $(MEMORY_LIB)
$(MEMORY_BUILD)/test_memory_kernels.x: $(MEMORY_TEST) $(MEMORY_HDR) $(MEMORY_LIB)
	$(CC) $(MEMORY_CFLAGS) $(MEMORY_INC) $< $(MEMORY_LIB) -lm -o $@
$(MEMORY_BUILD)/test_memory_kernels_cpp.x: tests/test_memory_kernels_cpp.cpp $(MEMORY_HDR) $(MEMORY_LIB)
	$(CXX) -std=c++11 -Wall -Wextra -Wpedantic -Iinclude $< $(MEMORY_LIB) -o $@
check-memory: $(MEMORY_BUILD)/test_memory_kernels.x $(MEMORY_BUILD)/test_memory_kernels_cpp.x
	taskset -c $(MEMORY_CPU) $(MEMORY_BUILD)/test_memory_kernels.x
	$(MEMORY_BUILD)/test_memory_kernels_cpp.x
check-memory-portable: $(MEMORY_TEST) $(MEMORY_SRC) $(MEMORY_HDR) | $(MEMORY_BUILD)
	$(CC) $(MEMORY_CFLAGS) $(MEMORY_INC) -DMEMORY_PORTABLE tests/test_memory_kernels.c $(MEMORY_SRC) -lm -o $(MEMORY_BUILD)/test_memory_kernels_portable.x
	taskset -c $(MEMORY_CPU) $(MEMORY_BUILD)/test_memory_kernels_portable.x
check-memory-sanitize: $(MEMORY_TEST) $(MEMORY_SRC) $(MEMORY_HDR) | $(MEMORY_BUILD)
	clang -O1 -g -std=c11 -Wall -Wextra -Wpedantic -fno-fast-math -ffp-contract=off -fsanitize=address,undefined -fno-omit-frame-pointer $(MEMORY_INC) tests/test_memory_kernels.c $(MEMORY_SRC) -lm -o $(MEMORY_BUILD)/test_memory_kernels_sanitize.x
	ASAN_OPTIONS=detect_leaks=1 taskset -c $(MEMORY_CPU) $(MEMORY_BUILD)/test_memory_kernels_sanitize.x
$(MEMORY_BUILD)/bench_memory_kernels.x: benchmarks/bench_memory_kernels.c benchmarks/memory_kernel_timer.h tests/memory_kernel_fixtures.h tests/universal_concat_workloads.h $(MEMORY_LIB)
	$(CC) $(MEMORY_CFLAGS) $(MEMORY_INC) $< $(MEMORY_LIB) -lm -o $@
$(MEMORY_BUILD)/bench_l1_store.x: benchmarks/bench_l1_store.c benchmarks/memory_kernel_timer.h | $(MEMORY_BUILD)
	$(CC) $(MEMORY_CFLAGS) $(MEMORY_INC) $< -o $@
bench-memory: $(MEMORY_BUILD)/bench_memory_kernels.x $(MEMORY_BUILD)/bench_l1_store.x
.PHONY: bench-memory-l1
bench-memory-l1: bench-memory
	taskset -c $(MEMORY_CPU) $(MEMORY_BUILD)/bench_l1_store.x 1 > $(MEMORY_BUILD)/l1-store-before.csv
	for run in 1 2 3; do env -u MEMORY_BENCH_KERNEL MEMORY_BENCH_FILTER=L1 taskset -c $(MEMORY_CPU) $(MEMORY_BUILD)/bench_memory_kernels.x warm $$run > $(MEMORY_BUILD)/l1-public-$$run.csv || exit; done
	taskset -c $(MEMORY_CPU) $(MEMORY_BUILD)/bench_l1_store.x 2 > $(MEMORY_BUILD)/l1-store-after.csv
	$(PYTHON) benchmarks/summarize_memory_l1.py $(MEMORY_BUILD)
clean-memory:
	rm -f $(MEMORY_OBJ) $(MEMORY_LIB) $(MEMORY_BUILD)/test_memory_kernels*.x $(MEMORY_BUILD)/bench_memory_kernels.x $(MEMORY_BUILD)/bench_l1_store.x

# Existing functional entry targets now use the one consolidated implementation.
.PHONY: concat residual upconcat check-concat check-residual check-upconcat check-concat-portable check-concat-sanitize check-residual-sanitize bench-concat bench-residual benchmark-upconcat
concat residual upconcat: memory-kernels
check-concat check-residual check-upconcat: check-memory
check-concat-portable: check-memory-portable
check-concat-sanitize check-residual-sanitize: check-memory-sanitize
bench-concat: bench-memory
bench-residual: bench-memory
	MEMORY_BENCH_KERNEL=add taskset -c $(MEMORY_CPU) $(MEMORY_BUILD)/bench_memory_kernels.x warm 1
benchmark-upconcat: bench-memory
	for run in 1 2 3; do MEMORY_BENCH_KERNEL=upconcat taskset -c $(MEMORY_CPU) $(MEMORY_BUILD)/bench_memory_kernels.x warm $$run || exit; done
all kernels: memory-kernels
test: check-memory
clean: clean-memory
