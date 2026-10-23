# ece020: keep the minimal AVX2 production kernels

This report describes the original measured variants. Its AVX2 stage B used
ROW order. The current production stage B uses QUAD order with the same API;
the measurements below have not been updated for that change.

**Recommendation: retain MINIMAL_AVX2 for both stages and all three image sizes.** Neither full-width AVX-512 candidate provides a repeatable, material improvement that justifies a shape-specific dispatch. The requested experiment is complete; the source baseline was preserved unchanged.

The primary controlled session and a separate confirmation session each contain three independent processes. Every process measured all six shapes with 21 randomized/interleaved rounds, four warmup calls and five timed calls per candidate. Together they contain 2,268 round samples and 11,340 timed complete fused calls. These are warmed-buffer operator measurements, not end-to-end YOLO inference.

## Primary measurements

Times are milliseconds, using the median of three process medians. Each speedup is reference time divided by candidate time; above 1 favors the candidate.

| Image | Site | AVX2 minimal ms | ZMM16 ms | ZMM32 ms | ZMM16/AVX2 speedup | ZMM32/AVX2 speedup | ZMM32/ZMM16 speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| 640 | A | 0.182534 | 0.180462 | 0.177015 | 1.0115 | 1.0312 | 1.0195 |
| 640 | B | 0.371742 | 0.378646 | 0.384816 | 0.9818 | 0.9660 | 0.9840 |
| 1024 | A | 0.502373 | 0.508473 | 0.524041 | 0.9880 | 0.9587 | 0.9703 |
| 1024 | B | 1.692487 | 1.781560 | 1.827564 | 0.9500 | 0.9261 | 0.9748 |
| 2048 | A | 3.506911 | 3.591779 | 3.689218 | 0.9764 | 0.9506 | 0.9736 |
| 2048 | B | 7.373669 | 7.732069 | 7.781647 | 0.9536 | 0.9476 | 0.9936 |

A+B is the sum of the separately measured stage medians above.

| Image | AVX2 A+B ms | ZMM16 A+B ms | ZMM32 A+B ms | ZMM16/AVX2 speedup | ZMM32/AVX2 speedup | ZMM32/ZMM16 speedup |
|---|---:|---:|---:|---:|---:|---:|
| 640 | 0.554276 | 0.559108 | 0.561831 | 0.9914 | 0.9866 | 0.9952 |
| 1024 | 2.194859 | 2.290033 | 2.351605 | 0.9584 | 0.9333 | 0.9738 |
| 2048 | 10.880580 | 11.323848 | 11.470865 | 0.9609 | 0.9485 | 0.9872 |

## Confirmation and variability

The confirmation session used the same benchmark binary and new processes/seeds. It was conducted later in this work session; it is not a different-day or isolated-machine replication.

| Image | AVX2 A+B ms | ZMM16 A+B ms | ZMM32 A+B ms | ZMM16/AVX2 speedup | ZMM32/AVX2 speedup |
|---|---:|---:|---:|---:|---:|
| 640 | 0.550387 | 0.549185 | 0.552068 | 1.0022 | 0.9970 |
| 1024 | 2.137350 | 2.264964 | 2.409862 | 0.9437 | 0.8869 |
| 2048 | 10.832484 | 11.306001 | 11.463399 | 0.9581 | 0.9450 |

The 640 A advantage for ZMM32 shrank from speedup 1.0312 to 1.0205. ZMM16 became slightly faster than ZMM32 for that stage in the confirmation session. These small changes do not justify dispatch complexity. At 640 A+B the candidates are effectively tied; the small winner changes between sessions. At 1024 and 2048 the AVX2 recommendation is consistent.

For example, primary 2048 B process medians are AVX2 7.329100/7.393458/7.373669 ms, ZMM16 7.732069/7.740139/7.723527 ms, and ZMM32 7.756977/7.781647/7.829257 ms. Their between-process ranges are 0.87%, 0.21% and 0.93% of their medians. Smaller shapes show more variation: primary 640 B ranges are 5.16%, 3.68% and 4.44%, respectively; 1024 B ZMM32 varies 6.15%. These are descriptive ranges, not confidence intervals.

Both [primary results](results/RESULTS.md) and [confirmation results](results-confirmation/RESULTS.md) contain every process median, per-stage and A+B spread. Their `process-statistics.csv` files contain within-process p10/p90 for all 54 process/shape/variant combinations per session. The raw `process-1.csv` through `process-3.csv` preserve all samples and execution order. No historical result contributed to any table.

## Correctness and assembly evidence

| Configuration | Bitwise/guard tests | ASan + UBSan | Verified register/ISA properties |
|---|---|---|---|
| MINIMAL_AVX2 | PASS | PASS | ymm0–ymm15; 256-bit SIMD; no EVEX, ZMM or mask-register instructions |
| ZMM16_512 | PASS | PASS | zmm0–zmm15 only; full 512-bit memory moves; 16 source loads before first store |
| ZMM32_512 | PASS | PASS | all zmm0–zmm31; full 512-bit memory moves; 32 source loads before first store, each used for four destinations |

The suite passes 204 cases, including all six exact shapes, full-output `memcmp`, NaN payloads/signaling patterns, infinities, signed zero, subnormals, random bits, varied float-aligned offsets, poison, all canary bytes, protected bounds, read-only input pages and byte snapshots. A deliberately wrong no-op implementation failed first. V1 SHA-256 is `fec59759d8b389ae1e154391681881dcb586e6a589790b19b94bdd196fcacfcd`, identical to the supplied minimal source.

The assembly checker passes both objects and both linked binaries. There are no called helpers, external tail jumps, partial-width ZMM transfers or vector stack spills/reloads in these hot functions. V3 does use ordinary integer stack slots for pointer/loop state, including some row-loop loads; these are real scheduling costs of the supplied implementation, not vector spills. They were retained and documented. Complete disassembly and [assembly proof](results/assembly-proof.txt) are included.

GCC 8.5.0 could compile sanitizer instrumentation but its dynamic ASan/UBSan libraries were missing. The reproducible fallback linked the GCC-instrumented kernel objects using Clang 20.1.8 compiler-rt and compiled the correctness harness with Clang. All 204 cases passed with nonrecovering sanitizer errors. Inline assembly memory accesses are outside ASan instrumentation; the protected allocations and byte checks supplement it. Full [sanitizer log](results/sanitizer.log) and the initial failed GCC link are preserved.

## What these results establish

1. **Vector width:** actual 512-bit memory transfers did not produce a material overall win over this actual minimal AVX2 baseline under this protocol. V1 was built with the requested historical strict-AVX2 flags, not silently changed to a different baseline.
2. **Register capacity:** using all 32 full-width ZMM registers did not improve the complete operator relative to ZMM16 on a repeatable, useful basis. This is an observation about these implementations, not a causal register-capacity experiment.
3. **Source-pixel batching:** V2 batches A=1/B=2 source pixels; V3 batches A=2/B=4. More batching changes scheduling and integer-register pressure, while leaving minimum logical U/S/O bytes unchanged.
4. **Destination ordering:** V1 B uses ROW; V2/V3 B use QUAD. Stage B differences combine width, batching and traversal. A width-only conclusion would require a supplementary matched-traversal control.
5. **Frequency and sharing:** AVX-512 frequency effects are possible but were not measured. Interleaving can carry frequency state between candidates. The node had other active jobs, while CPU 0 was pinned and CPU 16 is its sibling. Provenance and per-process workload snapshots are preserved; no job was stopped and no frequency setting was changed.

Every timed kernel materializes the complete dense NHWC output, including all distinct skip suffixes. Buffers were allocated/first-touched after pinning to CPU 0, and candidates used the same addresses within each process. No allocation, initialization, checksum, correctness check or I/O occurred inside the timed call loop. Ordinary cached stores were used. Logical byte counts are not measured DRAM bandwidth, and no hardware-peak percentage is claimed.

Build commands, GCC version, CPU/socket/cache details, affinity, timestamp and binary hashes are retained in `results/`. Normal and confirmation benchmark hashes agree. The minimal production choice remains the original two-function AVX2 interface; the ZMM candidates remain reproducible experimental alternatives.
