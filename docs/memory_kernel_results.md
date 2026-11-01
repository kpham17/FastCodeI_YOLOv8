# Memory-kernel measurements

The table below records the implementation before the warm-L1 investigation.
The current fused implementation now has a width-one flat SIMD traversal.
See [the current diagnosis](memory_l1_diagnosis.md) for its improvement, fresh
measurements and remaining 90% failures. Checked contracts are unchanged.

Prior ece020 results (three processes, five warmed rounds per case):

| Complete checked call, tiny shape | 64-byte output units/core-cycle |
|---|---:|
| Concat, two C16 inputs, H=W=8 | 0.539 |
| Fused stage A, input H=W=1 | 0.431 |
| Fused stage B, input H=W=1 | 0.307 |
| Residual, H=W=8,C=16, out of place | 0.374 |
| Store-only reference | 0.998–0.999 |

The complete public functions have not reached one 64-byte output store per
cycle for all three operations. Checks, traversal and buffer placement remain
part of the measured call. The isolated residual inner loop reached 0.688
stores/cycle; that is not the complete-call result.

Across production cases, the prior geometric mean old-time/new-time ratios
were concat 0.981x, fused 0.940x, residual 1.074x. These are individual kernel
measurements, not whole-model speedups. Original optimized implementations
and paired benchmark code exist only in the recovery/evidence archives.

Pre-replacement recovery snapshot:
`/scratch/zihengli-memory-replace-20261031/before.tar.gz`.

Historical raw measurements and full explanation:
`/scratch/zihengli-memory-replace-20261031/historical-evidence.tar.gz` and
`/scratch/zihengli-memory-replace-20261031/historical-sources.tar.gz`.

Replacement validation logs and source hashes are under
`/scratch/zihengli-memory-replace-20261031`.
