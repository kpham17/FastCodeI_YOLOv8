"""Compare warmed complete calls with the bracketed store-only reference."""
import csv,statistics,sys,math
from pathlib import Path
root=Path(sys.argv[1])
def read(name):return list(csv.DictReader((root/name).open()))
def positive(value):
    try:v=float(value)
    except (ValueError,TypeError):raise SystemExit('Missing or invalid measurement.')
    if not math.isfinite(v) or v<=0:raise SystemExit('Non-finite or non-positive measurement.')
    return v
def calibration(name,run):
    rows=read(name)
    if len(rows)!=21 or {(int(x['run']),int(x['round'])) for x in rows}!={(run,r) for r in range(21)}:
        raise SystemExit('Incomplete or duplicate store-calibration samples: '+name)
    if any(int(x['width_B'])!=64 or int(x['ring_B'])!=16384 for x in rows):
        raise SystemExit('Unexpected calibration store width or footprint.')
    for x in rows:positive(x['core_cycles'])
    return [positive(x['stores_per_cycle']) for x in rows]
before=calibration('l1-store-before.csv',1)
after=calibration('l1-store-after.csv',2)
if min(statistics.median(before),statistics.median(after))<.95:
    raise SystemExit('Store calibration fell below 0.95 stores/cycle; rerun on an idle core and sibling.')
peak=statistics.median(before+after);cases={}
expected={'L1_1x1_A':('upconcat',9216),'L1_1x1_B':('upconcat',4608),
    'L1_add_1536B':('add',1536),'L1_concat_2KiB':('concat',2048),
    'L1_dense_in':('add',8192),'L1_dense_out':('add',12288),
    'L1_fused_832B':('upconcat',832),'L1_tiny':('concat',16384),
    'L1_view_in':('add',12288),'L1_view_out':('add',16384)}
for run in (1,2,3):
    rows=read('l1-public-%d.csv'%run)
    if len(rows)!=5*len(expected):raise SystemExit('Incomplete public benchmark process %d.'%run)
    seen=set()
    for row in rows:
        if row['correctness']!='PASS' or row['output_64B_per_core_cycle']=='NA':
            raise SystemExit('Invalid result or unavailable/non-scheduled core-cycle counter.')
        key=(row['case'],int(row['round']))
        if int(row['run'])!=run or key in seen or not 0<=key[1]<5:
            raise SystemExit('Wrong process, round, or duplicate sample.')
        seen.add(key)
        if expected.get(row['case'])!=(row['kernel'],int(row['tensor_footprint_B'])):
            raise SystemExit('Unexpected case, operation or tensor footprint.')
        positive(row['core_cycles_per_call']);positive(row['output_64B_per_core_cycle'])
        cases.setdefault(row['case'],[]).append(row)
    if seen!={(case,r) for case in expected for r in range(5)}:
        raise SystemExit('Missing public benchmark case/round.')
print('Store reference: %.4f 64-byte stores/core-cycle; 90%% target: %.4f'%(peak,.9*peak))
print('Complete checked calls; >=10 ms warm-up per sample; three processes x five rounds.')
print('case                         tensor bytes   stores/cycle   peak %   >=90%')
for case,rows in sorted(cases.items()):
    rate=statistics.median(float(x['output_64B_per_core_cycle']) for x in rows)
    print('%-29s %7s        %.4f       %5.1f     %s'%(case,rows[0]['tensor_footprint_B'],rate,100*rate/peak,'YES' if rate>=.9*peak else 'NO'))
print('Causes for sub-90% results and PMU evidence: docs/memory_l1_diagnosis.md')
