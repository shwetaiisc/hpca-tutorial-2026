#!/usr/bin/env python3
"""List the shared-memory instructions that cause bank conflicts in an ncu report.

Usage: python3 tools/bank_conflicts.py reports/shared_bank_conflict.ncu-rep
Reads the per-instruction (SASS) source page of the report and prints every
instruction with excess shared-memory wavefronts, plus the CUDA line it came from.
"""
import csv, io, subprocess, sys

rep = sys.argv[1]
def page(view):
    out = subprocess.run(['ncu', '--import', rep, '--page', 'source', '--print-source', view, '--csv'],
                         capture_output=True, text=True, check=True).stdout
    rows = list(csv.reader(io.StringIO(out)))
    i = next(i for i, r in enumerate(rows) if r and r[0] in ('Address', 'Line No'))
    return rows[i], rows[i + 1:]

def num(s):
    try: return float(s.replace(',', ''))
    except ValueError: return 0.0

head, rows = page('sass')
col = {h: j for j, h in enumerate(head)}
total_ideal = total_excess = 0.0
hits = []
for r in rows:
    if len(r) < len(head): continue
    ideal, excess = num(r[col['L1 Wavefronts Shared Ideal']]), num(r[col['L1 Wavefronts Shared Excessive']])
    total_ideal += ideal; total_excess += excess
    if excess > 0:
        hits.append((excess, r[col['Address']], r[col['Source']].strip(), r[col['L1 Conflicts Shared N-Way']], ideal))

print(f'{rep}: shared wavefronts ideal {total_ideal:,.0f}, excess {total_excess:,.0f}'
      + (f' ({(total_ideal + total_excess) / total_ideal:.1f}x ideal)' if total_ideal else ''))
if not hits:
    print('No instruction has bank conflicts.')
else:
    print(f'{"excess wavefronts":>18}  {"n-way":>6}  address  instruction')
    for excess, addr, src, nway, ideal in sorted(hits, reverse=True):
        print(f'{excess:18,.0f}  {nway:>6}  {addr:>7}  {src}')
