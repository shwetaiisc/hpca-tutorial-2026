#!/usr/bin/env bash
# Step 1: build and time every kernel. No profilers.
# Usage: ./run.sh [N=4096] [REPS=20]
set -euo pipefail
cd "$(dirname "$0")"
N=${1:-4096}
REPS=${2:-20}
KERNELS="naive_uncoalesced coalesced shared shared_bank_conflict shared_padded"

make
mkdir -p results
out=results/timing_N${N}.csv
echo "kernel,n,reps,median_ms,gflops,checked_elements,max_abs_error" > "$out"
for k in $KERNELS; do
  ./build/mm "$k" "$N" "$REPS" >> "$out"
done

echo
column -s, -t "$out"
echo
echo "Saved $out"
