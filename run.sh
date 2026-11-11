#!/usr/bin/env bash
# Step 1: build and time every kernel. No profilers.
# Usage: ./run.sh [N=4096] [REPS=20]
set -euo pipefail
cd "$(dirname "$0")"
N=${1:-4096}
REPS=${2:-20}

# stage number : name used for output files
KERNELS="0:naive_uncoalesced 1:coalesced 2:shared 17:shared_bank_conflict 18:shared_padded"

make
mkdir -p results
out=results/timing_N${N}.csv
echo "kernel,stage,n,precision,reps,median_ms,tflops,checked_elements,max_abs_error" > "$out"

for entry in $KERNELS; do
  stage=${entry%%:*}
  name=${entry#*:}
  # mm prints a header line and a result line; keep the result line
  echo "$name,$(./build/mm "$stage" "$N" "$REPS" | tail -1)" >> "$out"
done

echo
column -s, -t "$out"
echo
echo "Saved $out"
