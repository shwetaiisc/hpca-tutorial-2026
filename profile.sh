#!/usr/bin/env bash
# Step 2: profile every kernel with Nsight Systems (timeline) and Nsight Compute (counters).
# Usage: ./profile.sh [N=4096]
# Output: reports/<kernel>.nsys-rep and reports/<kernel>.ncu-rep
set -euo pipefail
cd "$(dirname "$0")"
N=${1:-4096}
KERNELS="naive_uncoalesced coalesced shared shared_bank_conflict shared_padded"

make
mkdir -p reports

for k in $KERNELS; do
  echo "== $k"

  # Nsight Systems: whole program (copies, kernel launches, NVTX range). 5 timed launches.
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite true \
       -o "reports/$k" ./build/mm "$k" "$N" 5 > /dev/null

  # Nsight Compute: one timed launch, every section.
  #   --nvtx-include  skip the warm-up launches, profile inside "measured_gemm"
  #   --launch-count 1  one kernel is enough; ncu replays it many times
  #   --clock-control none  keep the GPU's normal boost clock, so ncu time matches run.sh
  ncu --set full --clock-control none --nvtx --nvtx-include "measured_gemm/" \
      --launch-count 1 --force-overwrite -o "reports/$k" ./build/mm "$k" "$N" 1 > /dev/null
done

echo
echo "Reports in reports/:"
ls -1 reports/*.ncu-rep reports/*.nsys-rep
