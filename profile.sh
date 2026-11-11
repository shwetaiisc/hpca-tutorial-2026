#!/usr/bin/env bash
# Step 2: profile every kernel with Nsight Systems (timeline) and Nsight Compute (counters).
# Usage: ./profile.sh [N=4096]
# Output: reports/<kernel>.nsys-rep and reports/<kernel>.ncu-rep
set -euo pipefail
cd "$(dirname "$0")"
N=${1:-4096}
# stage number : name used for report files
KERNELS="0:naive_uncoalesced 1:coalesced 2:shared 17:shared_bank_conflict 18:shared_padded"

make
mkdir -p reports

for entry in $KERNELS; do
  stage=${entry%%:*}
  k=${entry#*:}
  echo "== $k (stage $stage)"

  # Nsight Systems: whole program (copies, kernel launches, NVTX range). 5 timed launches.
  nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite true \
       -o "reports/$k" ./build/mm "$stage" "$N" 5 > /dev/null

  # Nsight Compute: one timed launch, every section.
  #   --nvtx-include  skip the warm-up launches, profile inside "measured_gemm"
  #   (only 1 timed launch runs inside the range, so ncu profiles exactly one kernel)
  #   --clock-control none  keep the GPU's normal boost clock, so ncu time matches run.sh
  ncu --set full --clock-control none --nvtx --nvtx-include "measured_gemm/" \
      --force-overwrite -o "reports/$k" ./build/mm "$stage" "$N" 1 > /dev/null
done

echo
echo "Reports in reports/:"
ls -1 reports/*.ncu-rep reports/*.nsys-rep
