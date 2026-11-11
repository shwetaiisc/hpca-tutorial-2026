# HPCA 2026 tutorial: matrix multiply and the GPU memory system

One CUDA file, five kernels, two scripts. Each kernel changes one thing about how
threads touch memory, and the profiler shows why the time changes.

| Stage | Report name | Kernel | What changes | Lesson |
|---|---|---|---|---|
| 0 | `naive_uncoalesced` | `naive_uncoalesced` | `threadIdx.x` picks the row of C | neighbouring threads load addresses N floats apart: uncoalesced |
| 1 | `coalesced` | `naive_coalesced` | `threadIdx.x` picks the column | a warp loads 32 consecutive floats: coalesced |
| 2 | `shared` | `shared_tile` | 32×32 tiles of A and B in shared memory | each global load is reused 32 times; no bank conflicts |
| 17 | `shared_bank_conflict` | `shared_tile_pitch`, pitch 32 | B tile stored transposed | a warp reads one bank 32 times: 32-way bank conflict |
| 18 | `shared_padded` | `shared_tile_pitch`, pitch 33 | one padding column | the reads spread across all 32 banks |

Stages 17 and 18 run the same compiled kernel; only the pitch argument (32 or 33)
differs. Compare them for a clean before/after on bank conflicts. (The stage numbers
come from a larger version of this code with more kernels.)

## Requirements

- An NVIDIA GPU and the CUDA Toolkit (nvcc, Nsight Systems `nsys`, Nsight Compute `ncu`).
- Tested on an RTX A6000 (sm_86) with CUDA 13.0.
- `ncu` needs access to GPU performance counters. If it reports `ERR_NVGPUCTRPERM`, run it
  with `sudo` or ask an admin to enable counter access for users
  ([NVIDIA instructions](https://developer.nvidia.com/ERR_NVGPUCTRPERM)).

## Step 1: run without profilers

```bash
./run.sh            # N = 4096, 20 timed launches per kernel
./run.sh 2048 10    # smaller matrix, fewer launches
```

The script builds `build/mm`, checks every kernel against a CPU reference, and writes
`results/timing_N4096.csv`. To run one kernel by hand:

```bash
./build/mm 1 4096 20      # stage, N, timed launches
```

Reference result (RTX A6000, N = 4096):

| Kernel | Median time | GFLOP/s |
|---|---|---|
| naive_uncoalesced | 449 ms | 306 |
| coalesced | 62.8 ms | 2187 |
| shared | 46.3 ms | 2971 |
| shared_bank_conflict | 465 ms | 296 |
| shared_padded | 45.9 ms | 2996 |

## Step 2: profile with nsys and ncu

```bash
./profile.sh        # N = 4096, takes a few minutes
```

For each kernel this writes two reports into `reports/`:

- `<kernel>.nsys-rep` from Nsight Systems: the whole-program timeline (memory copies, kernel launches).
- `<kernel>.ncu-rep` from Nsight Compute: hardware counters for one kernel launch (`--set full`).

Nsight Compute only profiles a launch inside the NVTX range `measured_gemm`, so warm-up
launches are skipped. It runs with `--clock-control none` so the GPU keeps its normal clock
and the profiled time matches `run.sh`.

## Step 3: read the reports from the command line

### Nsight Compute (`.ncu-rep`)

```bash
# Everything, section by section
ncu --import reports/coalesced.ncu-rep --page details

# Speed of Light: compute vs memory throughput, with the per-unit breakdown
ncu --import reports/coalesced.ncu-rep --page details --section SpeedOfLight --print-details all

# Memory: L1/L2/DRAM traffic and the shared-memory table (bank conflicts)
ncu --import reports/shared_bank_conflict.ncu-rep --page details \
    --section MemoryWorkloadAnalysis_Tables --print-details all

# Why warps are waiting (stall reasons)
ncu --import reports/shared.ncu-rep --page details --section WarpStateStats --print-details all

# Per instruction: which shared-memory loads/stores have bank conflicts
python3 tools/bank_conflicts.py reports/shared_bank_conflict.ncu-rep
python3 tools/bank_conflicts.py reports/shared_padded.ncu-rep
#   (the raw per-instruction table it reads: --page source --print-source sass --csv)

# A few chosen metrics for every report
for r in reports/*.ncu-rep; do
  echo "== $r"
  ncu --import "$r" --page raw --metrics \
gpu__time_duration.sum,\
l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum | tail -6
done
```

The metrics in that loop:

| Metric | Meaning |
|---|---|
| `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | global-load requests (one per warp load instruction) |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | 32-byte sectors those requests touched. A warp reading 32 consecutive floats needs 4; all threads reading one address needs 1; 32 scattered floats need 32 |
| `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum` | shared-memory load passes; extra passes come from bank conflicts |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` | bank conflicts on shared loads |

`ncu --import x.ncu-rep --page details --print-details all` also lists the metric
behind each line, which helps when writing your own `--metrics` query.

What to look for:

| Kernel | Speed of Light breakdown: top row | Warp stall | Key counter |
|---|---|---|---|
| naive_uncoalesced | L1 wavefronts ≈ 99% | LG throttle | 16.5 sectors per global-load request (A: 32, B: 1) |
| coalesced | L1 LSU requests ≈ 90% | LG throttle | 2.5 sectors per request (A: 1, B: 4) |
| shared | L1 LSU requests ≈ 80% | MIO throttle | no shared-load bank conflicts |
| shared_bank_conflict | L1 wavefronts ≈ 98% | MIO throttle (very high) | ~21× more shared wavefronts than ideal |
| shared_padded | L1 LSU requests ≈ 81% | MIO throttle | bank conflicts gone |

In the headline numbers, "Compute (SM) Throughput" and "Memory Throughput" are often equal
for these kernels. That doesn't mean they are balanced: both numbers include the SM's
load/store path. Read the breakdown table (`--print-details all`) to see which unit is
actually the limit.

### Nsight Systems (`.nsys-rep`)

```bash
# Kernel times
nsys stats --report cuda_gpu_kern_sum reports/coalesced.nsys-rep

# Host <-> device copies: time and size
nsys stats --report cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum reports/coalesced.nsys-rep

# The NVTX range around the timed launches
nsys stats --report nvtx_sum reports/coalesced.nsys-rep

# Every CUDA event in order (the timeline as text)
nsys stats --report cuda_gpu_trace reports/coalesced.nsys-rep

# CSV output, e.g. for a spreadsheet (writes reports/coalesced_cuda_gpu_kern_sum.csv)
nsys stats --report cuda_gpu_kern_sum --format csv --output reports/coalesced reports/coalesced.nsys-rep
```

### GUI (optional)

Copy `reports/` to a machine with the Nsight tools installed and open them there:
`ncu-ui reports/coalesced.ncu-rep` and `nsys-ui reports/coalesced.nsys-rep`.
In Nsight Compute, open two reports and use **Add Baseline** to compare them
(for example `shared_bank_conflict` against `shared_padded`).
