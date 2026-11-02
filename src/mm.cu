// Matrix multiply C = A * B (row-major, FP32, N x N) for the GPU memory tutorial.
//
// Kernels (all use 32x32 thread blocks, one output element per thread):
//   naive_uncoalesced     threadIdx.x walks down rows   -> uncoalesced global loads
//   coalesced             threadIdx.x walks along a row -> coalesced global loads
//   shared                Lecture 3 tile in shared memory, conflict-free
//   shared_bank_conflict  B tile stored transposed with pitch 32 -> 32-way bank conflicts
//   shared_padded         same kernel, pitch 33 -> conflicts removed by padding
//
// Usage: ./build/mm KERNEL [N=4096] [REPS=20]
// Prints one CSV row: kernel,n,reps,median_ms,gflops,checked_elements,max_abs_error
// The timed launches sit inside an NVTX range "measured_gemm", so profilers can
// skip the warm-up launches.
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

constexpr int TILE = 32;

// Kernel 1 and 2: one thread per output, A and B read straight from global memory.
// The only difference is which thread index walks along the row of C.
//   Coalesced = false: neighbouring threads (threadIdx.x) get neighbouring ROWS,
//                      so their loads of B are N floats apart -> one sector each.
//   Coalesced = true:  neighbouring threads get neighbouring COLUMNS,
//                      so a warp reads 32 consecutive floats of B -> 4 sectors.
template <bool Coalesced>
__global__ void naive(const float* a, const float* b, float* c, int n) {
  int row = blockIdx.y * TILE + (Coalesced ? threadIdx.y : threadIdx.x);
  int col = blockIdx.x * TILE + (Coalesced ? threadIdx.x : threadIdx.y);
  float acc = 0.f;
  for (int k = 0; k < n; k++) acc = fmaf(a[row * n + k], b[k * n + col], acc);
  c[row * n + col] = acc;
}

// Kernel 3: the shared-memory tile. Each block loads a 32x32 tile of A and B once,
// then every thread reuses it 32 times. Within a warp only threadIdx.x changes:
//   as[y][k]  -> one address for the whole warp (broadcast)
//   bs[k][x]  -> 32 consecutive floats = 32 different banks
// so there are no bank conflicts.
__global__ void shared_tile(const float* a, const float* b, float* c, int n) {
  __shared__ float as[TILE][TILE], bs[TILE][TILE];
  int x = threadIdx.x, y = threadIdx.y;
  int row = blockIdx.y * TILE + y, col = blockIdx.x * TILE + x;
  float acc = 0.f;
  for (int base = 0; base < n; base += TILE) {
    as[y][x] = a[row * n + base + x];
    bs[y][x] = b[(base + y) * n + col];
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; k++) acc = fmaf(as[y][k], bs[k][x], acc);
    __syncthreads();
  }
  c[row * n + col] = acc;
}

// Kernel 4 and 5: same tile, but the B tile is stored TRANSPOSED in shared memory,
// bs[x * pitch + k]. A warp (32 values of x, fixed k) then reads addresses that are
// `pitch` floats apart:
//   pitch = 32 -> all 32 addresses fall in the same bank -> 32-way conflict
//   pitch = 33 -> each address lands in a different bank -> no conflict
// Pitch is a runtime argument, so both cases run the exact same compiled code.
__global__ void shared_tile_pitch(const float* a, const float* b, float* c, int n, int pitch) {
  __shared__ float as[TILE][TILE], bs[TILE * (TILE + 1)];
  int x = threadIdx.x, y = threadIdx.y;
  int row = blockIdx.y * TILE + y, col = blockIdx.x * TILE + x;
  float acc = 0.f;
  for (int base = 0; base < n; base += TILE) {
    as[y][x] = a[row * n + base + x];
    bs[x * pitch + y] = b[(base + y) * n + col];
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; k++) acc = fmaf(as[y][k], bs[x * pitch + k], acc);
    __syncthreads();
  }
  c[row * n + col] = acc;
}

static const char* KERNELS[] = {"naive_uncoalesced", "coalesced", "shared",
                                "shared_bank_conflict", "shared_padded"};

int main(int argc, char** argv) {
  try {
    if (argc < 2) throw std::runtime_error("usage: mm KERNEL [N] [REPS]");
    std::string kernel = argv[1];
    int which = -1;
    for (int i = 0; i < 5; i++) if (kernel == KERNELS[i]) which = i;
    int n = argc > 2 ? std::stoi(argv[2]) : 4096;
    int reps = argc > 3 ? std::stoi(argv[3]) : 20;
    if (which < 0) throw std::runtime_error("unknown kernel; use one of: naive_uncoalesced coalesced shared shared_bank_conflict shared_padded");
    if (n < TILE || n % TILE || n > 16384 || reps < 1) throw std::runtime_error("N must be a multiple of 32 (<= 16384), REPS >= 1");

    cudaDeviceProp prop; CUDA(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr, "device=%s cc=%d.%d kernel=%s n=%d\n", prop.name, prop.major, prop.minor, kernel.c_str(), n);

    size_t count = size_t(n) * n, bytes = count * sizeof(float);
    std::vector<float> a(count), b(count), out(count);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-8, 8);
    // Multiples of 1/8 keep every product and sum exact, so the check is strict.
    for (size_t i = 0; i < count; i++) { a[i] = dist(rng) / 8.f; b[i] = dist(rng) / 8.f; }

    float *da, *db, *dc;
    CUDA(cudaMalloc(&da, bytes)); CUDA(cudaMalloc(&db, bytes)); CUDA(cudaMalloc(&dc, bytes));
    CUDA(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));

    dim3 grid(n / TILE, n / TILE), block(TILE, TILE);
    auto launch = [&]() {
      switch (which) {
        case 0: naive<false><<<grid, block>>>(da, db, dc, n); break;
        case 1: naive<true><<<grid, block>>>(da, db, dc, n); break;
        case 2: shared_tile<<<grid, block>>>(da, db, dc, n); break;
        case 3: shared_tile_pitch<<<grid, block>>>(da, db, dc, n, 32); break;
        case 4: shared_tile_pitch<<<grid, block>>>(da, db, dc, n, 33); break;
      }
      CUDA(cudaGetLastError());
    };

    // Warm-up, then check the result against a CPU reference.
    for (int i = 0; i < 2; i++) launch();
    CUDA(cudaDeviceSynchronize());
    CUDA(cudaMemcpy(out.data(), dc, bytes, cudaMemcpyDeviceToHost));
    size_t checks = n <= 256 ? count : 1024;  // every element when small, samples when large
    double maxerr = 0;
    for (size_t i = 0; i < checks; i++) {
      size_t idx = n <= 256 ? i : (size_t(rng()) * 2654435761u + i) % count;
      int r = idx / n, col = idx % n;
      double ref = 0;
      for (int k = 0; k < n; k++) ref += double(a[size_t(r) * n + k]) * b[size_t(k) * n + col];
      double err = std::abs(double(out[idx]) - ref);
      maxerr = std::max(maxerr, err);
      if (!std::isfinite(out[idx]) || err > 1e-3 + 1e-4 * std::abs(ref))
        throw std::runtime_error("correctness check failed");
    }

    // Timed launches, inside the NVTX range the profilers filter on.
    cudaEvent_t start, end;
    CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&end));
    std::vector<float> times;
    nvtxRangePushA("measured_gemm");
    for (int i = 0; i < reps; i++) {
      CUDA(cudaEventRecord(start)); launch(); CUDA(cudaEventRecord(end));
      CUDA(cudaEventSynchronize(end));
      float ms; CUDA(cudaEventElapsedTime(&ms, start, end)); times.push_back(ms);
    }
    nvtxRangePop();
    std::sort(times.begin(), times.end());
    float ms = times[times.size() / 2];
    printf("%s,%d,%d,%.4f,%.1f,%zu,%g\n", kernel.c_str(), n, reps, ms,
           2.0 * n * n * n / (ms * 1e6), checks, maxerr);

    CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(end));
    CUDA(cudaFree(da)); CUDA(cudaFree(db)); CUDA(cudaFree(dc));
  } catch (const std::exception& e) {
    fprintf(stderr, "%s\n", e.what());
    return 1;
  }
}
