// Matrix multiply C = A * B, FP32, N x N, row-major.
//
// Stages (the numbers match the earlier, larger version of this file, so old results line up):
//
//     0     naive, uncoalesced global loads
//     1     naive, coalesced global loads
//     2     shared-memory tile (Lecture 3), no bank conflicts
//     17    shared tile with B stored transposed, pitch 32   ->  32-way bank conflicts
//     18    same kernel, pitch 33 (one padding column)      ->  no bank conflicts
//
// Usage:   ./build/mm STAGE [N=1024] [REPS=20]
// Output:  stage,n,precision,reps,median_ms,tflops,checked_elements,max_abs_error

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>


#define TILE 32


#define CUDA(call) {										\
	cudaError_t err = (call);								\
	if (err != cudaSuccess) {								\
		fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));	\
		exit(1);									\
	}											\
}



// Stage 0: threadIdx.x picks the ROW, so neighbouring threads read addresses N floats apart.
__global__ void naive_uncoalesced(const float* a, const float* b, float* c, int n) {
	int row = blockIdx.y * TILE + threadIdx.x; // OBSERVE: threadIdx.x picks the ROW, not the COLUMN
	int col = blockIdx.x * TILE + threadIdx.y;

	float v = 0;

	for (int k = 0; k < n; k++) {
		v += a[row * n + k] * b[k * n + col];
	}

	c[row * n + col] = v;
}



// Stage 1: threadIdx.x picks the COLUMN, so a warp reads 32 consecutive floats of B.
__global__ void naive_coalesced(const float* a, const float* b, float* c, int n) {
	int row = blockIdx.y * TILE + threadIdx.y; // OBSERVE: threadIdx.y picks the ROW, not the COLUMN
	int col = blockIdx.x * TILE + threadIdx.x;

	float v = 0;

	for (int k = 0; k < n; k++) {
		v += a[row * n + k] * b[k * n + col];
	}

	c[row * n + col] = v;
}



// Stage 2: one result per thread, 32 x 32 tiles of A and B in shared memory.
// Within a warp only x changes: as[y][k] is a broadcast and bs[k][x] hits 32 different banks.
__global__ void shared_tile(const float* a, const float* b, float* c, int n) {
	__shared__ float as[TILE][TILE];
	__shared__ float bs[TILE][TILE];

	int x = threadIdx.x;
	int y = threadIdx.y;

	int row = blockIdx.y * TILE + y;
	int col = blockIdx.x * TILE + x;

	float v = 0;

	for (int base = 0; base < n; base += TILE) {
		as[y][x] = a[row * n + base + x];
		bs[y][x] = b[(base + y) * n + col];

		__syncthreads();

		#pragma unroll
		for (int k = 0; k < TILE; k++) {
			v += as[y][k] * bs[k][x];
		}

		__syncthreads();
	}

	c[row * n + col] = v;
}



// Stages 17 and 18: same tiling, but B is stored transposed in shared memory.
// A warp reads bs[x * pitch + k]: pitch 32 puts all 32 threads in one bank, pitch 33 spreads them.
// Pitch is a runtime argument so both stages run exactly the same compiled kernel.
__global__ void shared_tile_pitch(const float* a, const float* b, float* c, int n, int pitch) {
	__shared__ float as[TILE][TILE];
	__shared__ float bs[TILE * (TILE + 1)];

	int x = threadIdx.x;
	int y = threadIdx.y;

	int row = blockIdx.y * TILE + y;
	int col = blockIdx.x * TILE + x;

	float v = 0;

	for (int base = 0; base < n; base += TILE) {
		as[y][x] = a[row * n + base + x];
		bs[x * pitch + y] = b[(base + y) * n + col];

		__syncthreads();

		#pragma unroll
		for (int k = 0; k < TILE; k++) {
			v += as[y][k] * bs[x * pitch + k];
		}

		__syncthreads();
	}

	c[row * n + col] = v;
}



void launch(int stage, const float* a, const float* b, float* c, int n) {
	dim3 grid(n / TILE, n / TILE);
	dim3 block(TILE, TILE);

	if (stage == 0) {
		naive_uncoalesced<<<grid, block>>>(a, b, c, n);
	}

	if (stage == 1) {
		naive_coalesced<<<grid, block>>>(a, b, c, n);
	}

	if (stage == 2) {
		shared_tile<<<grid, block>>>(a, b, c, n);
	}

	if (stage == 17) {
		shared_tile_pitch<<<grid, block>>>(a, b, c, n, 32);
	}

	if (stage == 18) {
		shared_tile_pitch<<<grid, block>>>(a, b, c, n, 33);
	}

	CUDA(cudaGetLastError());
}



int compare_floats(const void* p, const void* q) {
	float x = *(const float*) p;
	float y = *(const float*) q;

	if (x < y) {
		return -1;
	}

	if (x > y) {
		return 1;
	}

	return 0;
}



int main(int argc, char** argv) {

	// ---- Arguments ----

	int stage = 0;
	int n = 1024;
	int reps = 20;

	if (argc > 1) {
		stage = atoi(argv[1]);
	}

	if (argc > 2) {
		n = atoi(argv[2]);
	}

	if (argc > 3) {
		reps = atoi(argv[3]);
	}

	int valid_stage = (stage == 0 || stage == 1 || stage == 2 || stage == 17 || stage == 18);

	if (!valid_stage || n < TILE || n % TILE != 0 || n > 16384 || reps < 1) {
		fprintf(stderr, "usage: mm STAGE[0, 1, 2, 17, 18] N[multiple of 32, <= 16384] REPS[>= 1]\n");
		return 1;
	}

	cudaDeviceProp prop;
	CUDA(cudaGetDeviceProperties(&prop, 0));
	fprintf(stderr, "device=%s cc=%d.%d stage=%d n=%d\n", prop.name, prop.major, prop.minor, stage, n);


	// Each step below is wrapped in an NVTX range, so the nsys timeline shows it as a labeled bar.


	// ---- Inputs on the CPU: multiples of 1/8, so every product and sum is exact ----

	nvtxRangePushA("make_inputs");

	size_t count = (size_t) n * n;
	size_t bytes = count * sizeof(float);

	float* a = (float*) malloc(bytes);
	float* b = (float*) malloc(bytes);
	float* out = (float*) malloc(bytes);

	srand(42);

	for (size_t i = 0; i < count; i++) {
		a[i] = (rand() % 17 - 8) / 8.0f;
		b[i] = (rand() % 17 - 8) / 8.0f;
	}

	nvtxRangePop();


	// ---- GPU memory (the first CUDA call also starts the GPU context, ~0.1 s) ----

	nvtxRangePushA("allocate");

	float *da, *db, *dc;

	CUDA(cudaMalloc(&da, bytes));
	CUDA(cudaMalloc(&db, bytes));
	CUDA(cudaMalloc(&dc, bytes));

	nvtxRangePop();


	// ---- Copy inputs to the GPU ----

	nvtxRangePushA("copy_inputs");

	CUDA(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
	CUDA(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));

	nvtxRangePop();


	// ---- Warm-up: 3 launches, not timed, and outside "measured_gemm" so ncu skips them ----

	nvtxRangePushA("warmup");

	for (int i = 0; i < 3; i++) {
		launch(stage, da, db, dc, n);
	}

	// Copy the result back. This also waits for the warm-up kernels to finish.
	CUDA(cudaMemcpy(out, dc, bytes, cudaMemcpyDeviceToHost));

	nvtxRangePop();


	// ---- Correctness on the CPU: every element for small N, 1024 samples for large N ----

	nvtxRangePushA("check_on_cpu");

	size_t checks = 1024;

	if (n <= 256) {
		checks = count;
	}

	double maxerr = 0;

	for (size_t i = 0; i < checks; i++) {
		size_t index = i;

		if (checks != count) {
			index = (i * 1000003) % count;		// 1000003 is prime: spreads samples over the matrix
		}

		int r = index / n;
		int col = index % n;

		double ref = 0;

		for (int k = 0; k < n; k++) {
			ref += (double) a[(size_t) r * n + k] * b[(size_t) k * n + col];
		}

		double err = fabs((double) out[index] - ref);

		// isfinite matters: a NaN error would pass a plain "err > limit" test.
		if (!isfinite(out[index]) || err > 1e-3 + 1e-4 * fabs(ref)) {
			fprintf(stderr, "correctness check failed at element %zu\n", index);
			return 1;
		}

		if (err > maxerr) {
			maxerr = err;
		}
	}

	nvtxRangePop();


	// ---- Timed launches inside the NVTX range "measured_gemm" (ncu profiles only this range) ----
	// Each launch gets its own start/stop event. All launches are queued back to back and
	// the CPU waits once at the end, so the timeline shows one wait instead of one per launch.

	cudaEvent_t* start = (cudaEvent_t*) malloc(reps * sizeof(cudaEvent_t));
	cudaEvent_t* stop = (cudaEvent_t*) malloc(reps * sizeof(cudaEvent_t));

	for (int i = 0; i < reps; i++) {
		CUDA(cudaEventCreate(&start[i]));
		CUDA(cudaEventCreate(&stop[i]));
	}

	nvtxRangePushA("measured_gemm");

	for (int i = 0; i < reps; i++) {
		CUDA(cudaEventRecord(start[i]));
		launch(stage, da, db, dc, n);
		CUDA(cudaEventRecord(stop[i]));
	}

	CUDA(cudaEventSynchronize(stop[reps - 1]));		// one wait for all launches

	nvtxRangePop();

	float* times = (float*) malloc(reps * sizeof(float));

	for (int i = 0; i < reps; i++) {
		CUDA(cudaEventElapsedTime(&times[i], start[i], stop[i]));
	}


	// ---- Median time ----

	qsort(times, reps, sizeof(float), compare_floats);

	float ms = times[reps / 2];
	double tflops = 2.0 * n * n * n / (ms * 1e9);

	printf("stage,n,precision,reps,median_ms,tflops,checked_elements,max_abs_error\n");
	printf("%d,%d,fp32,%d,%.6f,%.6f,%zu,%.8g\n", stage, n, reps, ms, tflops, checks, maxerr);


	// ---- Clean up ----

	for (int i = 0; i < reps; i++) {
		cudaEventDestroy(start[i]);
		cudaEventDestroy(stop[i]);
	}

	free(start);
	free(stop);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	free(a);
	free(b);
	free(out);
	free(times);

	return 0;
}
