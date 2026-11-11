// Matrix multiply C = A * B for the GPU memory tutorial.
//
// All matrices are N x N, float, stored row-major:
//     element (row, col) is at index [row * N + col]
//
// Every kernel uses 32 x 32 thread blocks and computes ONE element of C per thread.
// The kernels differ only in how threads read A and B from memory:
//
//     naive_uncoalesced       global memory, neighbouring threads read far-apart addresses
//     coalesced               global memory, neighbouring threads read neighbouring addresses
//     shared                  32 x 32 tiles copied into shared memory and reused
//     shared_bank_conflict    shared tile of B stored transposed  ->  32-way bank conflicts
//     shared_padded           same, with one padding column       ->  no bank conflicts
//
// Usage:   ./build/mm KERNEL [N=4096] [REPS=20]
// Output:  kernel,n,reps,median_ms,gflops,checked_elements,max_abs_error

#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>		// NVTX: labels a code range so profilers can find it
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define TILE 32		// block is TILE x TILE threads, and the shared-memory tile is TILE x TILE


// Stop with a message if a CUDA call fails.
#define CHECK(call) {								\
	cudaError_t err = (call);						\
	if (err != cudaSuccess) {						\
		printf("CUDA error: %s (line %d)\n", cudaGetErrorString(err), __LINE__);	\
		exit(1);							\
	}									\
}



// =====================================================================================
// Kernel 1: naive, UNCOALESCED
//
// threadIdx.x picks the ROW.
// The 32 threads of a warp (consecutive threadIdx.x) work on 32 different rows,
// so their reads of A[row][k] are N floats apart: 32 separate memory transactions.
// =====================================================================================
__global__ void matmul_uncoalesced(float *A, float *B, float *C, int N) {
	int row = blockIdx.y * TILE + threadIdx.x;		// x -> row     (the mistake)
	int col = blockIdx.x * TILE + threadIdx.y;

	float sum = 0.0f;

	for (int k = 0; k < N; k++) {
		sum += A[row * N + k] * B[k * N + col];
	}

	C[row * N + col] = sum;
}



// =====================================================================================
// Kernel 2: naive, COALESCED
//
// Only change: threadIdx.x picks the COLUMN.
// A warp now reads 32 consecutive floats of B (one 128-byte transaction)
// and a single value of A that all 32 threads share.
// =====================================================================================
__global__ void matmul_coalesced(float *A, float *B, float *C, int N) {
	int row = blockIdx.y * TILE + threadIdx.y;
	int col = blockIdx.x * TILE + threadIdx.x;		// x -> column  (the fix)

	float sum = 0.0f;

	for (int k = 0; k < N; k++) {
		sum += A[row * N + k] * B[k * N + col];
	}

	C[row * N + col] = sum;
}



// =====================================================================================
// Kernel 3: SHARED-MEMORY TILING
//
// The block walks along K one 32-wide tile at a time:
//     1. each thread copies one element of A and one element of B into shared memory
//     2. wait until the whole tile is loaded
//     3. each thread does 32 multiply-adds, reading only shared memory
//     4. wait until everyone is done before the tile is overwritten
// Each value loaded from global memory is now reused by 32 threads.
//
// Bank conflicts? Inside a warp only tx changes:
//     As[ty][k]    same address for all 32 threads     ->  broadcast, no conflict
//     Bs[k][tx]    32 consecutive floats = 32 banks    ->  no conflict
// =====================================================================================
__global__ void matmul_shared(float *A, float *B, float *C, int N) {
	__shared__ float As[TILE][TILE];
	__shared__ float Bs[TILE][TILE];

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = blockIdx.y * TILE + ty;
	int col = blockIdx.x * TILE + tx;

	float sum = 0.0f;

	for (int t = 0; t < N; t += TILE) {

		// 1. load one tile of A and one tile of B
		As[ty][tx] = A[row * N + (t + tx)];
		Bs[ty][tx] = B[(t + ty) * N + col];

		// 2. wait for the whole tile
		__syncthreads();

		// 3. multiply using shared memory only
		for (int k = 0; k < TILE; k++) {
			sum += As[ty][k] * Bs[k][tx];
		}

		// 4. wait before loading the next tile
		__syncthreads();
	}

	C[row * N + col] = sum;
}



// =====================================================================================
// Kernels 4 and 5: SHARED TILING, B STORED TRANSPOSED
//
// Element (k, tx) of the B tile is stored at Bs[tx * pitch + k].
// In the inner loop a warp (32 values of tx, same k) reads addresses `pitch` floats apart.
// Shared memory has 32 banks, and bank = (float index) % 32:
//
//     pitch = 32    every thread hits the SAME bank     ->  32-way conflict
//     pitch = 33    threads hit 32 DIFFERENT banks      ->  no conflict
//
// One extra column of padding (33 instead of 32) removes the conflict.
//
// Why is pitch a function argument and not a fixed number?
// If the compiler knows pitch = 32, it merges 4 neighbouring reads into one 128-bit load,
// which changes the conflict pattern. Passing pitch at run time keeps both cases on exactly
// the same instructions, so the only difference you measure is the bank conflict.
// =====================================================================================
__global__ void matmul_shared_transposed(float *A, float *B, float *C, int N, int pitch) {
	__shared__ float As[TILE][TILE];
	__shared__ float Bs[TILE * (TILE + 1)];			// room for pitch up to 33

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int row = blockIdx.y * TILE + ty;
	int col = blockIdx.x * TILE + tx;

	float sum = 0.0f;

	for (int t = 0; t < N; t += TILE) {

		As[ty][tx]          = A[row * N + (t + tx)];
		Bs[tx * pitch + ty] = B[(t + ty) * N + col];		// store B transposed

		__syncthreads();

		for (int k = 0; k < TILE; k++) {
			sum += As[ty][k] * Bs[tx * pitch + k];		// warp reads with stride = pitch
		}

		__syncthreads();
	}

	C[row * N + col] = sum;
}



// =====================================================================================
// Host code
// =====================================================================================

const char *KERNEL_NAMES[5] = {
	"naive_uncoalesced",
	"coalesced",
	"shared",
	"shared_bank_conflict",
	"shared_padded"
};


void launch(int kernel, float *A, float *B, float *C, int N) {
	dim3 block(TILE, TILE);				// 32 x 32 = 1024 threads
	dim3 grid(N / TILE, N / TILE);			// one block per 32 x 32 tile of C

	if (kernel == 0) {
		matmul_uncoalesced<<<grid, block>>>(A, B, C, N);
	}

	if (kernel == 1) {
		matmul_coalesced<<<grid, block>>>(A, B, C, N);
	}

	if (kernel == 2) {
		matmul_shared<<<grid, block>>>(A, B, C, N);
	}

	if (kernel == 3) {
		matmul_shared_transposed<<<grid, block>>>(A, B, C, N, 32);	// pitch 32: bank conflicts
	}

	if (kernel == 4) {
		matmul_shared_transposed<<<grid, block>>>(A, B, C, N, 33);	// pitch 33: padded
	}

	CHECK(cudaGetLastError());
}


// Used by qsort to sort the timings.
int compare_floats(const void *a, const void *b) {
	float x = *(float *)a;
	float y = *(float *)b;

	if (x < y) {
		return -1;
	}

	if (x > y) {
		return 1;
	}

	return 0;
}



int main(int argc, char **argv) {
	// ---------------- 1. Read the arguments ----------------

	int kernel = -1;

	for (int i = 0; i < 5 && argc > 1; i++) {
		if (strcmp(argv[1], KERNEL_NAMES[i]) == 0) {
			kernel = i;
		}
	}

	int N = 4096;					// default matrix size
	int reps = 20;					// default number of timed launches

	if (argc > 2) {
		N = atoi(argv[2]);
	}

	if (argc > 3) {
		reps = atoi(argv[3]);
	}

	if (kernel < 0 || N < TILE || N % TILE != 0 || reps < 1) {
		printf("usage: %s KERNEL [N] [REPS]\n", argv[0]);
		printf("    KERNEL:  naive_uncoalesced | coalesced | shared | shared_bank_conflict | shared_padded\n");
		printf("    N:       multiple of 32 (default 4096)\n");
		printf("    REPS:    number of timed launches (default 20)\n");
		return 1;
	}

	cudaDeviceProp prop;
	CHECK(cudaGetDeviceProperties(&prop, 0));
	fprintf(stderr, "device=%s kernel=%s N=%d\n", prop.name, KERNEL_NAMES[kernel], N);


	// ---------------- 2. Create the input matrices on the CPU ----------------
	// Values are multiples of 1/8 between -1 and 1, so every product and sum is exact
	// and the GPU result must match the CPU result exactly.

	size_t count = (size_t)N * N;
	size_t bytes = count * sizeof(float);

	float *A = (float *)malloc(bytes);
	float *B = (float *)malloc(bytes);
	float *C = (float *)malloc(bytes);

	srand(42);

	for (size_t i = 0; i < count; i++) {
		A[i] = (rand() % 17 - 8) / 8.0f;
		B[i] = (rand() % 17 - 8) / 8.0f;
	}


	// ---------------- 3. Copy the inputs to the GPU ----------------

	float *dA, *dB, *dC;

	CHECK(cudaMalloc(&dA, bytes));
	CHECK(cudaMalloc(&dB, bytes));
	CHECK(cudaMalloc(&dC, bytes));

	CHECK(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice));


	// ---------------- 4. Warm-up: 2 launches, not timed, not profiled by ncu ----------------

	for (int i = 0; i < 2; i++) {
		launch(kernel, dA, dB, dC, N);
	}

	CHECK(cudaDeviceSynchronize());


	// ---------------- 5. Check the result against the CPU ----------------
	// Small N: check every element.  Large N: check 1024 elements spread over the matrix.

	CHECK(cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost));

	size_t checks = 1024;

	if (N <= 256) {
		checks = count;
	}

	double max_error = 0.0;

	for (size_t i = 0; i < checks; i++) {

		size_t idx = i;

		if (checks != count) {
			idx = (i * 1000003) % count;		// jump around the matrix (1000003 is prime)
		}

		int row = idx / N;
		int col = idx % N;

		double expected = 0.0;

		for (int k = 0; k < N; k++) {
			expected += (double)A[(size_t)row * N + k] * B[(size_t)k * N + col];
		}

		double error = fabs(C[idx] - expected);

		if (error > max_error) {
			max_error = error;
		}
	}

	if (max_error > 1e-3) {
		printf("WRONG RESULT: max error %g\n", max_error);
		return 1;
	}


	// ---------------- 6. Timed launches, inside the NVTX range "measured_gemm" ----------------
	// nsys shows this range on its timeline; ncu profiles only kernels inside it.

	cudaEvent_t start, stop;
	CHECK(cudaEventCreate(&start));
	CHECK(cudaEventCreate(&stop));

	float *times_ms = (float *)malloc(reps * sizeof(float));

	nvtxRangePushA("measured_gemm");

	for (int i = 0; i < reps; i++) {

		CHECK(cudaEventRecord(start));
		launch(kernel, dA, dB, dC, N);
		CHECK(cudaEventRecord(stop));

		CHECK(cudaEventSynchronize(stop));
		CHECK(cudaEventElapsedTime(&times_ms[i], start, stop));
	}

	nvtxRangePop();


	// ---------------- 7. Report the median time ----------------

	qsort(times_ms, reps, sizeof(float), compare_floats);

	float  median_ms = times_ms[reps / 2];
	double gflops    = 2.0 * N * N * N / (median_ms * 1e6);		// 2 * N^3 flops (multiply + add)

	printf("%s,%d,%d,%.4f,%.1f,%zu,%g\n",
	       KERNEL_NAMES[kernel], N, reps, median_ms, gflops, checks, max_error);


	// ---------------- 8. Clean up ----------------

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	free(A);
	free(B);
	free(C);
	free(times_ms);

	return 0;
}
