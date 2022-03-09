// Vector addition: C = 1/A + 1/B.
// compile with the following command:
//
// (for GTX1060)
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O2 -m64 -o MatAdd MatAdd.cu

// Includes
#include <stdio.h>
#include <stdlib.h>

#define N 6400

// Variables
float h_A[N][N]; // host matrices
float h_B[N][N];
float h_C[N][N];
float (*d_A)[N]; // device matrices
float (*d_B)[N];
float (*d_C)[N];

// Functions
void RandomInit(float matrix[N][N]) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = rand() / (float)RAND_MAX;
        }
    }
}

// Device code
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

// Host code
int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: ./MatAdd <gpu_id> <block_size>\n");
        exit(1);
    }
    int gid = atoi(argv[1]); // GPU_ID
    int block_size = atoi(argv[2]);

    printf("Block size: (%d, %d)\n", block_size, block_size);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
        fprintf(stderr, "Cannot select GPU with device ID = %d\n", gid);
        exit(1);
    }

    cudaSetDevice(gid);

    long size = N * N * sizeof(float);

    // Initialize the input matrices with random numbers
    RandomInit(h_A);
    RandomInit(h_B);

    // Allocate matrices in device memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy matrices from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // create the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Set the sizes of threads and blocks
    dim3 dimBlock(block_size, block_size);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                 (N + dimBlock.y - 1) / dimBlock.y);

    // start the timer
    cudaEventRecord(start, 0);

    MatAdd<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float gputime;
    cudaEventElapsedTime(&gputime, start, stop);
    printf("Processing time for GPU: %f (ms) \n", gputime);
    printf("GPU Gflops: %f\n", 3 * N * N / (1000000.0 * gputime));

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy result from device memory to host memory
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // check result
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (fabs(h_A[i][j] + h_B[i][j] - h_C[i][j]) > 1e-10) {
                fprintf(stderr, "Result checking failed at (%d, %d)!\n", i, j);
                exit(1);
            }
        }
    }

    cudaDeviceReset();
}