#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Variables
float *h_A; // host vectors
float *h_B;
float *d_A; // device vectors
float *d_B;

// Functions
void RandomInit(float *, int);

// Device code
__global__ void vecMaxAbs(const float *A, float *B, int N) {
    extern __shared__ float cache[];

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    float temp = 0.0; // register for each thread
    while (i < N) {
        temp = fmax(temp, fabs(A[i]));
        i += blockDim.x * gridDim.x;
    }

    cache[cacheIndex] = temp; // set the cache value

    __syncthreads();

    // perform parallel reduction, threadsPerBlock must be 2^m
    int ib = blockDim.x / 2;
    while (ib != 0) {
        if (cacheIndex < ib)
            cache[cacheIndex] =
                fmax(cache[cacheIndex], fabs(cache[cacheIndex + ib]));

        __syncthreads();
        ib /= 2;
    }

    if (cacheIndex == 0)
        B[blockIdx.x] = cache[0];
}

// Host code
int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <gpu_id> <block_size> <grid_size>\n",
                argv[0]);
        exit(1);
    }
    int gid = atoi(argv[1]);
    int threadsPerBlock = atoi(argv[2]);
    int blocksPerGrid = atoi(argv[3]);

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
        printf("Cannot select GPU with device ID = %d!\n", gid);
        exit(1);
    }
    cudaSetDevice(gid);

    int N = 81920007;

    if (threadsPerBlock > 1024) {
        printf("The number of threads per block must be less than 1024!\n");
        exit(0);
    }

    if (blocksPerGrid > 2147483647) {
        printf("The number of blocks must be less than 2147483647!\n");
        exit(0);
    }

    // Allocate vectors in host memory
    int size = N * sizeof(float);
    int sb = blocksPerGrid * sizeof(float);

    h_A = (float *)malloc(size);
    h_B = (float *)malloc(sb); // contains the result from each block

    if (h_A == NULL || h_B == NULL) {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize input vectors
    RandomInit(h_A, N);

    // create the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the timer
    cudaEventRecord(start, 0);

    // Allocate vectors in device memory
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, sb);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    // stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float Intime;
    cudaEventElapsedTime(&Intime, start, stop);
    printf("Input time for GPU: %f (ms) \n", Intime);

    // start the timer
    cudaEventRecord(start, 0);

    int sm = threadsPerBlock * sizeof(float);
    vecMaxAbs<<<blocksPerGrid, threadsPerBlock, sm>>>(d_A, d_B, N);

    // stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float gputime;
    cudaEventElapsedTime(&gputime, start, stop);
    printf("Processing time for GPU: %f (ms) \n", gputime);
    printf("GPU Gflops: %f\n", N / (1000000.0 * gputime));

    // Copy result from device memory to host memory
    // h_C contains the result of each block in host memory

    // start the timer
    cudaEventRecord(start, 0);

    cudaMemcpy(h_B, d_B, sb, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);

    double gpu_ans = 0.0;
    for (int i = 0; i < blocksPerGrid; i++)
        gpu_ans = fmax(gpu_ans, h_B[i]);

    // stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float Outime;
    cudaEventElapsedTime(&Outime, start, stop);
    printf("Output time for GPU: %f (ms) \n", Outime);

    float gputime_tot;
    gputime_tot = Intime + gputime + Outime;
    printf("Total time for GPU: %f (ms) \n", gputime_tot);

    // start the timer
    cudaEventRecord(start, 0);

    // to compute the reference solution
    double cpu_ans = 0.0;
    for (int i = 0; i < N; i++)
        cpu_ans = fmax(cpu_ans, h_A[i]);

    // stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime(&cputime, start, stop);
    printf("Processing time for CPU: %f (ms) \n", cputime);
    printf("CPU Gflops: %f\n", N / (1000000.0 * cputime));
    printf("Speed up of GPU = %f\n", cputime / (gputime_tot));

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // check result
    printf("Check result:\n");
    double diff = fabs(gpu_ans - cpu_ans);
    printf("|gpu_ans - cpu_ans|=%20.15e\n", diff);
    printf("gpu_ans =%20.15e\n", gpu_ans);
    printf("cpu_ans =%20.15e\n", cpu_ans);

    free(h_A);
    free(h_B);

    cudaDeviceReset();
}

// Allocates an array with random float entries in (-1,1)
void RandomInit(float *data, int n) {
    for (int i = 0; i < n; ++i)
        data[i] = 2.0 * rand() / (float)RAND_MAX - 1.0;
}
