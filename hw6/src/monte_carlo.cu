#include <cuda_runtime.h>
#include <gsl/gsl_rng.h> /* header file for using gsl_rng */
#include <math.h>
#include <omp.h>
#include <stdio.h>

#define N 10
#define C 1.5819767068693265

float *h_A;
float *h_B;

// Functions
double f(double *); /* function for MC integration */

// Device code
__global__ void monte_carlo(const float *A, float *B, int dnum) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float w;

    float x, y, fx;

    if (i < dnum) {
        w = 1;
        fx = 1;
        for (int j = 0; j < N; j++) {
            y = A[N * i + j];
            x = -log(1 - y / C);
            fx += x * x;
            w *= C * exp(-x);
        }
        fx = 1 / fx / w;
        B[i] = fx;
    }

    __syncthreads();
}

int main(void) {
    gsl_rng *rng; /* pointer to gsl_rng random number generator */
    int seed = 1234, cpu_thread_id = 0;
    int i, NGPU;
    int *Dev;
    double x[N], y[N];
    int dnum;
    double mean, sigma, mean1, sigma1, mean2, sigma2;
    double fx, w;
    rng = gsl_rng_alloc(gsl_rng_mt19937); /* allocate RNG to mt19937 */
    gsl_rng_set(rng, seed);               /* set the seed */

    printf("Monte-Carlo integration of one-dimensional integral\n");
    printf("Enter the number of samplings: ");
    scanf("%d", &dnum);
    printf("%d\n", dnum);

    int size = dnum * sizeof(float);
    h_A = (float *)malloc(N * size);
    h_B = (float *)malloc(size);
    for (i = 0; i < N * dnum; ++i)
        h_A[i] = gsl_rng_uniform(rng);

    printf("Enter the number of GPUs: ");
    scanf("%d", &NGPU);
    printf("%d\n", NGPU);
    Dev = (int *)malloc(sizeof(int) * NGPU);
    int numDev = 0;
    printf("GPU device number: ");
    for (int i = 0; i < NGPU; i++) {
        scanf("%d", &Dev[i]);
        printf("%d ", Dev[i]);
        numDev++;
        if (getchar() == '\n')
            break;
    }
    printf("\n");
    if (numDev != NGPU) {
        fprintf(stderr, "Should input %d GPU device numbers\n", NGPU);
        exit(1);
    }

    // Set the sizes of threads and blocks
    int threadsPerBlock;
    printf("Enter the number of threads per block: ");
    scanf("%d", &threadsPerBlock);
    printf("%d\n", threadsPerBlock);
    if (threadsPerBlock > 1024) {
        printf("The number of threads per block must be less than 1024 ! \n");
        exit(1);
    }
    int blocksPerGrid =
        (dnum + threadsPerBlock * NGPU - 1) / (threadsPerBlock * NGPU);
    printf("The number of blocks is %d\n", blocksPerGrid);
    if (blocksPerGrid > 2147483647) {
        printf("The number of blocks must be less than 2147483647 ! \n");
        exit(1);
    }

    mean = 0.0; /* for simple sampling */
    sigma = 0.0;
    mean1 = 0.0; /* for importance sampling */
    sigma1 = 0.0;
    mean2 = 0.0; /* for gpu */
    sigma2 = 0.0;
    for (i = 0; i < N * dnum; i += N) {
        w = 1;
        for (int j = 0; j < N; ++j) {
            y[j] = h_A[i + j];
            x[j] = -log(1 - y[j] / C);
            w *= C * exp(-x[j]);
        }
        fx = f(y);
        mean += fx;
        sigma += fx * fx;
        fx = f(x) / w;
        mean1 += fx;
        sigma1 += fx * fx;
    }
    mean /= dnum;
    sigma = sqrt((sigma / dnum - mean * mean) / dnum);
    mean1 /= dnum;
    sigma1 = sqrt((sigma1 / dnum - mean1 * mean1) / dnum);
    printf("\n");
    printf("Simple sampling:     %.10f +/- %.10f\n", mean, sigma);
    printf("Importance sampling: %.10f +/- %.10f\n", mean1, sigma1);

    omp_set_num_threads(NGPU);
#pragma omp parallel private(cpu_thread_id)
    {
        float *d_A, *d_B;
        cpu_thread_id = omp_get_thread_num();
        cudaSetDevice(Dev[cpu_thread_id]);

        // Allocate vectors in device memory
        cudaMalloc((void **)&d_A, N * size / NGPU);
        cudaMalloc((void **)&d_B, size / NGPU);

        cudaMemcpy(d_A, h_A + N * dnum / NGPU * cpu_thread_id, N * size / NGPU,
                   cudaMemcpyHostToDevice);
#pragma omp barrier

        monte_carlo<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, dnum / NGPU);
        cudaDeviceSynchronize();

        cudaMemcpy(h_B + dnum / NGPU * cpu_thread_id, d_B, size / NGPU,
                   cudaMemcpyDeviceToHost);
        cudaFree(d_A);
        cudaFree(d_B);
    }

    for (i = 0; i < dnum; ++i) {
        fx = h_B[i];
        mean2 += fx;
        sigma2 += fx * fx;
    }
    mean2 /= dnum;
    sigma2 = sqrt((sigma2 / dnum - mean2 * mean2) / dnum);
    printf("Importance sampling (GPU): %.10f +/- %.10f\n", mean2, sigma2);

    free(h_A);
    free(h_B);

    for (i = 0; i < NGPU; i++) {
        cudaSetDevice(i);
        cudaDeviceReset();
    }
}

/* the function for MC integration */

double f(double *x) {
    double ans = 1;
    for (int i = 0; i < N; ++i) {
        ans += x[i] * x[i];
    }
    return 1 / ans;
}
