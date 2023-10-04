#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

extern "C" {
#include <stdio.h>

__host__ void host_get_vector_length(float* coordinates, float* vector_length, int N) {
    *vector_length = 0;
    for(int i = 0; i < N; i++) {
        *vector_length += pow(coordinates[i], 2);
    }
    *vector_length = sqrtf(*vector_length);
}

__global__ void device_get_vector_length(float* coordinates, float* vector_length, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i == 0) {
        *vector_length = 0;
    }
    if (i < N) {
        atomicAdd(vector_length, (coordinates[i]*coordinates[i]));
    }
}

__global__ void d_fill_uniform(
    float *a, int n, float r, unsigned long long seed) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        curandState_t state;
        curand_init(seed, i, 0, &state);

        a[i] = -r + 2 * r * curand_uniform(&state);
    }
}

}

#ifndef REDEFINE
    #define VEC_LEN 51200000
    #define VEC_LEN_INC 512000
    #define CHECK_FIRST 51200
    #define BLOCK_SIZE 128
    #define FNAME_STAMPS "timings.stmp"
    #define PRECISION 10e-2
    #define SEED 27
    #define VEC_MAX_ABS_VAL 101
#endif

#define VEC_MEM_SIZE (VEC_LEN * sizeof(float))
#define calc_grid_size(m) ((m + BLOCK_SIZE - 1) / BLOCK_SIZE)
#define ts_to_ms(ts) (ts.tv_sec * 10e3 + ts.tv_nsec * 10e-6)


int main() {
    float *host_coordinates __attribute__ ((aligned (64)));
    float *host_vector_length __attribute__ ((aligned (64)));
    float *host_fromdevice_vector_length __attribute__ ((aligned (64)));

    host_coordinates = (float*)malloc(VEC_MEM_SIZE);
    host_vector_length = (float*)malloc(sizeof(float));
    host_fromdevice_vector_length = (float*)malloc(sizeof(float));

    float *device_coordinates, *device_vector_length;
    cudaMalloc((void**)&device_coordinates, VEC_MEM_SIZE);
    cudaMalloc((void**)&device_vector_length, sizeof(float));

    d_fill_uniform<<<calc_grid_size(VEC_LEN), BLOCK_SIZE>>>(device_coordinates, VEC_LEN, VEC_MAX_ABS_VAL, SEED);
    cudaMemcpy(host_coordinates, device_coordinates, VEC_MEM_SIZE, cudaMemcpyDeviceToHost);

    host_get_vector_length(host_coordinates, host_vector_length, CHECK_FIRST);
    device_get_vector_length<<<calc_grid_size(CHECK_FIRST), BLOCK_SIZE>>>(device_coordinates, device_vector_length, CHECK_FIRST);

    cudaMemcpy(host_fromdevice_vector_length, device_vector_length, sizeof(float), cudaMemcpyDeviceToHost);
    *host_fromdevice_vector_length = sqrtf(*host_fromdevice_vector_length);

    if (fabs(*host_vector_length - *host_fromdevice_vector_length) > PRECISION) {
       printf("Panic!\n");
       return -1;
    }

    printf("h_length: %f\n", *host_vector_length);
    printf("d_length: %f\n", *host_fromdevice_vector_length);