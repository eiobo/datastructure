
#include "./common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE 64
#define FULL_MASK 0xffffffff

__global__ 
void test_shfl_broadcast_32(int* in, int* out)
{
    int tid = threadIdx.x;                                      // tid: 0~63
    int value = in[tid];                                        // 
    int value_broad = __shfl_sync(FULL_MASK, value, 3, 32);     // 将第3个线程的值广播到所有线程
    out[tid] = value_broad;
}

__global__
void test_shfl_broadcast_16(int* in, int* out)
{
    int tid = threadIdx.x;                                      // tid: 0~63
    int value = in[tid];                                        // 
    int value_broad = __shfl_sync(FULL_MASK, value, 3, 16);     // 将第3个线程的值广播到所有线程
    out[tid] = value_broad;
}

__global__ 
void test_shfl_up_2(int* in, int* out)
{
    int tid = threadIdx.x;                                      // tid: 0~63
    int value = in[tid];                                        // 
    int value_up = __shfl_up_sync(FULL_MASK, value, 2, 32);     // 将所有线程的值向上广播2个单位
    out[tid] = value_up;
}

__global__ 
void test_shfl_down_2(int* in, int* out)
{
    int tid = threadIdx.x;                                      // tid: 0~63
    int value = in[tid];                                        // 
    int value_down = __shfl_down_sync(FULL_MASK, value, 2, 32); // 将第2个线程的值向下广播到所有线程
    out[tid] = value_down;
}

__global__ 
void test_shfl_xor(int* in, int* out)
{
    int tid = threadIdx.x;                                      // tid: 0~63
    int value = in[tid];                                        // 
    int value_xor = __shfl_xor_sync(FULL_MASK, value, 1, 32);
    out[tid] = value_xor;
}

int main(int argc, char **argv)
{
    // get device count
    int nDeviceCount = 0;
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&nDeviceCount),__FILE__, __LINE__);
    if (error != cudaSuccess || nDeviceCount == 0)
    {
        printf("There is no device supporting CUDA.\n");
        return -1; 
    }

    // set up device
    int dev = 0;
    error =  ErrorCheck(cudaSetDevice(dev),__FILE__, __LINE__);
    if(error != cudaSuccess)
        printf("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
    else
        printf("CUDA Device %d ready for computing!\n", dev);

    // set up data size
    int nelems = ARRAY_SIZE;
    size_t nbytes = nelems * sizeof(int);

    // set up data on host
    int *h_data = (int *)malloc(nbytes);
    int *gpu_ref = (int *)malloc(nbytes);
    memset(h_data, 0, nbytes);
    memset(gpu_ref, 0, nbytes);

    // init the input data
    for(int i = 0; i < nelems; i++)         // 输入数组中存放0 - 31
        h_data[i] = i;
    // InitArrayElem(h_data, nelems);

    // print origin data
    printf("origin data:\n");
    for(int i = 0; i < nelems; i++)
        printf("%d ", h_data[i]);
    printf("\n");

    // Move the data from host to device
    int *d_data;
    int *d_out;
    ErrorCheck(cudaMalloc((void **)&d_data, nbytes),__FILE__, __LINE__);
    ErrorCheck(cudaMemcpy(d_data, h_data, nbytes, cudaMemcpyHostToDevice),__FILE__, __LINE__);
    ErrorCheck(cudaMalloc((void**)&d_out, nbytes), __FILE__, __LINE__);

    // set up thread block and grid size
    dim3 block(ARRAY_SIZE);
    dim3 grid(1);

    printf("Launching kernel: test_shfl_broadcast_32: \n");
    // launch the kernel: test_shfl_broadcast_32
    test_shfl_broadcast_32<<<grid, block>>>(d_data, d_out);

    // copy the result back to host
    ErrorCheck(cudaMemcpy(gpu_ref, d_out, nbytes, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    // verify the result
    printf("gpu result 1:\n");
    for(int i = 0; i < nelems; i++)
        printf("%d ", gpu_ref[i]);
    printf("\n");

    // Set to zero
    cudaMemset(d_out, 0, nbytes);
    memset(gpu_ref, 0, nbytes);

    // launch the kernel: 
    printf("Launching kernel: test_shfl_broadcast_16: \n");
    test_shfl_broadcast_16<<<grid, block>>>(d_data, d_out);
    ErrorCheck(cudaMemcpy(gpu_ref, d_out, nbytes, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    // verify the result
    printf("gpu result 2:\n");
    for(int i = 0; i < nelems; i++)
        printf("%d ", gpu_ref[i]);
    printf("\n");

    // Set to zero
    cudaMemset(d_out, 0, nbytes);
    memset(gpu_ref, 0, nbytes);

    // launch the kernel: test_shfl_up_2
    printf("Launching kernel: test_shfl_up_2: \n");
    test_shfl_up_2<<<grid, block>>>(d_data, d_out);
    ErrorCheck(cudaMemcpy(gpu_ref, d_out, nbytes, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    // verify the result
    printf("gpu result 3:\n");
    for(int i = 0; i < nelems; i++)
        printf("%d ", gpu_ref[i]);
    printf("\n");

    // Set to zero
    cudaMemset(d_out, 0, nbytes);
    memset(gpu_ref, 0, nbytes);

    // launch the kernel: test_shfl_down_2
    printf("Launching kernel: test_shfl_down_2: \n");
    test_shfl_down_2<<<grid, block>>>(d_data, d_out);
    ErrorCheck(cudaMemcpy(gpu_ref, d_out, nbytes, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    // verify the result
    printf("gpu result 4:\n");
    for(int i = 0; i < nelems; i++)
        printf("%d ", gpu_ref[i]);
    printf("\n");

    // Set to zero
    cudaMemset(d_out, 0, nbytes);
    memset(gpu_ref, 0, nbytes);

    // launch the kernel: test_shfl_down_2
    printf("Launching kernel: test_shfl_xor: \n");
    test_shfl_xor<<<grid, block>>>(d_data, d_out);
    ErrorCheck(cudaMemcpy(gpu_ref, d_out, nbytes, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    // verify the result
    printf("gpu result 5:\n");
    for(int i = 0; i < nelems; i++)
        printf("%d ", gpu_ref[i]);
    printf("\n");

    // Set to zero
    cudaMemset(d_out, 0, nbytes);
    memset(gpu_ref, 0, nbytes);


    // free memory
    free(h_data);
    free(gpu_ref);
    cudaFree(d_data);
    cudaFree(d_out);
    
    return 0;
}

/*
origin data:
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 
Launching kernel: test_shfl_broadcast_32: 
gpu result 1:
3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 35 35 35 35 35 35 35 35 35 35 35 35 35 35 35 35 35 35 35 35 35 35 35 35 35 35 35 35 35 35 35 35 
Launching kernel: test_shfl_broadcast_16: 
gpu result 2:
3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 19 19 19 19 19 19 19 19 19 19 19 19 19 19 19 19 35 35 35 35 35 35 35 35 35 35 35 35 35 35 35 35 51 51 51 51 51 51 51 51 51 51 51 51 51 51 51 51 
Launching kernel: test_shfl_up_2: 
gpu result 3:
0 1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 32 33 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 
Launching kernel: test_shfl_down_2: 
gpu result 4:
2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 30 31 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 62 63 
Launching kernel: test_shfl_xor: 
gpu result 5:
1 0 3 2 5 4 7 6 9 8 11 10 13 12 15 14 17 16 19 18 21 20 23 22 25 24 27 26 29 28 31 30 33 32 35 34 37 36 39 38 41 40 43 42 45 44 47 46 49 48 51 50 53 52 55 54 57 56 59 58 61 60 63 62 
*/