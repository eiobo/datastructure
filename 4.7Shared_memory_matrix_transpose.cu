/*

*/
#include "./common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>

#define BDIM_X 64 
#define BDIM_Y 8

// transpose with global memory
__global__ void transpose_GMEM(int *input, int *output, int nx, int ny)
{
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;

	output[ix * ny + iy] = input[iy * nx + ix];		// 对显存读是连续的，写是不连续的，所以效率低
}

__global__ void transpose_SMEM(int *input, int *output, int nx, int ny)
{
	__shared__ int tile[BDIM_Y][BDIM_X];

	// cache the input matrix to shared memory
	int ix = threadIdx.x + blockIdx.x * blockDim.x;
	int iy = threadIdx.y + blockIdx.y * blockDim.y;
	int GMEM_gid_in = iy * nx + ix;
	if(ix < nx && iy < ny)
	{
		tile[threadIdx.y][threadIdx.x] = input[GMEM_gid_in];	// threadIdx.y : 0~7, threadIdx.x : 0~63
		__syncthreads();
	}
	
	// write the child pieces transposed matrix to global memory
	int SMEM_gid = threadIdx.y * blockDim.x + threadIdx.x;
	int i_row = SMEM_gid / blockDim.y;
	int i_col = SMEM_gid % blockDim.y;

	ix = blockDim.y * blockIdx.y + i_col;
	iy = blockDim.x * blockIdx.x + i_row;

	int GMEM_gid_out = iy * ny + ix;
	if(ix < ny && iy < nx)
	{
		output[GMEM_gid_out] = tile[i_col][i_row];	// threadIdx.x : 0~7, threadIdx.y : 0~63
	}

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
	int nx = 1024;
	int ny = 1024;
	int num_elements = nx * ny;
	int nbytes = num_elements * sizeof(int);
	printf("Matrix size: nx %d ny %d, is transposed with BLOCK :blockdim.x %d, blockdim.y %d\n", nx, ny, BDIM_X, BDIM_Y);

	// set up data
	int *h_A = (int *)malloc(nbytes);			// hold the original matrix
	int *h_B = (int *)malloc(nbytes);			// hold the transposed matrix done by cpu
	int *h_C = (int *)malloc(nbytes);			// hold the transposed matrix done by gpu
	InitArrayElem(h_A, num_elements);
	memset(h_B, 0, nbytes);
	memset(h_C, 0, nbytes);

	// do the transposition on cpu
	for(int i = 0; i < ny; i++)
		for(int j = 0; j < nx; j++)
			h_B[i * nx + j] = h_A[j * ny + i];
	
	// allocate memory on device
	int *d_A;					// GMEM to hold the original matrix
	int *d_B;					// GMEM to hold the transposed matrix 
	ErrorCheck(cudaMalloc((void **)&d_A, nbytes),__FILE__, __LINE__);
	ErrorCheck(cudaMalloc((void **)&d_B, nbytes),__FILE__, __LINE__);

	// copy data from host to device
	ErrorCheck(cudaMemcpy(d_A, h_A, nbytes, cudaMemcpyHostToDevice),__FILE__, __LINE__);
	ErrorCheck(cudaMemset(d_B, 0, nbytes),__FILE__, __LINE__);
	
	// setup launch parameters
	dim3 block(BDIM_X, BDIM_Y);
	dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

	// kernel 1: naive transpose
	printf("Testing the kernel1: transpose_GMEM<<<>>>  \n");
    transpose_GMEM<<<grid, block>>>(d_A, d_B, nx, ny);
	ErrorCheck((cudaMemcpy(h_C, d_B, nbytes, cudaMemcpyDeviceToHost)),__FILE__, __LINE__);
	// kernel 1 check result
	checkresult<int>(h_B, h_C, num_elements);
	cudaMemset(d_B, 0, nbytes);
	memset(h_C, 0, nbytes);

	// kernel 2: shared memory transpose
	printf("Testing the kernel1: transpose_SMEM<<<>>>  \n");
	transpose_SMEM<<<grid, block, BDIM_X*BDIM_Y*sizeof(int)>>>(d_A, d_B, nx, ny);
	ErrorCheck((cudaMemcpy(h_C, d_B, nbytes, cudaMemcpyDeviceToHost)),__FILE__, __LINE__);
	// kernel 2 check result
	checkresult<int>(h_B, h_C, num_elements);

	// free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);

    // reset device
    cudaDeviceReset();
    return 0;
}