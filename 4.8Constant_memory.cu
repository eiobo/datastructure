/*

*/
#include "./common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>

#define c0 1
#define c1 2
#define c2 3
#define c3 4
#define c4 5

#define RADIUS 4
#define BLOCK_SIZE 128

__constant__ int coef[9];
const int h_coef[9] = {c0, c1, c2, c3, c4, c3, c2, c1, c0};


void setup_coefs()
{
	cudaMemcpyToSymbol(coef, h_coef, 9 * sizeof(int));
}

void host_const_calculation(int * in, int * out, int size)
{
	for (int i = 0; i < size; i++)
	{
		if (i < RADIUS)
		{
			out[i] = in[i + 4] * c0
				+ in[i + 3] * c1
				+ in[i + 2] * c2
				+ in[i + 1] * c3
				+ in[i] * c4;

			if (i == 3)
			{
				out[i] += in[2] * c3;
				out[i] += in[1] * c2;
				out[i] += in[0] * c1;
			}
			else if (i == 2)
			{
				out[i] += in[1] * c3;
				out[i] += in[0] * c2;
			}
			else if (i == 1)
			{
				out[i] += in[0] * c3;
			}
		}
		else if ((i + RADIUS) >= size)
		{
			out[i] = in[i - 4] * c0
				+ in[i - 3] * c1
				+ in[i - 2] * c2
				+ in[i - 1] * c3
				+ in[i] * c4;
			
			if (i == size - 4)
			{
				out[i] += in[size - 3] * c3;
				out[i] += in[size - 2] * c2;
				out[i] += in[size - 1] * c1;
			}
			else if (i == size -3)
			{
				out[i] += in[size - 2] * c3;
				out[i] += in[size - 1] * c2;
			}
			else if (i == size - 2)
			{
				out[i] += in[size - 1] * c3;
			}
		}
		else
		{
			out[i] = (in[i - 4] + in[i + 4])*c0
				+ (in[i - 3] + in[i + 3])*c1
				+ (in[i - 2] + in[i + 2])*c2
				+ (in[i - 1] + in[i + 1])*c3
				+ in[i] * c4;
		}
	}
}

__global__ void constant_stencil_smem_test(int * in, int * out, int size)
{
	__shared__ int smem[BLOCK_SIZE + 2 * RADIUS];			// 大小：128 + 8 = 136  索引：0 ~ 135
	int gid = threadIdx.x + blockIdx.x * blockDim.x;		// 0 ~ (1 << 22 - 1) =  (0 ~ 4194303)
	int bid = blockIdx.x;
	int num_of_block = gridDim.x;  							// len / blocksize = 1 << 22 / 128 
	int result = 0;

	if(gid < size)
	{
		// index with offset
		int sidx = threadIdx.x + RADIUS;  	// T0 ~ T127 -> smem[4] ~ smem[131]

		// load data to smem
		smem[sidx] = in[gid];		// smem[0] ~ smem[3] 为空， smem[4] ~ smem[131] 为数据， smem[132] ~ smem[135] 为空
		__syncthreads();

		// 并非首尾block
		if(bid != 0 && bid != num_of_block - 1)
		{
			if(threadIdx.x < RADIUS)
			{
				smem[sidx - RADIUS] = in[gid - RADIUS];				// 左侧边界 smem[0] ~ smem[3]
				smem[BLOCK_SIZE + sidx] = in[gid + BLOCK_SIZE];		// 右侧边界 smem[132] ~ smem[135]
			}
		}
		// 首block
		else if(bid == 0)
		{
			if(threadIdx.x < RADIUS)
			{
				smem[sidx - RADIUS] = 0;
				smem[BLOCK_SIZE + sidx] = in[gid + BLOCK_SIZE];
			}
		}
		// 尾block
		else if(bid == num_of_block - 1)
		{
			if(threadIdx.x < RADIUS)
			{
				smem[sidx - RADIUS] = in[gid - RADIUS];
				smem[BLOCK_SIZE + sidx] = 0;
			}
		}
		__syncthreads();
		
		// stencil calculation
		result += smem[sidx - RADIUS] * coef[0];
		result += smem[sidx - RADIUS + 1] * coef[1];
		result += smem[sidx - RADIUS + 2] * coef[2];
		result += smem[sidx - RADIUS + 3] * coef[3];
		result += smem[sidx] * coef[4];
		result += smem[sidx + RADIUS - 3] * coef[5];
		result += smem[sidx + RADIUS - 2] * coef[6];
		result += smem[sidx + RADIUS - 1] * coef[7];
		result += smem[sidx + RADIUS] * coef[8];

		out[gid] = result;

		// if(bid == 0)
		// 	printf("B0, T%d, sidx%d, smem[sidx] = %d\n", threadIdx.x, sidx, smem[sidx]);
		// if(bid == 1)
		// 	printf("B1, T%d, sidx%d, smem[sidx] = %d\n", threadIdx.x, sidx, smem[sidx]);

	}
	// if(gid == 4194303)
	// {
	// 	printf("num of block: %d\n", num_of_block);
	// 	printf("block id: %d\n", 1 << 22);
	// }
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

	// setup coefficients
	setup_coefs();

	// init data
	int len = 1 << 22;
	size_t nBytes = len * sizeof(int);
	int * h_in = (int *)malloc(nBytes);			// origin data
	int * h_out = (int *)malloc(nBytes);		// 接受GPU结果
	int * h_ref = (int *)malloc(nBytes);		// host上的参考值
	for(int i = 0; i < len; i++)
		h_in[i] = i;
	memset(h_out, 0, nBytes);
	host_const_calculation(h_in, h_ref, len);	// 求出参考值

	// copy data to device
	int* d_in, * d_out;
	ErrorCheck(cudaMalloc((void **)&d_in, nBytes),__FILE__, __LINE__);
	ErrorCheck(cudaMalloc((void **)&d_out, nBytes),__FILE__, __LINE__);
	ErrorCheck(cudaMemcpy(d_in, h_in, nBytes, cudaMemcpyHostToDevice),__FILE__, __LINE__);
	ErrorCheck(cudaMemset(d_out, 0, nBytes),__FILE__, __LINE__);

	// kernel parameters:
	dim3 block(BLOCK_SIZE);
	dim3 grid((len + block.x - 1) / block.x);
	printf("The kernel runs at grid (%d, %d, %d), block (%d, %d, %d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);

	// launch kernel
	constant_stencil_smem_test<<<grid, block>>>(d_in, d_out, len);
	ErrorCheck(cudaMemcpy(h_out, d_out, nBytes, cudaMemcpyDeviceToHost),__FILE__, __LINE__);

	// for(int i = 0; i < 30; i++)
	// 	printf("%d ", h_out[i]);

	// printf("\n");

	// for(int i = 0; i < 30; i++)
	// 	printf("%d ", h_ref[i]);
	// check result
	checkresult<int>(h_out, h_ref, len);

	// free host memory
    free(h_in);
    free(h_out);

    // free device memory
    cudaFree(d_in);
    cudaFree(d_out);

    // reset device
    cudaDeviceReset();
    return 0;

    // reset device
    cudaDeviceReset();
    return 0;
}