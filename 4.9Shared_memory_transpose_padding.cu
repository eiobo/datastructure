/*========================Profile with:
ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,
              l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,
              l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum,
              l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum  
              ./exefile/4.9Shared_memory_transpose_padding
*/


#include "./common/common.h"
#include <stdio.h>
#include <cuda_runtime.h>

#define BDIM_X 64   // Block.x
#define BDIM_Y 8    // Block.y
#define PAD 2

// transpose with global memory
// 原理：大矩阵转置等于子矩阵分别转置再拼接
__global__ void transpose_GMEM(int *input, int *output, int nx, int ny)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;

    if(ix < nx && iy < ny)                              // bound check
        output[ix * ny + iy] = input[iy * nx + ix];
}

__global__ void transpose_SMEM(int *input, int *output, int nx, int ny)
{
    __shared__ int SMEM[BDIM_Y][BDIM_X];               // shared memory, 8 rows, 64 columns, 和block重合
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int GMEM_input_gid = iy * nx + ix;

    // cache data into shared memory from global memory
    if(ix < nx && iy < ny)                              // bound check
    {
        SMEM[threadIdx.y][threadIdx.x] = input[GMEM_input_gid];
        __syncthreads();
    }

    int SMEM_gid = threadIdx.y * blockDim.x + threadIdx.x;  // SMEM的全局线性索引
    int i_row = SMEM_gid / blockDim.y;                      // SMEM转置之后的行索引
    int i_col = SMEM_gid % blockDim.y;                      // SMEM转置之后的列索引

    ix = i_col + blockIdx.y * blockDim.y;                   // SMEM在全局矩阵中的ix
    iy = i_row + blockIdx.x * blockDim.x;                   // SMEM在全局矩阵中的iy
    
    int GMEM_gid_out = iy * ny + ix;                        // 按行排列的全局索引

    if(ix < ny && iy < nx)                                  // bound check
        output[GMEM_gid_out] = SMEM[i_col][i_row];          

    // if(blockIdx.x == 0 && blockIdx.y == 0)                  // block(0, 0)的线程打印SMEM
    //     printf("SMEM_gid = %d, SMEM[%d][%d] = %d\n", SMEM_gid, threadIdx.y, threadIdx.x, SMEM[threadIdx.y][threadIdx.x]);
    // if(blockIdx.x == 0 && blockIdx.y == 0)                  // block(0, 0)的线程打印SMEM
    //     printf("SMEM_gid = %d, SMEM[%d][%d] = %d\n", SMEM_gid, threadIdx.y, threadIdx.x, tile[threadIdx.y][threadIdx.x]);
}

__global__ void transpose_SMEM_Padding(int *input, int *output, int nx, int ny)
{
    __shared__ int SMEM[BDIM_Y][BDIM_X + PAD];               // shared memory, 8 rows, 64 columns, 和block重合
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int GMEM_input_gid = iy * nx + ix;

    // cache data into shared memory from global memory
    if(ix < nx && iy < ny)                              // bound check
    {
        SMEM[threadIdx.y][threadIdx.x] = input[GMEM_input_gid];
        __syncthreads();
    }

    int SMEM_gid = threadIdx.y * blockDim.x + threadIdx.x;  // SMEM的全局线性索引
    int i_row = SMEM_gid / blockDim.y;                      // SMEM转置之后的行索引
    int i_col = SMEM_gid % blockDim.y;                      // SMEM转置之后的列索引

    ix = i_col + blockIdx.y * blockDim.y;                   // SMEM在全局矩阵中的ix
    iy = i_row + blockIdx.x * blockDim.x;                   // SMEM在全局矩阵中的iy
    
    int GMEM_gid_out = iy * ny + ix;                        // 按行排列的全局索引

    if(ix < ny && iy < nx)                                  // bound check
        output[GMEM_gid_out] = SMEM[i_col][i_row];          

    // if(blockIdx.x == 0 && blockIdx.y == 0)                  // block(0, 0)的线程打印SMEM
    //     printf("SMEM_gid = %d, SMEM[%d][%d] = %d\n", SMEM_gid, threadIdx.y, threadIdx.x, SMEM[threadIdx.y][threadIdx.x]);
    // if(blockIdx.x == 0 && blockIdx.y == 0)                  // block(0, 0)的线程打印SMEM
    //     printf("SMEM_gid = %d, SMEM[%d][%d] = %d\n", SMEM_gid, threadIdx.y, threadIdx.x, tile[threadIdx.y][threadIdx.x]);
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

    // setup data size
    int nx = 1 << 10;
    int ny = 1 << 10;
    int nelem = nx * ny;
    size_t nbytes = nelem * sizeof(int);
    printf("Matrix to be transposed is size: %d x %d\n", nx, ny);

    // setup data on host
    int *h_input = (int *)malloc(nbytes);       // origin matrix
    int *h_output = (int *)malloc(nbytes);      // output matrix by cpu
    int *gpu_ref = (int *)malloc(nbytes);       // output matrix by gpu
    for(int i = 0; i < nelem; i++)              // 数组的值就是他的全局id，方便调试学习 
        h_input[i] = i;
    // InitArrayElem(h_output, nelem);          // 随机值初始化数组
    memset(h_output, 0, nbytes);
    memset(gpu_ref, 0, nbytes);
    
    // do transpose on host
    for(int ix = 0; ix < nx; ix++)
        for(int iy = 0; iy < ny; iy++)
            h_output[iy * nx + ix] = h_input[ix * ny + iy];
    
    // host transpose result
    // printMatrix_2D_INT(h_input, 15, 15);
    // printf("\n");
    // printMatrix_2D_INT(h_output, 15, 15);

    
    // setup data on device
    int *d_input, *d_output;
    cudaMalloc((void **)&d_input, nbytes);
    cudaMalloc((void **)&d_output, nbytes);
    cudaMemset(d_output, 0, nbytes);
    cudaMemcpy(d_input, h_input, nbytes, cudaMemcpyHostToDevice);

    // Inovke the kernel: transpose_GMEM
    dim3 block(BDIM_X, BDIM_Y);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    printf("Kernel: transpose_GMEM<<<>>> launch with grid(%d, %d), block(%d, %d)\n", grid.x, grid.y, block.x, block.y);
    transpose_GMEM<<<grid, block>>>(d_input, d_output, nx, ny);
    cudaMemcpy(gpu_ref, d_output, nbytes, cudaMemcpyDeviceToHost);
    // check result
    checkresult<int>(h_output, gpu_ref, nelem);

    cudaMemset(d_output, 0, nbytes);
    memset(gpu_ref, 0, nbytes);

    // Invoke the kernel: transpose_SMEM
    printf("Kernel: transpose_SMEM<<<>>> launch with grid(%d, %d), block(%d, %d)\n", grid.x, grid.y, block.x, block.y);
    transpose_SMEM<<<grid, block>>>(d_input, d_output, nx, ny);
    cudaMemcpy(gpu_ref, d_output, nbytes, cudaMemcpyDeviceToHost);
    // check result
    checkresult<int>(h_output, gpu_ref, nelem);

    cudaMemset(d_output, 0, nbytes);
    memset(gpu_ref, 0, nbytes);

    // Invoke the kernel: transpose_SMEM_Padding
    printf("Kernel: transpose_SMEM_Padding<<<>>> launch with grid(%d, %d), block(%d, %d)\n", grid.x, grid.y, block.x, block.y);
    transpose_SMEM_Padding<<<grid, block>>>(d_input, d_output, nx, ny);
    cudaMemcpy(gpu_ref, d_output, nbytes, cudaMemcpyDeviceToHost);
    // check result
    checkresult<int>(h_output, gpu_ref, nelem);


    // free memory
    free(h_input);
    free(h_output);
    free(gpu_ref);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
// 
// shared_load_transactions
// shared_store_transactions
// gst_transactions
// gld_transactions
/*==========================================Profiling Result==========================================================
    Section: Command line profiler metrics
    ---------------------------------------------------- ----------- ------------
    Metric Name                                          Metric Unit Metric Value
    ---------------------------------------------------- ----------- ------------
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                        0
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                        0
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum            sector       131072
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum            sector      1048576
    ---------------------------------------------------- ----------- ------------

  transpose_SMEM(int *, int *, int, int) (16, 128, 1)x(64, 8, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    ---------------------------------------------------- ----------- ------------
    Metric Name                                          Metric Unit Metric Value
    ---------------------------------------------------- ----------- ------------
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                   266246
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                    33446
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum            sector       131072
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum            sector       131072
    ---------------------------------------------------- ----------- ------------

  transpose_SMEM_Padding(int *, int *, int, int) (16, 128, 1)x(64, 8, 1), Context 1, Stream 7, Device 0, CC 8.9
    Section: Command line profiler metrics
    ---------------------------------------------------- ----------- ------------
    Metric Name                                          Metric Unit Metric Value
    ---------------------------------------------------- ----------- ------------
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum                    66910
    l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum                    33382
    l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum            sector       131072
    l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum            sector       131072
    ---------------------------------------------------- ----------- ------------
*/


/*
Block(0, 0)的线程打印SMEM:
SMEM_gid = 0, SMEM[0][0] = 0
SMEM_gid = 1, SMEM[0][1] = 1
SMEM_gid = 2, SMEM[0][2] = 2
SMEM_gid = 3, SMEM[0][3] = 3
SMEM_gid = 4, SMEM[0][4] = 4
SMEM_gid = 5, SMEM[0][5] = 5
SMEM_gid = 6, SMEM[0][6] = 6
SMEM_gid = 7, SMEM[0][7] = 7
SMEM_gid = 8, SMEM[0][8] = 8
SMEM_gid = 9, SMEM[0][9] = 9
SMEM_gid = 10, SMEM[0][10] = 10
SMEM_gid = 11, SMEM[0][11] = 11
SMEM_gid = 12, SMEM[0][12] = 12
SMEM_gid = 13, SMEM[0][13] = 13
SMEM_gid = 14, SMEM[0][14] = 14
SMEM_gid = 15, SMEM[0][15] = 15
SMEM_gid = 16, SMEM[0][16] = 16
SMEM_gid = 17, SMEM[0][17] = 17
SMEM_gid = 18, SMEM[0][18] = 18
SMEM_gid = 19, SMEM[0][19] = 19
SMEM_gid = 20, SMEM[0][20] = 20
SMEM_gid = 21, SMEM[0][21] = 21
SMEM_gid = 22, SMEM[0][22] = 22
SMEM_gid = 23, SMEM[0][23] = 23
SMEM_gid = 24, SMEM[0][24] = 24
SMEM_gid = 25, SMEM[0][25] = 25
SMEM_gid = 26, SMEM[0][26] = 26
SMEM_gid = 27, SMEM[0][27] = 27
SMEM_gid = 28, SMEM[0][28] = 28
SMEM_gid = 29, SMEM[0][29] = 29
SMEM_gid = 30, SMEM[0][30] = 30
SMEM_gid = 31, SMEM[0][31] = 31
SMEM_gid = 32, SMEM[0][32] = 32
SMEM_gid = 33, SMEM[0][33] = 33
SMEM_gid = 34, SMEM[0][34] = 34
SMEM_gid = 35, SMEM[0][35] = 35
SMEM_gid = 36, SMEM[0][36] = 36
SMEM_gid = 37, SMEM[0][37] = 37
SMEM_gid = 38, SMEM[0][38] = 38
SMEM_gid = 39, SMEM[0][39] = 39
SMEM_gid = 40, SMEM[0][40] = 40
SMEM_gid = 41, SMEM[0][41] = 41
SMEM_gid = 42, SMEM[0][42] = 42
SMEM_gid = 43, SMEM[0][43] = 43
SMEM_gid = 44, SMEM[0][44] = 44
SMEM_gid = 45, SMEM[0][45] = 45
SMEM_gid = 46, SMEM[0][46] = 46
SMEM_gid = 47, SMEM[0][47] = 47
SMEM_gid = 48, SMEM[0][48] = 48
SMEM_gid = 49, SMEM[0][49] = 49
SMEM_gid = 50, SMEM[0][50] = 50
SMEM_gid = 51, SMEM[0][51] = 51
SMEM_gid = 52, SMEM[0][52] = 52
SMEM_gid = 53, SMEM[0][53] = 53
SMEM_gid = 54, SMEM[0][54] = 54
SMEM_gid = 55, SMEM[0][55] = 55
SMEM_gid = 56, SMEM[0][56] = 56
SMEM_gid = 57, SMEM[0][57] = 57
SMEM_gid = 58, SMEM[0][58] = 58
SMEM_gid = 59, SMEM[0][59] = 59
SMEM_gid = 60, SMEM[0][60] = 60
SMEM_gid = 61, SMEM[0][61] = 61
SMEM_gid = 62, SMEM[0][62] = 62
SMEM_gid = 63, SMEM[0][63] = 63
*/