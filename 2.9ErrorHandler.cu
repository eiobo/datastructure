#include <stdio.h>
#include "common/common.h"

// __FILE__ 和 __LINE__ 是预定义宏，用于获取当前文件名和代码行号
int main(int argc, char **argv)
{
    float* gpuMemory = NULL;    // 定义GPU空内存指针
    ErrorCheck(cudaMalloc(&gpuMemory, sizeof(float)), __FILE__, __LINE__);  // 分配GPU内存
    ErrorCheck(cudaFree(gpuMemory), __FILE__, __LINE__);        // 释放GPU内存
    ErrorCheck(cudaFree(gpuMemory), __FILE__, __LINE__);        // 再次释放GPU内存，模拟出错误
    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);

    return 1;
}