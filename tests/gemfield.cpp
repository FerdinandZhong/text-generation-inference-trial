#include <stdio.h>
#include <cuda_runtime.h>
int main() {
    int device = 0;
    int gpuDeviceCount = 0;
    struct cudaDeviceProp properties;

    cudaError_t cudaResultCode = cudaGetDeviceCount(&gpuDeviceCount);

    if (cudaResultCode == cudaSuccess){
        cudaGetDeviceProperties(&properties, device);
        printf("%d GPU CUDA devices(s)(%d)\n", gpuDeviceCount, properties.major);
        printf("\t Product Name: %s\n"          , properties.name);
        printf("\t TotalGlobalMem: %ld MB\n"    , properties.totalGlobalMem/(1024^2));
        printf("\t GPU Count: %d\n"             , properties.multiProcessorCount);
        printf("\t Kernels found: %d\n"         , properties.concurrentKernels);
        return 0;
    }
    printf("\t gemfield error: %d\n",cudaResultCode);
}