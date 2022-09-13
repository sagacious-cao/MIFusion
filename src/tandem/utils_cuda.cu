#include "utils_cuda.h"

#include <stdio.h>
#include <assert.h> 

#include <cuda_runtime.h>

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) 
{
	if (code != cudaSuccess) 
    {
		fprintf(stderr,"GPUassert: %s %s %d\n",cudaGetErrorString(code), file, line);
		if (abort) 
            exit(code);
	} 
}


__global__ 
void maskCleanOverlapKernel(unsigned char* masks, const int masksNum,const int width, const int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	int flag = 0;
	for(int maskID = masksNum-1;maskID>=0;maskID--)
	{
		if(flag) 
            masks[ (maskID*width*height) + (x+y*width) ] = 0;
		if(masks[ (maskID*width*height) + (x+y*width) ]) 
            flag = 1;
	}
}

__host__ 
void maskCleanOverlap(unsigned char* masks, const int masksNum,const int width, const int height)
{
	const int blocks = 32;
	dim3 dimGrid(blocks,blocks);
	dim3 dimBlock(width/blocks,height/blocks);
	
	maskCleanOverlapKernel<<<dimGrid,dimBlock>>>(masks, masksNum, width, height);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}
