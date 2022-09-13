#pragma once
#include <cuda_runtime.h>
#include <cuda.h>

//============ Mask Post processing ��Depth�ϲ�rgb�����طָ���===============================================
void depthMapGaussianfilter(unsigned short *oriDepthMap, const int width, const int height, unsigned short * depthMapG);

//CudaTask SuperPiexl

void getPosMapFromDepth(unsigned short *depthMap, float* cam, const int width, const int height, float* posMap);

void getNormalMapFromDepth(unsigned short *depthMap, float* cam, const int width, const int height, float*normalMap);

void getSuperPixelInfoCuda(int* segMask, unsigned short * depthMap, float* posMap, float* normalMap,
	int spNum, float* spInfo, int* spInfoStruct, const int width, const int height);

void getFinalSuperPiexl(float* spInfo, const int width, const int height, int* spInfoStruct, int* segMask);