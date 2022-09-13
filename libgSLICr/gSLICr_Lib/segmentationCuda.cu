#include <stdio.h>
#include <assert.h> 

#include <cuda_runtime.h>
#include "segmentationCuda.h"

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool
	abort = true) {
	if (code != cudaSuccess) {
		fprintf(stderr, "GPUassert: %s %s %d\n",
			cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__inline__ __device__
bool checkNeighbours(unsigned short *map, const int x, const int y, const int width, const int height)
{

	if (x + 1 >= width)	return false;
	if (x - 1<0)		return false;
	if (y + 1 >= height)	return false;
	if (y - 1<0)		return false;

	if (!map[y*width + x + 1])     return false;
	if (!map[y*width + x - 1])     return false;
	if (!map[(y + 1)*width + x])   return false;
	if (!map[(y - 1)*width + x])   return false;

	if (!map[(y + 1)*width + x + 1])   return false;
	if (!map[(y + 1)*width + x - 1])   return false;
	if (!map[(y - 1)*width + x + 1])   return false;
	if (!map[(y - 1)*width + x - 1])   return false;

	return true;
}

__inline__ __device__
void getVertex(unsigned short *depth, const int x, const int y, const int width, const int height, float* cam, float* vertex)
{
	float z = float(depth[y*width + x]) / 1186.0f;
	vertex[0] = (x - cam[0]) * z * cam[2];
	vertex[1] = (y - cam[1]) * z * cam[3];
	vertex[2] = z;
}

__inline__ __device__
void getNormalCross(float* left, float* right, float* up, float* down, float* ans)
{
	float del_x[3];
	float del_y[3];
	//del_x
	del_x[0] = left[0] - right[0];
	del_x[1] = left[1] - right[1];
	del_x[2] = left[2] - right[2];
	//del_y
	del_y[0] = up[0] - down[0];
	del_y[1] = up[1] - down[1];
	del_y[2] = up[2] - down[2];
	//ans
	ans[0] = del_x[1] * del_y[2] - del_x[2] * del_y[1];
	ans[1] = del_x[2] * del_y[0] - del_x[0] * del_y[2];
	ans[2] = del_x[0] * del_y[1] - del_x[1] * del_y[0];
}

__inline__ __device__
void getNormal(unsigned short *depth, const int x, const int y, const int width, const int height, float* cam, float* nor)
{
	float vPosition[3];
	getVertex(depth, x, y, width, height, cam, vPosition);

	float vPosition_xf[3];
	float vPosition_xb[3];
	//get
	getVertex(depth, x + 1, y, width, height, cam, vPosition_xf);
	getVertex(depth, x - 1, y, width, height, cam, vPosition_xb);
	//xb
	//vPosition_xb[0] = (vPosition_xb[0] + vPosition[0]) / 2;
	//vPosition_xb[1] = (vPosition_xb[1] + vPosition[1]) / 2;
	//vPosition_xb[2] = (vPosition_xb[2] + vPosition[2]) / 2;
	//xf
	//vPosition_xf[0] = (vPosition_xf[0] + vPosition[0]) / 2;
	//vPosition_xf[1] = (vPosition_xf[1] + vPosition[1]) / 2;
	//vPosition_xf[2] = (vPosition_xf[2] + vPosition[2]) / 2;

	float vPosition_yf[3];
	float vPosition_yb[3];
	getVertex(depth, x, y + 1, width, height, cam, vPosition_yf);
	getVertex(depth, x, y - 1, width, height, cam, vPosition_yb);
	//yb
	//vPosition_yb[0] = (vPosition_yb[0] + vPosition[0]) / 2;
	//vPosition_yb[1] = (vPosition_yb[1] + vPosition[1]) / 2;
	//vPosition_yb[2] = (vPosition_yb[2] + vPosition[2]) / 2;
	//yf
	//vPosition_yf[0] = (vPosition_yf[0] + vPosition[0]) / 2;
	//vPosition_yf[1] = (vPosition_yf[1] + vPosition[1]) / 2;
	//vPosition_yf[2] = (vPosition_yf[2] + vPosition[2]) / 2;

	float temp[3];
	float sum[3];
	getNormalCross(vPosition_xb, vPosition_xf, vPosition_yb, vPosition_yf, temp);
	sum[0] = temp[0] * 4;
	sum[1] = temp[1] * 4;
	sum[2] = temp[2] * 4;

	getNormalCross(vPosition_xb, vPosition, vPosition_yb, vPosition, temp);
	sum[0] += temp[0] * 2;
	sum[1] += temp[1] * 2;
	sum[2] += temp[2] * 2;

	getNormalCross(vPosition, vPosition_xf, vPosition_yb, vPosition, temp);
	sum[0] += temp[0] * 2;
	sum[1] += temp[1] * 2;
	sum[2] += temp[2] * 2;

	getNormalCross(vPosition_xb, vPosition, vPosition, vPosition_yf, temp);
	sum[0] += temp[0] * 2;
	sum[1] += temp[1] * 2;
	sum[2] += temp[2] * 2;

	getNormalCross(vPosition, vPosition_xf, vPosition, vPosition_yf, temp);
	sum[0] += temp[0] * 2;
	sum[1] += temp[1] * 2;
	sum[2] += temp[2] * 2;

	float len = sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
	nor[0] = sum[0] / len;
	nor[1] = sum[1] / len;
	nor[2] = sum[2] / len;
}




__global__
void depthMapGaussianfilterKernel(unsigned short *oriDepthMap, const int width, const int height, unsigned short * depthMapG)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (checkNeighbours(oriDepthMap, x, y, width, height))
	{
		int sum = 0;
		int n = 0;
		if (oriDepthMap[(y)*width + (x)]) { n += 4; sum += 4 * oriDepthMap[(y)*width + (x)]; }

		if (oriDepthMap[(y)*width + (x + 1)]) { n += 2; sum += 2 * oriDepthMap[(y)*width + (x + 1)]; }
		if (oriDepthMap[(y)*width + (x - 1)]) { n += 2; sum += 2 * oriDepthMap[(y)*width + (x - 1)]; }
		if (oriDepthMap[(y + 1)*width + (x)]) { n += 2; sum += 2 * oriDepthMap[(y + 1)*width + (x)]; }
		if (oriDepthMap[(y - 1)*width + (x)]) { n += 2; sum += 2 * oriDepthMap[(y - 1)*width + (x)]; }

		if (oriDepthMap[(y + 1)*width + (x + 1)]) { n += 1; sum += oriDepthMap[(y + 1)*width + (x + 1)]; }
		if (oriDepthMap[(y + 1)*width + (x - 1)]) { n += 1; sum += oriDepthMap[(y + 1)*width + (x - 1)]; }
		if (oriDepthMap[(y - 1)*width + (x + 1)]) { n += 1; sum += oriDepthMap[(y - 1)*width + (x + 1)]; }
		if (oriDepthMap[(y - 1)*width + (x - 1)]) { n += 1; sum += oriDepthMap[(y - 1)*width + (x - 1)]; }

		if (n) depthMapG[y*width + x] = sum / n;
	}
}

__host__
void depthMapGaussianfilter(unsigned short *oriDepthMap, const int width, const int height, unsigned short * depthMapG)
{
	const int blocks = 32;
	dim3 dimGrid(blocks, blocks);
	dim3 dimBlock(width / blocks, height / blocks);

	depthMapGaussianfilterKernel << <dimGrid, dimBlock >> >(oriDepthMap, width, height, depthMapG);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}

__global__
void getPosMapFromDepthKernel(unsigned short *depthMap, float* cam, const int width, const int height, float* posMap)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;


	if (depthMap[y*width + x])
	{
		float vPosition[3];
		getVertex(depthMap, x, y, width, height, cam, vPosition);

		posMap[y*width * 3 + x * 3 + 0] = vPosition[0];
		posMap[y*width * 3 + x * 3 + 1] = vPosition[1];
		posMap[y*width * 3 + x * 3 + 2] = vPosition[2];
	}
}
__host__
void getPosMapFromDepth(unsigned short *depthMap, float* cam, const int width, const int height, float* posMap)
{
	const int blocks = 32;
	dim3 dimGrid(blocks, blocks);
	dim3 dimBlock(width / blocks, height / blocks);

	getPosMapFromDepthKernel << <dimGrid, dimBlock >> >(depthMap, cam, width, height, posMap);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}

__global__
void getNormalMapFromDepthKernel(unsigned short *depthMap, float* cam, const int width, const int height, float*normalMap)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (checkNeighbours(depthMap, x, y, width, height))
	{
		float thisNor[3];
		getNormal(depthMap, x, y, width, height, cam, thisNor);

		normalMap[y*width * 3 + x * 3 + 0] = thisNor[0];
		normalMap[y*width * 3 + x * 3 + 1] = thisNor[1];
		normalMap[y*width * 3 + x * 3 + 2] = thisNor[2];
	}
}

__host__
void getNormalMapFromDepth(unsigned short *depthMap, float* cam, const int width, const int height, float*normalMap)
{
	const int blocks = 32;
	dim3 dimGrid(blocks, blocks);
	dim3 dimBlock(width / blocks, height / blocks);

	getNormalMapFromDepthKernel << <dimGrid, dimBlock >> >(depthMap, cam, width, height, normalMap);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}

__global__	//init spInfo  �ⲽû���޸�segMask�ĵط�
void getSuperPixelInfoCudaKernel0(int* segMask, unsigned short * depthMap, float* posMap, float* normalMap,
int spNum, float* spInfo, int* spInfoStruct, const int width, const int height)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id<spNum)
	{
		//spInfoStruct[18]=SPI_CONNECT_N	
		// spInfoStruct[19] = SPI_NP_FIRST	
		// spInfoStruct[20] = SPI_NP_MAX
		int connectMaxNum = spInfoStruct[20];  // ��������ھӵ�������
		for (int i = 0; i<connectMaxNum; i++)
		{								// 18
			spInfo[id*spInfoStruct[0] + spInfoStruct[19] + i] = -1;  // ��һ�������ص��ھӵľ���ȫ��ֵΪ-1��
			spInfo[id*spInfoStruct[0] + spInfoStruct[18]] = connectMaxNum;  // connectNum
		}								// 17
	}
}

__global__ //First ��ά����x*x+y*y+z*z + ��ά�����x*x+y*y+z*z <0.01ʱ��������RGB�ָ�ı����Ϊ-1
void getSuperPixelInfoCudaKernelA(int* segMask, unsigned short * depthMap, float* posMap, float* normalMap,
int spNum, float* spInfo, int* spInfoStruct, const int width, const int height)
{
	//// YJ debug
	//for (int ii = 0; ii < width*height; ii++)
	//	if (segMask[ii] == -1)
	//		printf("getSuperPixelInfoCudaKernelA��ͷ��finalSPixel_gpu�ĳ����ر�Ŵ���-1 ��%d\n", segMask[ii]);

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	int id = segMask[y*width + x];  // ���̴߳�������ض�Ӧ��RGB�����طָ�ı��
	float pnTest = 0;
	pnTest += (posMap[y*width * 3 + x * 3 + 0] * posMap[y*width * 3 + x * 3 + 0]);  // ��ά�����x*x
	pnTest += (posMap[y*width * 3 + x * 3 + 1] * posMap[y*width * 3 + x * 3 + 1]);  // y*y
	pnTest += (posMap[y*width * 3 + x * 3 + 2] * posMap[y*width * 3 + x * 3 + 2]);  // z*z

	pnTest += (normalMap[y*width * 3 + x * 3 + 0] * normalMap[y*width * 3 + x * 3 + 0]);  // ��ά���ߵ�x*x
	pnTest += (normalMap[y*width * 3 + x * 3 + 1] * normalMap[y*width * 3 + x * 3 + 1]);  // y*y
	pnTest += (normalMap[y*width * 3 + x * 3 + 2] * normalMap[y*width * 3 + x * 3 + 2]);  // z*z

	if (pnTest<0.01 || id >= spNum || id<0)  // pnTest<0.01����ʲô��˼�� pnTest < 0.01���ºܶ����ص�ĳ����ر�ű��-1
	{
		// if (pnTest < 0.01) printf("****pnTest<0.01****");  // pnTest < 0.01���ºܶ����ص�ĳ����ر�ű��-1
		// if (id >= spNum) printf("****id >= spNum****\n"); // id >= spNum������
		// if (id<0) printf("****id<0****");
		segMask[y*width + x] = -1;  // �����صı����Ϊ-1
	}
}
__global__ //First sum
void getSuperPixelInfoCudaKernelB(int* segMask, unsigned short * depthMap, float* posMap, float* normalMap,
int spNum, float* spInfo, int* spInfoStruct, const int width, const int height)
{
	//spInfoStruct[0]=SPI_SIZE		spInfoStruct[1] = SPI_PNUM
	//spInfoStruct[2]=SPI_POS_SX	spInfoStruct[3] = SPI_POS_SY	spInfoStruct[4] = SPI_POS_SZ
	//spInfoStruct[5]=SPI_NOR_SX	spInfoStruct[6] = SPI_NOR_SY	spInfoStruct[7] = SPI_NOR_SZ
	//spInfoStruct[14]=SPI_DEPTH_SUM	
	//spInfoStruct[18]=SPI_CONNECT_N	spInfoStruct[19] = SPI_NP_FIRST	spInfoStruct[20] = SPI_NP_MAX

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	int id = segMask[y*width + x];  // ���̴߳�������ض�Ӧ��RGB�����طָ�ı��
	if (id >= 0 && id<spNum)  // �������ص�RGB�����ر�Ż���
	{
		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[1], 1);  // spInfo��ͳ�Ƹ�id��ŵ����ظ���

		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[2], posMap[y*width * 3 + x * 3 + 0]);  // spInfo��ͳ�Ƹ�id��ŵ�����������ά�����x����֮��
		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[3], posMap[y*width * 3 + x * 3 + 1]);  // spInfo��ͳ�Ƹ�id��ŵ�����������ά�����y����֮��
		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[4], posMap[y*width * 3 + x * 3 + 2]);  // spInfo��ͳ�Ƹ�id��ŵ�����������ά�����z����֮��

		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[5], normalMap[y*width * 3 + x * 3 + 0]);  // spInfo��ͳ�Ƹ�id��ŵ�����������ά�����x����֮��
		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[6], normalMap[y*width * 3 + x * 3 + 1]);  // spInfo��ͳ�Ƹ�id��ŵ�����������ά�����y����֮��
		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[7], normalMap[y*width * 3 + x * 3 + 2]);  // spInfo��ͳ�Ƹ�id��ŵ�����������ά�����z����֮��
		
		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[14], depthMap[y*width + x]);  // spInfo��ͳ�Ƹ�id��ŵ��������ص����ֵ֮��

		//check neighbor
		const int StepX[4] = { 0, 0, 1, -1 };
		const int StepY[4] = { 1, -1, 0, 0 };
		for (int i = 0; i<4; i++)
		{
			if (x == 0 || x == width - 1 || y == 0 || y == height - 1) 
				continue;

			int dx = x + StepX[i];
			int dy = y + StepY[i];
			int nbID = segMask[dy*width + dx];  // ���������������ھӵ�rgb�����ر��ֵ
			if (nbID >= spNum || nbID<0) 
				continue; 	//-1
			if (id != nbID) // ��ǰ�̴߳�������ص�RGB�����ر�ź��ھӵ�rgb�����ر�Ų�һ��
			{
				int connectMaxNum = spInfoStruct[20]; // 11
				if (connectMaxNum != 11) 
					printf("ERROR connectMaxNum!=list_num(11) in InstanceFusionCuda line 345.Please set list_num = connectMaxNum.");

				int list[11];
				int p = 0;
				for (int j = 0; j<connectMaxNum; j++)
				{
					//***********unstable code************//
					int existInList = 0;
					for (int k = 0; k<p; k++)
					{
						if (list[k] == spInfo[id*spInfoStruct[0] + spInfoStruct[19] + j])
						{
							existInList = 1;
							break;
						}
					}
					if (existInList || spInfo[id*spInfoStruct[0] + spInfoStruct[19] + j] == -1)  // ��ǰ�̴߳�������ض�Ӧ��RGB�����ص��ھ�ֵΪ-1
					{
						spInfo[id*spInfoStruct[0] + spInfoStruct[19] + j] = nbID;  // ��ǰ�̴߳�������ض�Ӧ��RGB�����صĵ�j���ھ���ΪnbID
					}
					else
					{
						list[p++] = spInfo[id*spInfoStruct[0] + spInfoStruct[19] + j]; // ��ǰ�̴߳�������ض�Ӧ��RGB�����صĵ�j���ھӵ�ֵ���浽list����
						if (p >= 11) 
							p--;
					}
					//***********unstable code************//
				}
			}
		}


		//check neighbor
		/*
		const int StepX[4] = {0,0,1,-1};
		const int StepY[4] = {1,-1,0,0};
		bool next = true;
		while(next)
		{
		int v = atomicCAS(spLock+id,-1,y*width+x);	//lock
		if(spLock[id]==y*width+x)
		{
		//handle
		for(int i=0; i<4; i++)
		{
		if(x==0||x==width-1||y==0||y==height-1) continue;

		int dx = x+StepX[i];
		int dy = y+StepY[i];
		int nbID = segMask[dy*width+dx];
		if(nbID>=spNum||nbID<0) continue; 	//-1

		if(id!=nbID)
		{
		int connectNum = spInfo[id*spInfoStruct[0]+spInfoStruct[18]];
		if(connectNum>=spInfoStruct[20])continue;

		int exist = 0;
		for(int j=0;j<connectNum;j++)
		{
		if(spInfo[id*spInfoStruct[0]+spInfoStruct[19]+j]==nbID)
		{
		exist = 1;
		break;
		}
		}
		if(!exist&&connectNum<spInfoStruct[20])
		{
		spInfo[id*spInfoStruct[0]+spInfoStruct[19]+connectNum]=nbID;
		spInfo[id*spInfoStruct[0]+spInfoStruct[18]]++;
		}
		}
		}
		//unlock
		atomicExch(spLock+id,-1);
		next = 0;
		}
		}
		*/

	}
}
__global__ //First avg
void getSuperPixelInfoCudaKernelC(int* segMask, unsigned short * depthMap, float* posMap, float* normalMap,
int spNum, float* spInfo, int* spInfoStruct, const int width, const int height)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id<spNum)
	{
		//spInfoStruct[0] =  SPI_SIZE;		spInfoStruct[1] =  SPI_PNUM;
		//spInfoStruct[2] =  SPI_POS_SX;		spInfoStruct[3] =  SPI_POS_SY;	spInfoStruct[4] =  SPI_POS_SZ;
		//spInfoStruct[5] =  SPI_NOR_SX;		spInfoStruct[6] =  SPI_NOR_SY;	spInfoStruct[7] =  SPI_NOR_SZ;
		//spInfoStruct[8] =  SPI_POS_AX;		spInfoStruct[9] =  SPI_POS_AY;	spInfoStruct[10] =  SPI_POS_AZ;
		//spInfoStruct[11] =  SPI_NOR_AX;		spInfoStruct[12] =  SPI_NOR_AY;	spInfoStruct[13] =  SPI_NOR_AZ;
		//spInfoStruct[14] =  SPI_DEPTH_SUM;	spInfoStruct[15] =  SPI_DEPTH_AVG;

		int t = spInfo[id*spInfoStruct[0] + spInfoStruct[1]];  // ����ĵ�ǰ�����ص����ظ���
		if (t != 0)
		{
			spInfo[id*spInfoStruct[0] + spInfoStruct[8]] = spInfo[id*spInfoStruct[0] + spInfoStruct[2]] / t;  // ����ó���������ά����x��ƽ��ֵ
			spInfo[id*spInfoStruct[0] + spInfoStruct[9]] = spInfo[id*spInfoStruct[0] + spInfoStruct[3]] / t;  // ����ó���������ά����y��ƽ��ֵ
			spInfo[id*spInfoStruct[0] + spInfoStruct[10]] = spInfo[id*spInfoStruct[0] + spInfoStruct[4]] / t;  // ����ó���������ά����z��ƽ��ֵ

			float nx = spInfo[id*spInfoStruct[0] + spInfoStruct[5]]; // �ó���������ά����x�ĺ�
			float ny = spInfo[id*spInfoStruct[0] + spInfoStruct[6]]; // �ó���������ά����y�ĺ�
			float nz = spInfo[id*spInfoStruct[0] + spInfoStruct[7]]; // �ó���������ά����z�ĺ�

			float len = sqrt(nx*nx + ny*ny + nz*nz);  // �ó���������ά����ĳ���
			spInfo[id*spInfoStruct[0] + spInfoStruct[11]] = spInfo[id*spInfoStruct[0] + spInfoStruct[5]] / len;  //��һ������
			spInfo[id*spInfoStruct[0] + spInfoStruct[12]] = spInfo[id*spInfoStruct[0] + spInfoStruct[6]] / len;
			spInfo[id*spInfoStruct[0] + spInfoStruct[13]] = spInfo[id*spInfoStruct[0] + spInfoStruct[7]] / len;

			spInfo[id*spInfoStruct[0] + spInfoStruct[15]] = spInfo[id*spInfoStruct[0] + spInfoStruct[14]] / t;  // ���㳬�����ڵ�ƽ�����ֵ
		}
	}
}
__global__ //Second 
void getSuperPixelInfoCudaKernelD(int* segMask, unsigned short * depthMap, float* posMap, float* normalMap,
int spNum, float* spInfo, int* spInfoStruct, const int width, const int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	//depth_stand_deviation
	int id = segMask[y*width + x];  // ��ǰ�������ص�rgb�����ر��
	if (id >= spNum || id<0) return;	//-1

	//spInfoStruct[0] =  SPI_SIZE;			spInfoStruct[1] =  SPI_PNUM;
	//spInfoStruct[2] =  SPI_POS_SX;		spInfoStruct[3] =  SPI_POS_SY;	spInfoStruct[4] =  SPI_POS_SZ;
	//spInfoStruct[5] =  SPI_NOR_SX;		spInfoStruct[6] =  SPI_NOR_SY;	spInfoStruct[7] =  SPI_NOR_SZ;
	//spInfoStruct[8] =  SPI_POS_AX;		spInfoStruct[9] =  SPI_POS_AY;	spInfoStruct[10] =  SPI_POS_AZ;
	//spInfoStruct[11] =  SPI_NOR_AX;		spInfoStruct[12] =  SPI_NOR_AY;	spInfoStruct[13] =  SPI_NOR_AZ;
	//spInfoStruct[14] =  SPI_DEPTH_SUM;	spInfoStruct[15] =  SPI_DEPTH_AVG;
	//spInfoStruct[16] =  SPI_DIST_DEV;		spInfoStruct[17] =  SPI_NOR_DEV;	
	//spInfoStruct[18] =  SPI_CONNECT_N;	spInfoStruct[19] =  SPI_NP_FIRST;	spInfoStruct[20] =  SPI_NP_MAX;
	//spInfoStruct[21] =  SPI_FINAL;
	int connectNum = spInfoStruct[20];  // 11
	float minDist = 999999.9f;
	float minNor = 999999.9f;
	int minID = id;
	for (int i = 0; i <= connectNum; i++)
	{
		int idTest;
		if (i == connectNum) idTest = id;	//self
		else idTest = spInfo[id*spInfoStruct[0] + spInfoStruct[19] + i];

		float vecA[3];	//SPI_NOR_A  �����ص�ƽ������
		vecA[0] = spInfo[idTest*spInfoStruct[0] + spInfoStruct[11]];
		vecA[1] = spInfo[idTest*spInfoStruct[0] + spInfoStruct[12]];
		vecA[2] = spInfo[idTest*spInfoStruct[0] + spInfoStruct[13]];

		float vecB[3];	//SPI_POS_A - P ��ǰ���ص���ά����ı�׼��
		vecB[0] = spInfo[idTest*spInfoStruct[0] + spInfoStruct[8]] - posMap[y*width * 3 + x * 3 + 0];
		vecB[1] = spInfo[idTest*spInfoStruct[0] + spInfoStruct[9]] - posMap[y*width * 3 + x * 3 + 1];
		vecB[2] = spInfo[idTest*spInfoStruct[0] + spInfoStruct[10]] - posMap[y*width * 3 + x * 3 + 2];

		float lenA = sqrt(vecA[0] * vecA[0] + vecA[1] * vecA[1] + vecA[2] * vecA[2]);  // �����ص�ƽ������ĳ���
		float lenB = sqrt(vecB[0] * vecB[0] + vecB[1] * vecB[1] + vecB[2] * vecB[2]);  // ��ǰ���ص���ά����ı�׼�������ĳ���

		float dotAB = vecA[0] * vecB[0] + vecA[1] * vecB[1] + vecA[2] * vecB[2];   // �����ص�ƽ������ �� ��ǰ���ص���ά����ı�׼������
		float dist = abs(dotAB / lenA) + 1.0*lenB;

		float diffNor1 = abs(vecA[0] - normalMap[y*width * 3 + x * 3 + 0]);  // |������ƽ�������x - ��ǰ���صķ����x|
		float diffNor2 = abs(vecA[1] - normalMap[y*width * 3 + x * 3 + 1]);
		float diffNor3 = abs(vecA[2] - normalMap[y*width * 3 + x * 3 + 2]);
		float diffNor = diffNor1*diffNor1 + diffNor2*diffNor2 + diffNor3*diffNor3;  // 
		if (dist<minDist)
		{
			minNor = diffNor;
			minDist = dist;
			minID = idTest;
		}
	}

	float threshold = (0.026*spInfo[minID*spInfoStruct[0] + spInfoStruct[15]] - 4.0f) / 1186.0f;
	if (minDist>2 * threshold)	
	minID = -1;

	if (minID != -1)
	{
		atomicAdd(spInfo + minID*spInfoStruct[0] + spInfoStruct[16], (minDist * minDist));
		atomicAdd(spInfo + minID*spInfoStruct[0] + spInfoStruct[17], minNor);
	}

	if (id != minID)	//re-clustering
	{
		segMask[y*width + x] = minID;

		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[1], -1);
		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[2], -posMap[y*width * 3 + x * 3 + 0]);
		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[3], -posMap[y*width * 3 + x * 3 + 1]);
		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[4], -posMap[y*width * 3 + x * 3 + 2]);
		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[5], -normalMap[y*width * 3 + x * 3 + 0]);
		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[6], -normalMap[y*width * 3 + x * 3 + 1]);
		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[7], -normalMap[y*width * 3 + x * 3 + 2]);
		atomicAdd(spInfo + id*spInfoStruct[0] + spInfoStruct[14], -depthMap[y*width + x]);

		if (minID != -1)
		{
			atomicAdd(spInfo + minID*spInfoStruct[0] + spInfoStruct[1], 1);
			atomicAdd(spInfo + minID*spInfoStruct[0] + spInfoStruct[2], posMap[y*width * 3 + x * 3 + 0]);
			atomicAdd(spInfo + minID*spInfoStruct[0] + spInfoStruct[3], posMap[y*width * 3 + x * 3 + 1]);
			atomicAdd(spInfo + minID*spInfoStruct[0] + spInfoStruct[4], posMap[y*width * 3 + x * 3 + 2]);
			atomicAdd(spInfo + minID*spInfoStruct[0] + spInfoStruct[5], normalMap[y*width * 3 + x * 3 + 0]);
			atomicAdd(spInfo + minID*spInfoStruct[0] + spInfoStruct[6], normalMap[y*width * 3 + x * 3 + 1]);
			atomicAdd(spInfo + minID*spInfoStruct[0] + spInfoStruct[7], normalMap[y*width * 3 + x * 3 + 2]);
			atomicAdd(spInfo + minID*spInfoStruct[0] + spInfoStruct[14], depthMap[y*width + x]);
		}
	}

}
__global__ //Second avg
void getSuperPixelInfoCudaKernelE(int* segMask, unsigned short * depthMap, float* posMap, float* normalMap,
int spNum, float* spInfo, int* spInfoStruct, const int width, const int height)
{
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id<spNum)
	{
		//spInfoStruct[0] =  SPI_SIZE;		spInfoStruct[1] =  SPI_PNUM;
		//spInfoStruct[2] =  SPI_POS_SX;		spInfoStruct[3] =  SPI_POS_SY;	spInfoStruct[4] =  SPI_POS_SZ;
		//spInfoStruct[5] =  SPI_NOR_SX;		spInfoStruct[6] =  SPI_NOR_SY;	spInfoStruct[7] =  SPI_NOR_SZ;
		//spInfoStruct[8] =  SPI_POS_AX;		spInfoStruct[9] =  SPI_POS_AY;	spInfoStruct[10] =  SPI_POS_AZ;
		//spInfoStruct[11] =  SPI_NOR_AX;		spInfoStruct[12] =  SPI_NOR_AY;	spInfoStruct[13] =  SPI_NOR_AZ;
		//spInfoStruct[14] =  SPI_DEPTH_SUM;	spInfoStruct[15] =  SPI_DEPTH_AVG;
		//spInfoStruct[16] =  SPI_DIST_DEV;		spInfoStruct[17] =  SPI_NOR_DEV;	
		int t = spInfo[id*spInfoStruct[0] + spInfoStruct[1]];
		if (t != 0)
		{
			spInfo[id*spInfoStruct[0] + spInfoStruct[16]] = sqrt(spInfo[id*spInfoStruct[0] + spInfoStruct[16]] / t);
			spInfo[id*spInfoStruct[0] + spInfoStruct[17]] = sqrt(spInfo[id*spInfoStruct[0] + spInfoStruct[17]] / t);

			spInfo[id*spInfoStruct[0] + spInfoStruct[8]] = spInfo[id*spInfoStruct[0] + spInfoStruct[2]] / t;
			spInfo[id*spInfoStruct[0] + spInfoStruct[9]] = spInfo[id*spInfoStruct[0] + spInfoStruct[3]] / t;
			spInfo[id*spInfoStruct[0] + spInfoStruct[10]] = spInfo[id*spInfoStruct[0] + spInfoStruct[4]] / t;

			float nx = spInfo[id*spInfoStruct[0] + spInfoStruct[5]];
			float ny = spInfo[id*spInfoStruct[0] + spInfoStruct[6]];
			float nz = spInfo[id*spInfoStruct[0] + spInfoStruct[7]];
			float len = sqrt(nx*nx + ny*ny + nz*nz);  // �����ģ
			spInfo[id*spInfoStruct[0] + spInfoStruct[11]] = spInfo[id*spInfoStruct[0] + spInfoStruct[5]] / len;  // �о�������ƽ��ֵ�ɣ������ǰѷ��ߵ�λ����
			spInfo[id*spInfoStruct[0] + spInfoStruct[12]] = spInfo[id*spInfoStruct[0] + spInfoStruct[6]] / len;
			spInfo[id*spInfoStruct[0] + spInfoStruct[13]] = spInfo[id*spInfoStruct[0] + spInfoStruct[7]] / len;

			spInfo[id*spInfoStruct[0] + spInfoStruct[15]] = spInfo[id*spInfoStruct[0] + spInfoStruct[14]] / t;  // ���ֵƽ��
		}
	}
}

__host__
void getSuperPixelInfoCuda(int* segMask, unsigned short * depthMap, float* posMap, float* normalMap,
int spNum, float* spInfo, int* spInfoStruct, const int width, const int height)
{
	//width*height �����ش����趨cuda������
	const int blocks1 = 32;
	dim3 dimGrid1(blocks1, blocks1);
	dim3 dimBlock1(width / blocks1, height / blocks1);

	//spNum �Գ����ش����趨cuda������
	const int threads2 = 512;
	const int blocks2 = (spNum + threads2 - 1) / threads2;
	dim3 dimGrid2(blocks2);
	dim3 dimBlock2(threads2);

	// ��ʼ��spInfo
	getSuperPixelInfoCudaKernel0 << <dimGrid2, dimBlock2 >> >(segMask, depthMap, posMap, normalMap, spNum, spInfo, spInfoStruct, width, height);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());

	//ȥ�����ܻ����0��ĸ��ֵ
	getSuperPixelInfoCudaKernelA << <dimGrid1, dimBlock1 >> >(segMask, depthMap, posMap, normalMap, spNum, spInfo, spInfoStruct, width, height);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());

	getSuperPixelInfoCudaKernelB << <dimGrid1, dimBlock1 >> >(segMask, depthMap, posMap, normalMap, spNum, spInfo, spInfoStruct, width, height);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());

	//��ά����͡�ƽ�����ߡ�ƽ�����
	getSuperPixelInfoCudaKernelC << <dimGrid2, dimBlock2 >> >(segMask, depthMap, posMap, normalMap, spNum, spInfo, spInfoStruct, width, height);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());

	getSuperPixelInfoCudaKernelD << <dimGrid1, dimBlock1 >> >(segMask, depthMap, posMap, normalMap, spNum, spInfo, spInfoStruct, width, height);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());

	getSuperPixelInfoCudaKernelE << <dimGrid2, dimBlock2 >> >(segMask, depthMap, posMap, normalMap, spNum, spInfo, spInfoStruct, width, height);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}

__global__
void getFinalSuperPiexlKernel(float* spInfo, const int width, const int height, int* spInfoStruct, int* segMask)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	int id_ori = segMask[y*width + x];
	if (id_ori<0)
	{
		segMask[y*width + x] = id_ori;
	}
	else
	{
		//spInfoStruct[0]=SPI_SIZE 	spInfoStruct[21]=SPI_FINAL
		int id_final = spInfo[id_ori*spInfoStruct[0] + spInfoStruct[21]];
		segMask[y*width + x] = id_final;
	}
}
__host__
void getFinalSuperPiexl(float* spInfo, const int width, const int height, int* spInfoStruct, int* segMask)
{
	const int blocks = 32;
	dim3 dimGrid(blocks, blocks);
	dim3 dimBlock(width / blocks, height / blocks);

	getFinalSuperPiexlKernel << <dimGrid, dimBlock >> >(spInfo, width, height, spInfoStruct, segMask);
	gpuErrChk(cudaGetLastError());
	gpuErrChk(cudaDeviceSynchronize());
}