#include "gSLICrTools.h"

void gSLICrTools::rgbdSuperPixelSeg(cv::Mat rgb,cv::Mat depth,int* gSLICrMask,cv::Mat& visual_img)
{
	int* finalSPixel = new int[width*height];	
	gSLICrInterface(rgb,finalSPixel,visual_img);
	cv::Size size(width, height);
	cv::Mat depthSeg;
	memcpy(gSLICrMask,finalSPixel,sizeof(int)*width*height);
    delete finalSPixel;

	// cv::resize(depth, depthSeg, size);
	//  1200, segMask, finalSPixel, cam
	// mergeSuperPixel(matToUshort_oneChanel(depthSeg),finalSPixel,gSLICrMask);

}

void gSLICrTools::imageCV2SLIC(const cv::Mat& inimg, gSLICr::UChar4Image* outimg)
{
	gSLICr::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < outimg->noDims.y;y++)
	{
		for (int x = 0; x < outimg->noDims.x; x++)
		{
			int idx = x + y * outimg->noDims.x;
			outimg_ptr[idx].b = inimg.at<cv::Vec3b>(y, x)[0];
			outimg_ptr[idx].g = inimg.at<cv::Vec3b>(y, x)[1];
			outimg_ptr[idx].r = inimg.at<cv::Vec3b>(y, x)[2];
		}
	}
}

void gSLICrTools::imageSLIC2CV(const gSLICr::UChar4Image* inimg, cv::Mat& outimg)
{
	const gSLICr::Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < inimg->noDims.y; y++)
	{
		for (int x = 0; x < inimg->noDims.x; x++)
		{
			int idx = x + y * inimg->noDims.x;
			outimg.at<cv::Vec3b>(y, x)[0] = inimg_ptr[idx].b;
			outimg.at<cv::Vec3b>(y, x)[1] = inimg_ptr[idx].g;
			outimg.at<cv::Vec3b>(y, x)[2] = inimg_ptr[idx].r;
		}
	}
}

void gSLICrTools::gSLICrInterface(cv::Mat inputrgb, int* gSLICrMask,cv::Mat& boundry_draw_frame)
{
	// gSLICr::UChar4Image* in_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);
	
    gSLICr::UChar4Image* in_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);
	cv::Size size(width, height);
	cv::Mat frame;
	cv::cvtColor(inputrgb,frame,cv::COLOR_BGR2RGB);
	cv::resize(frame, frame, size);
	imageCV2SLIC(frame, in_img);
	gSLICr_engine->Process_Frame(in_img);


	gSLICr::UChar4Image* out_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);
	gSLICr_engine->Draw_Segmentation_Result(out_img);	
	// cv::Mat boundry_draw_frame;
	cv::Size s(my_settings.img_size.x, my_settings.img_size.y);
	boundry_draw_frame.create(s, CV_8UC3);
	// load_image(out_img, boundry_draw_frame);
	// cv::imwrite("segmentation.png", boundry_draw_frame);


	//segMask = gSLICr_engine->Get_Seg_Res()->GetData(MEMORYDEVICE_CPU);
	memcpy(gSLICrMask, gSLICr_engine->Get_Seg_Res()->GetData(MEMORYDEVICE_CPU), width*height*sizeof(int));
	
    // delete in_img;
    // delete out_img;

}



void gSLICrTools::load_image(const gSLICr::UChar4Image* inimg, cv::Mat& outimg)
{
	const gSLICr::Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < inimg->noDims.y; y++)
		for (int x = 0; x < inimg->noDims.x; x++)
		{
			int idx = x + y * inimg->noDims.x;
			outimg.at<cv::Vec3b>(y, x)[0] = inimg_ptr[idx].b;
			outimg.at<cv::Vec3b>(y, x)[1] = inimg_ptr[idx].g;
			outimg.at<cv::Vec3b>(y, x)[2] = inimg_ptr[idx].r;
		}
}






void gSLICrTools::mergeSuperPixel(unsigned short* depthMap, int* segMask, int *finalSPixel)
{
	////Get Cam
	// Eigen::Vector4f cam = Eigen::Vector4f(Intrinsics::getInstance().cx(), Intrinsics::getInstance().cy(),
	// 	1.0 / Intrinsics::getInstance().fx(), 1.0 / Intrinsics::getInstance().fy());
	float* cam_gpu;						// 相机参数 input
	cudaMalloc((void **)&cam_gpu, 4 * sizeof(float));
	cudaMemcpy(cam_gpu, &cam(0), 4 * sizeof(float), cudaMemcpyHostToDevice);

	// 此时已经在gSLICr中获得了rgb的超像素分割(InstanceFusion:step -1_1)
	//Step 0	Gaussian filter  对原始深度图做高斯卷积平滑
	unsigned short * oriDepthMap_gpu;		// 原始深度图 input
	cudaMalloc((void **)&oriDepthMap_gpu, height*width * sizeof(unsigned short));
	cudaMemcpy(oriDepthMap_gpu, depthMap, height*width * sizeof(unsigned short), cudaMemcpyHostToDevice);

	unsigned short * depthMapG_gpu;			// 高斯过滤深度图 output
	cudaMalloc((void **)&depthMapG_gpu, height*width*sizeof(unsigned short));
	cudaMemset(depthMapG_gpu, 0, height*width*sizeof(unsigned short));
	depthMapGaussianfilter(oriDepthMap_gpu, width, height, depthMapG_gpu);

	//Step 1	getPosMap   用深度图计算相机坐标下的三维坐标图
	float* posMap_gpu;					// 三维坐标图（含每个像素） output
	cudaMalloc((void **)&posMap_gpu, height*width * 3 * sizeof(float));
	cudaMemset(posMap_gpu, 0, height*width * 3 * sizeof(float));
	getPosMapFromDepth(depthMapG_gpu, cam_gpu, width, height, posMap_gpu);


	//Step 2	getNormalMap  用深度图计算相机坐标下的三维法线图
	float* normalMap_gpu;				// 三维法线图（含每个像素） output
	cudaMalloc((void **)&normalMap_gpu, height*width * 3 * sizeof(float));
	cudaMemset(normalMap_gpu, 0, height*width * 3 * sizeof(float));

	getNormalMapFromDepth(depthMapG_gpu, cam_gpu, width, height, normalMap_gpu);


	//Step 3  统计每个超像素的信息，包括位置平均值、标准差，法线平均值、标准差，超像素的邻接超像素等等。结果保存在spInfo中。同时还根据平面估计对超像素重聚类，是一个写的有点乱的cuda函数。。（也写了cpu版本）
	float* spInfo_gpu;						//output
	cudaMalloc((void **)&spInfo_gpu, spNum*SPI_SIZE*sizeof(float));
	cudaMemset(spInfo_gpu, 0, spNum*SPI_SIZE*sizeof(float));

	int spInfoStruct[22];
	spInfoStruct[0] = SPI_SIZE;		spInfoStruct[1] = SPI_PNUM;  // 1个超像素标号的像素个数
	spInfoStruct[2] = SPI_POS_SX;		spInfoStruct[3] = SPI_POS_SY;	spInfoStruct[4] = SPI_POS_SZ;
	spInfoStruct[5] = SPI_NOR_SX;		spInfoStruct[6] = SPI_NOR_SY;	spInfoStruct[7] = SPI_NOR_SZ;
	spInfoStruct[8] = SPI_POS_AX;		spInfoStruct[9] = SPI_POS_AY;	spInfoStruct[10] = SPI_POS_AZ;
	spInfoStruct[11] = SPI_NOR_AX;		spInfoStruct[12] = SPI_NOR_AY;	spInfoStruct[13] = SPI_NOR_AZ;
	spInfoStruct[14] = SPI_DEPTH_SUM;	spInfoStruct[15] = SPI_DEPTH_AVG;
	spInfoStruct[16] = SPI_DIST_DEV;	spInfoStruct[17] = SPI_NOR_DEV;
	spInfoStruct[18] = SPI_CONNECT_N;	spInfoStruct[19] = SPI_NP_FIRST;	spInfoStruct[20] = SPI_NP_MAX;
	spInfoStruct[21] = SPI_FINAL;
	int* spInfoStruct_gpu;				//input
	cudaMalloc((void **)&spInfoStruct_gpu, 22 * sizeof(int));
	cudaMemcpy(spInfoStruct_gpu, spInfoStruct, 22 * sizeof(int), cudaMemcpyHostToDevice);

	int* finalSPixel_gpu;				//input   (next input & output )
	cudaMalloc((void **)&finalSPixel_gpu, height*width * sizeof(int));
	cudaMemcpy(finalSPixel_gpu, segMask, height*width * sizeof(int), cudaMemcpyHostToDevice);

	//*********result is not stable but code can run and fast*********//
	// Reclustering  这一步后finalSPixel_gpu就会出现很多-1
	getSuperPixelInfoCuda(finalSPixel_gpu, depthMapG_gpu, posMap_gpu, normalMap_gpu, spNum, spInfo_gpu, spInfoStruct_gpu, width, height);

	//// YJ Debug测试超像素标号是否存在-1
	//cudaMemcpy(segMask, finalSPixel_gpu, height*width * sizeof(int), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < width*height; i++)
	//	if (segMask[i] == -1)
	//		printf("%d\n", segMask[i]);  // 打印出来很多-1


	//Step 4  根据阈值，能量函数，将相似度较高的相邻超像素合并为同一个超像素。
	float* spInfo = (float*)malloc(spNum*SPI_SIZE*sizeof(float));	//next input
	cudaMemcpy(spInfo, spInfo_gpu, spNum*SPI_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
	//Debug
	/*for(int i=0;i<spNum;i++)
	{
	if(spInfo[i*SPI_SIZE+0])
	{
	std::cout<<"id: "<<i<<" num:"<<spInfo[i*SPI_SIZE + SPI_PNUM]<<" connectNum: "<<spInfo[i*SPI_SIZE + SPI_CONNECT_N];
	int n = spInfo[i*SPI_SIZE + SPI_CONNECT_N];
	std::cout<<"\n  nid: ";
	for(int j=0;j<n;j++)
	{
	std::cout<<spInfo[i*SPI_SIZE+SPI_NP_FIRST+j]<<" ";
	}
	std::cout<<std::endl;
	}
	}*/

	// mergeSuperPixel
	connectSuperPixel(spInfo);
	cudaMemcpy(spInfo_gpu, spInfo, spNum*SPI_SIZE*sizeof(float), cudaMemcpyHostToDevice);

	//Debug FINAL_SP
	cudaMemcpy(segMask, finalSPixel_gpu, height*width * sizeof(int), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < width*height; i++)
	//	printf("%d\n", segMask[i]);  // 打印出来很多-1

	//Step 5  按照Step 4的结果填充图像，生成最终的超像素图。
	// getFinalSuperPiexl
	getFinalSuperPiexl(spInfo_gpu, width, height, spInfoStruct_gpu, finalSPixel_gpu);
	cudaMemcpy(finalSPixel, finalSPixel_gpu, height*width * sizeof(int), cudaMemcpyDeviceToHost);



	//Debug DEP+DEP_G  输出原始深度图和经高斯过滤的深度图
	/*
	cv::Mat depthImage1(height, width, CV_16UC1, depthMap);
	std::string savename1 = "./temp/depth1.png";
	imwrite(savename1.c_str(),depthImage1);
	cv::Mat depthImage2(height, width, CV_16UC1, depthMapG);
	std::string savename2 = "./temp/depth2.png";
	imwrite(savename2.c_str(),depthImage2);
	std::cout<<"Debug DEP+DEP_G"<<std::endl;
	*/

	//Debug  POS+NOR  可视化输出由深度图算出的三维pos图和normal图
	/*
	float* posMap = (float*)malloc(height*width*3*sizeof(float));
	cudaMemcpy(posMap, posMap_gpu, height*width*3*sizeof(float), cudaMemcpyDeviceToHost);
	float* normalMap = (float*)malloc(height*width*3*sizeof(float));
	cudaMemcpy(normalMap, normalMap_gpu, height*width*3*sizeof(float), cudaMemcpyDeviceToHost);
	unsigned char* posMap2 = (unsigned char*)malloc(height*width*3*sizeof(unsigned char));
	unsigned char* normalMap2 = (unsigned char*)malloc(height*width*3*sizeof(unsigned char));
	for(int x=0;x<width;x++)
	{
	for(int y=0;y<height;y++)
	{
	for(int c =0; c<3;c++)
	{
	posMap2[y*width*3+x*3+c] = posMap[y*width*3+x*3+c] * 255 + 125;
	//normalMap2[y*width*3+x*3+c] = normalMap[y*width*3+x*3+c] * 255 + 0;
	normalMap2[y*width*3+x*3+c] = spInfo[segMask[y*width+x]*SPI_SIZE+SPI_NOR_AX+c] * 255 + 0;
	}
	}
	}
	cv::Mat posImage(height, width, CV_8UC3, posMap2);
	imwrite("./temp/posImage.png",posImage);
	cv::Mat normalImage(height, width, CV_8UC3, normalMap2);
	imwrite("./temp/normalImage.png",normalImage);
	std::cout<<"Debug POS+NOR"<<std::endl;
	if(posMap)		free(posMap);
	if(posMap2)		free(posMap2);
	if(normalMap)	free(normalMap);
	if(normalMap2)	free(normalMap2);
	*/

	//Debug FINAL_SP  可视化输出第一次超像素分割:segMask，可视化输出第二次超像素分割:SPixel，其中超像素与超像素之间用白线隔开
	// Ps: 第一次超像素分割和第二次超像素分割合起来是[二阶段分割]的第一阶段
	/*
	if (false)
	{
		TimeTick();
		const int StepX[4] = { 0, 0, 1, -1 };
		const int StepY[4] = { 1, -1, 0, 0 };
		unsigned short segMask_debug[height*width];
		unsigned short finalSPixel_debug[height*width];
		for (int x = 0; x<width; x++)
		{
			for (int y = 0; y<height; y++)
			{
				segMask_debug[y*width + x] = segMask[y*width + x] * 50;
				finalSPixel_debug[y*width + x] = segMask[y*width + x] * 50;
				for (int i = 0; i<4; i++)
				{
					if (x == 0 || x == width - 1 || y == 0 || y == height - 1) continue;

					int dx = x + StepX[i];
					int dy = y + StepY[i];
					int id = finalSPixel[y*width + x];
					int nbID = finalSPixel[dy*width + dx];
					//if(nbID>=spNum||nbID<0) continue;

					if (id != nbID) finalSPixel_debug[y*width + x] = 60000;


					id = segMask[y*width + x];
					nbID = segMask[dy*width + dx];

					if (id != nbID) segMask_debug[y*width + x] = 60000;

				}
			}
		}
		cv::Mat segMaskImage(height, width, CV_16UC1, segMask_debug);
		std::string sm_save_dir("./temp/");
		std::string sm_suffix("_segMask.png");
		sm_save_dir += std::to_string(0);	//frameID
		sm_save_dir += sm_suffix;
		cv::imwrite(sm_save_dir, segMaskImage);


		cv::Mat finalSPixelImage(height, width, CV_16UC1, finalSPixel_debug);
		std::string sp_save_dir("./temp/");
		std::string sp_suffix("_SPixel.png");
		sp_save_dir += std::to_string(0);	//frameID
		sp_save_dir += sp_suffix;
		cv::imwrite(sp_save_dir, finalSPixelImage);
		TimeTock(" Debug FINAL_SP");
	}
	*/

	if (posMap_gpu)			
		cudaFree(posMap_gpu);
	if (normalMap_gpu)		
		cudaFree(normalMap_gpu);
	if (spInfo_gpu)			
		cudaFree(spInfo_gpu);
	if (spInfoStruct_gpu)	
		cudaFree(spInfoStruct_gpu);

	if (cam_gpu)				
		cudaFree(cam_gpu);
	if (oriDepthMap_gpu) 	
		cudaFree(oriDepthMap_gpu);
	if (depthMapG_gpu) 		
		cudaFree(depthMapG_gpu);
	if (finalSPixel_gpu)		
		cudaFree(finalSPixel_gpu);

	if (spInfo)				
		free(spInfo);
}

void gSLICrTools::connectSuperPixel(float* spInfo)
{

	//spInfo
	//0 pixel_num   123 pos_sum   456 nor_sum 789 pos_avg   10-12 nor_avg   13 depth_avg 14 depth_avg   
	//15 distance_stand_deviation 16 num_after_cluster 17 connectNum   18-28 neighbor 29 finalID

	//connect test (dist term + normal term)
	for (int i = 0; i<spNum; i++)
	{
		spInfo[i*SPI_SIZE + SPI_FINAL] = -1;
		int connectNum = spInfo[i*SPI_SIZE + SPI_CONNECT_N];

		for (int j = 0; j<connectNum; j++)
		{
			int idA = i;
			int idB = spInfo[i*SPI_SIZE + SPI_NP_FIRST + j];
			if (idB == -1) 
				continue;

			//not negative
			//if(spInfo[idA*SPI_SIZE+SPI_DEPTH_AVG]<0) std::cout<<"ERROR7: "<<idA<<std::endl;
			//if(spInfo[idA*SPI_SIZE+SPI_DIST_DEV]<0) std::cout<<"ERROR8: "<<idA<<std::endl;

			int flag = 1;
			//D
			float vecA[3];
			vecA[0] = spInfo[idA*SPI_SIZE + SPI_NOR_AX];
			vecA[1] = spInfo[idA*SPI_SIZE + SPI_NOR_AY];
			vecA[2] = spInfo[idA*SPI_SIZE + SPI_NOR_AZ];
			float vecB[3];
			vecB[0] = spInfo[idA*SPI_SIZE + SPI_POS_AX] - spInfo[idB*SPI_SIZE + SPI_POS_AX];
			vecB[1] = spInfo[idA*SPI_SIZE + SPI_POS_AY] - spInfo[idB*SPI_SIZE + SPI_POS_AY];
			vecB[2] = spInfo[idA*SPI_SIZE + SPI_POS_AZ] - spInfo[idB*SPI_SIZE + SPI_POS_AZ];
			float lenA = std::sqrt(vecA[0] * vecA[0] + vecA[1] * vecA[1] + vecA[2] * vecA[2]);
			float lenB = std::sqrt(vecB[0] * vecB[0] + vecB[1] * vecB[1] + vecB[2] * vecB[2]);
			float dotAB = vecA[0] * vecB[0] + vecA[1] * vecB[1] + vecA[2] * vecB[2];
			float distTerm = std::abs(dotAB / lenA) + 1.0*lenB;
			//D_A
			//if(distTerm>3*spInfo[idA*SPI_SIZE+SPI_DIST_DEV] || distTerm>3*spInfo[idB*SPI_SIZE+SPI_DIST_DEV]) flag = 0;
			//D_B
			//if(distTerm>0.02) flag = 0;
			//D E
			float thresholdA1 = 1 * ((0.026*spInfo[idA*SPI_SIZE + SPI_DEPTH_AVG] - 4.0f) / 1186.0f);
			float thresholdB1 = 1 * ((0.026*spInfo[idB*SPI_SIZE + SPI_DEPTH_AVG] - 4.0f) / 1186.0f);
			float thresholdA2 = 2 * spInfo[idA*SPI_SIZE + SPI_DIST_DEV];			//4 (IF NO thresholdA3)
			float thresholdB2 = 2 * spInfo[idB*SPI_SIZE + SPI_DIST_DEV];			//4
			//if(distTerm>1.5*thresholdA1+ || distTerm>1.5*thresholdB1) flag = 0;
			//std::cout<<distTerm<<std::endl;



			float thisNor[3], leftNor[3];
			thisNor[0] = spInfo[idA*SPI_SIZE + SPI_NOR_AX];
			thisNor[1] = spInfo[idA*SPI_SIZE + SPI_NOR_AY];
			thisNor[2] = spInfo[idA*SPI_SIZE + SPI_NOR_AZ];

			leftNor[0] = spInfo[idB*SPI_SIZE + SPI_NOR_AX];
			leftNor[1] = spInfo[idB*SPI_SIZE + SPI_NOR_AY];
			leftNor[2] = spInfo[idB*SPI_SIZE + SPI_NOR_AZ];

			float diffNor1 = std::abs(thisNor[0] - leftNor[0]);
			float diffNor2 = std::abs(thisNor[1] - leftNor[1]);
			float diffNor3 = std::abs(thisNor[2] - leftNor[2]);
			float norTerm = 0.1*std::sqrt(diffNor1*diffNor1 + diffNor2*diffNor2 + diffNor3*diffNor3);
			//std::cout<<norTerm<<std::endl;

			float thresholdA3 = 0 * spInfo[idA*SPI_SIZE + SPI_NOR_DEV];	//0.005
			float thresholdB3 = 0 * spInfo[idB*SPI_SIZE + SPI_NOR_DEV];

			float finTest = distTerm + norTerm;
			float ThresholdA = thresholdA1 + thresholdA2 + thresholdA3;
			float ThresholdB = thresholdB1 + thresholdB2 + thresholdB3;

			if (finTest>ThresholdA || finTest>ThresholdB) 
				flag = 0;
			/*
			float thisVer[3],leftVer[3];
			thisVer[0] = spInfo[idA*SPI_SIZE+SPI_POS_AX];
			thisVer[1] = spInfo[idA*SPI_SIZE+SPI_POS_AY];
			thisVer[2] = spInfo[idA*SPI_SIZE+SPI_POS_AZ];

			leftVer[0] = spInfo[idB*SPI_SIZE+SPI_POS_AX];
			leftVer[1] = spInfo[idB*SPI_SIZE+SPI_POS_AY];
			leftVer[2] = spInfo[idB*SPI_SIZE+SPI_POS_AZ];

			float vecToL[3];
			vecToL[0] = leftVer[0] - thisVer[0];
			vecToL[1] = leftVer[1] - thisVer[1];
			vecToL[2] = leftVer[2] - thisVer[2];

			float dotPro1 = vecToL[0]*thisNor[0] + vecToL[1]*thisNor[1] + vecToL[2]*thisNor[2];
			float dotPro2 = leftNor[0]*thisNor[0] + leftNor[1]*thisNor[1] + leftNor[2]*thisNor[2];

			float cross2[3];
			cross2[0] = leftNor[1]*thisNor[2] - leftNor[2]*thisNor[1];
			cross2[1] = leftNor[2]*thisNor[0] - leftNor[0]*thisNor[3];
			cross2[2] = leftNor[0]*thisNor[1] - leftNor[1]*thisNor[0];
			float lenCross2 = std::sqrt( cross2[0]*cross2[0] + cross2[1]*cross2[1] + cross2[2]*cross2[2]);
			float len2L = std::sqrt( leftNor[0]*leftNor[0] + leftNor[1]*leftNor[1] + leftNor[2]*leftNor[2]);
			float len2T = std::sqrt( thisNor[0]*thisNor[0] + thisNor[1]*thisNor[1] + thisNor[2]*thisNor[2]);
			float cos2 = dotPro2 / (len2L*len2T);
			float sin2 = lenCross2 / (len2L*len2T);
			//if(dotPro1 > 0 && cos2 > 1) flag = 0;	//need test
			//finTest += dotPro1;
			//if(finTest>ThresholdA || finTest>ThresholdB) flag = 0;
			*/

			if (!flag)
			{

				//idA close
				spInfo[idA*SPI_SIZE + SPI_NP_FIRST + j] = -1;

				//idB close
				int connectNumB = spInfo[idB*SPI_SIZE + SPI_CONNECT_N];
				for (int k = 0; k<connectNumB; k++)
				{
					if (spInfo[idB*SPI_SIZE + SPI_NP_FIRST + k] == idA)
					{
						spInfo[idB*SPI_SIZE + SPI_NP_FIRST + k] = -1;
						break;
					}
				}
			}
		}

	}

	//finalID consistency
	int* stack = new int[spNum * 10];
	int p = 0;
	for (int i = 0; i<spNum; i++)
	{
		int finalID;
		if (spInfo[i*SPI_SIZE + SPI_FINAL] == -1) finalID = i;
		else finalID = spInfo[i*SPI_SIZE + SPI_FINAL];

		stack[p++] = i;
		while (p>0)
		{
			int target = stack[--p];
			if (spInfo[target*SPI_SIZE + SPI_FINAL] != -1) continue;

			spInfo[target*SPI_SIZE + SPI_FINAL] = finalID;	//Set

			int  connectNum = spInfo[target*SPI_SIZE + SPI_CONNECT_N];
			for (int j = 0; j<connectNum; j++)
			{
				int connectID = spInfo[target*SPI_SIZE + SPI_NP_FIRST + j];

				if (connectID == -1) continue;
				if (spInfo[connectID*SPI_SIZE + SPI_FINAL] != -1) continue;

				stack[p++] = connectID;
			}
		}
	}
	delete[] stack;
}

// opencv的Mat格式转为uchar格式（3通道的RGB）
uchar* gSLICrTools::matToUchar(cv::Mat img)
{
	int img_width = img.cols;
	int img_height = img.rows;
	uchar *p1 = (uchar*)malloc(sizeof(uchar)*img_height*img_width * 3);
	for (int i = 0; i < img_width * img_height * 3; i++)
	{
		p1[i] = (uchar)img.at<cv::Vec3b>(i / (img_width * 3), (i % (img_width * 3)) / 3)[i % 3];
	}
	return p1;
}

// opencv的Mat格式转为uchar格式（单通道的8位的图片）
uchar* gSLICrTools::matToUchar_oneChanel(cv::Mat img)
{
	int img_width = img.cols;
	int img_height = img.rows;
	uchar *p1 = (uchar*)malloc(sizeof(uchar)*img_height*img_width);
	
	int count = 0;
	for (int i = 0; i < img.rows; i++){
		for (int j = 0; j < img.cols; j++)
		{
			p1[count] = img.at<uchar>(i, j);  // 16位是2字节，用ushort。这里是8位的，则用uchar。
			count++;     // count的值也等于i*img.cols + j。   i:高（从0开始)，j：宽（从0开始)
		}
	}
	return p1;
}

// opencv的Mat格式转为ushort格式（单通道的16位的图片）
ushort* gSLICrTools::matToUshort_oneChanel(cv::Mat img)
{
	int img_width = img.cols;
	int img_height = img.rows;
	ushort *p1 = (ushort*)malloc(sizeof(ushort)*img_height*img_width);

	int count = 0;
	for (int i = 0; i < img.rows; i++)
	{
		for (int j = 0; j < img.cols; j++)
		{
			// p1[count] = img.at<ushort>(i, j);  // 16位是2字节，用ushort
			p1[count] = (unsigned short)(img.at<float>(i, j)*1000);  // depth  归一化深度  float 0-1
			count++;     // count的值也等于i*img.cols + j。   i:高（从0开始)，j：宽（从0开始)
		}
	}
	return p1;
}
