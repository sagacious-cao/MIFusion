// Copyright (c) 2021 Lukas Koestler, Nan Yang. All rights reserved.

#include "tandem_backend.h"

bool cmpx(cv::Point a, cv::Point b) 
{
    return a.x < b.x;
}

bool cmpy(cv::Point a, cv::Point b) 
{
    return a.y < b.y;
}

class TandemBackendImpl {
public:
  SampleMaskRCNN* maskrcnn;
  gSLICrTools*  slicr;
  int* finalSPixel;		// 存最终RGBD超像素分割结果

  explicit TandemBackendImpl(
      int width, int height, bool dense_tracking,
      DrMvsnet *mvsnet, DrFusion *fusion,
      float mvsnet_discard_percentage,
      Timer *dr_timer,
      std::vector<dso::IOWrap::Output3DWrapper *> const &outputWrapper,
      int mesh_extraction_freq,
      SampleMaskRCNN* maskrcnn,
      gSLICrTools*  slicr) : \
      width(width), height(height), dense_tracking(dense_tracking), mvsnet(mvsnet), fusion(fusion), \
      mvsnet_discard_percentage(mvsnet_discard_percentage), dr_timer(dr_timer), outputWrapper(outputWrapper), \
      mesh_extraction_freq(mesh_extraction_freq), get_mesh(mesh_extraction_freq > 0),maskrcnn(maskrcnn), slicr(slicr){
    dr_timing = dr_timer != nullptr;
    coarse_tracker_depth_map_A = new TandemCoarseTrackingDepthMap(width, height);
    coarse_tracker_depth_map_B = new TandemCoarseTrackingDepthMap(width, height);
    coarse_tracker_depth_map_valid = nullptr;
    coarse_tracker_depth_map_use_next = coarse_tracker_depth_map_A;

    finalSPixel = (int*)malloc(sizeof(int)*width*height);
    
    worker_thread = boost::thread(&TandemBackendImpl::Loop, this);
  };

  ~TandemBackendImpl() {
    delete coarse_tracker_depth_map_B;
    delete coarse_tracker_depth_map_A;
    free(finalSPixel);
  };


  // Blocking for last input. Non-blocking for this input.
  void CallAsync(
      int view_num_in,
      int index_offset_in,
      int corrected_ref_index_in,
      std::vector<cv::Mat> const &bgrs_in,
      cv::Mat const &intrinsic_matrix_in,
      std::vector<cv::Mat> const &cam_to_worlds_in,
      float depth_min_in,
      float depth_max_in,
      cv::Mat const &coarse_tracker_pose_in
  );

  // Non-blocking
  bool Ready();

  void Wait();

  boost::mutex &GetTrackingDepthMapMutex() { return mut_coarse_tracker; }

  TandemCoarseTrackingDepthMap const *GetTrackingDepthMap() { return coarse_tracker_depth_map_valid; };

private:
  void CallSequential();
  void Loop();

  void maskSuperPixelFilter_OverSeg(int spNum, int *finalSPixel, int resultMasks_Num,std::vector<std::pair<cv::Mat,MaskRCNNUtils::BBoxInfo>> &results);
  void visualMask(cv::Mat visual_img,char* name,std::vector<std::pair<cv::Mat,MaskRCNNUtils::BBoxInfo>> results);
  void visualRgb(cv::Mat rgb);
  void visualMasks(std::vector<cv::Mat> InstanceMasks);
    struct RenderBBox
    {
        float x1, y1, x2, y2;
        unsigned char Instanceindex;
    };
    std::vector<RenderBBox> renderInstanceBBoxResults;

    float calculateIoU(MaskRCNNUtils::BBox maskrcnnBBox,RenderBBox renderBBox);
    void computeCompareMap(std::vector<std::pair<cv::Mat,MaskRCNNUtils::BBoxInfo>> &maskrcnnResults,
                                                                                                        std::vector<RenderBBox>& RenderResults,
                                                                                                        float* computeCompareMap);
    void processInstance(
        std::vector<std::pair<cv::Mat,MaskRCNNUtils::BBoxInfo>> &maskrcnnResults,
        std::vector<RenderBBox>& RenderResults,
        int InstanceNum);

    // bool cmpx(cv::Point a, cv::Point b);
    // bool cmpy(cv::Point a, cv::Point b);

    void ExtractInstanceBBox(std::vector<cv::Mat>& RenderInstanceMasks,std::vector<RenderBBox>& BBoxResults);
    void visualRender(unsigned char* bgr, unsigned char* instance_bgr, std::vector<cv::Mat> RenderInstanceMasks,std::vector<RenderBBox> BBoxResults);
    unsigned char*  maskrcnnMask_cpu = nullptr;
    float* scores_cpu = nullptr;

    unsigned char*  maskrcnnMask_gpu = nullptr;
    float* scores_gpu = nullptr;



    void setmaskrcnnMask(std::vector<std::pair<cv::Mat,MaskRCNNUtils::BBoxInfo>> &results);
    void copyMaskToGpu(int masks_num);

    std::vector<cv::Mat> InstanceMasks;
    
    unsigned char* classColorList = nullptr;
    void visualMaskRCNNMask(cv::Mat visual_img,char* name,std::vector<std::pair<cv::Mat,MaskRCNNUtils::BBoxInfo>> results);
    void generateClassColor();

    void LargestConnecttedComponent(cv::Mat srcImage, cv::Mat &dstImage);

    const bool dense_tracking;

    DrMvsnet *mvsnet = nullptr;
    DrFusion *fusion = nullptr;

    // Will run Loop.
    boost::thread worker_thread;

    // Protects all below variables
    boost::mutex mut;
    bool running = true;
    bool unprocessed_data = false;

    boost::condition_variable newInputSignal;
    boost::condition_variable dataProcessedSignal;

    // Internal
    bool setting_debugout_runquiet = true;
    bool dr_timing;
    int call_num = 0;
    int last_render_instance_num = 0;
    int render_instance_num = 0;
    bool render_success = true;
    const bool get_mesh = true;

    float mesh_lower_corner[3] = {-5, -5, -5};
    float mesh_upper_corner[3] = {5, 5, 5};
    const int mesh_extraction_freq;

    const int width;
    const int height;
    float mvsnet_discard_percentage;
    Timer *dr_timer;

    std::vector<dso::IOWrap::Output3DWrapper *> const &outputWrapper;

    //
    boost::mutex mut_coarse_tracker;
    TandemCoarseTrackingDepthMap *coarse_tracker_depth_map_valid = nullptr;
    TandemCoarseTrackingDepthMap *coarse_tracker_depth_map_use_next = nullptr;
    TandemCoarseTrackingDepthMap *coarse_tracker_depth_map_A;
    TandemCoarseTrackingDepthMap *coarse_tracker_depth_map_B;

    // data from call
    bool has_to_wait_current = false;
    cv::Mat intrinsic_matrix_current;
    int index_offset_current;
    int view_num_current;
    int corrected_ref_index_current;
    std::vector<cv::Mat> bgrs_current;
    std::vector<cv::Mat> cam_to_worlds_current;
    float depth_min_current;
    float depth_max_current;

    DrMvsnetOutput *output_previous;


    // data from last call
    bool has_to_wait_previous = false;
    cv::Mat intrinsic_matrix_previous;
    int index_offset_previous;
    int view_num_previous;
    int corrected_ref_index_previous;
    std::vector<cv::Mat> bgrs_previous;
    std::vector<cv::Mat> cam_to_worlds_previous;
    float depth_min_previous;
    float depth_max_previous;

};

void TandemBackendImpl::Loop() {
  boost::unique_lock<boost::mutex> lock(mut);
  while (running) {
    if (unprocessed_data) {
      CallSequential();
      unprocessed_data = false;
      dataProcessedSignal.notify_all();
    }
    newInputSignal.wait(lock);
  }
}


// 用rgbd超像素分割的结果去优化原始的maskrcnn的mask
void TandemBackendImpl::maskSuperPixelFilter_OverSeg(int spNum, int *finalSPixel, int resultMasks_Num,std::vector<std::pair<cv::Mat,MaskRCNNUtils::BBoxInfo>>  &results)
{
	//Step 1 count pixel num。NumMatrix就是用来存每个mask和每个超像素重叠的像素数量，最后的spNum个元素存的是每个超像素标号的像素数
	int* PixelValueOfEachMask = (int*)malloc(resultMasks_Num * sizeof(int)); // 用于存每个mask的标签值
	int* PixelNumOfEachSPixel = (int*)malloc(spNum*sizeof(int));                         //每种超像素像素总量
    int* NumMatrix = (int*)malloc((resultMasks_Num)*spNum*sizeof(int));       // 每个mask上每种超像素数量，多个1是用来存每个超像素的像素数量

    memset(PixelValueOfEachMask, 0, resultMasks_Num*sizeof(int));
    memset(PixelNumOfEachSPixel, 0, spNum*sizeof(int));
	memset(NumMatrix, 0, resultMasks_Num*spNum*sizeof(int));

	for (int i = 0; i<resultMasks_Num; i++)  // 遍历masks上每一个对应位置上的像素点
    {
        PixelValueOfEachMask[i] = results[i].second.label+1;
    }
  
    for (int x = 0; x<width; x++)  // 遍历超像素上的每一个像素点
	{
		for (int y = 0; y<height; y++)
		{
            int pixelIndex = y*width + x;
			int id = finalSPixel[pixelIndex];  // 像素坐标为(x,y)的超像素标号
			if (id >= spNum || id<0) 
                continue;
			//last row(pointNum of each superPixel) 统计每个超像素的像素数量
			PixelNumOfEachSPixel[id] ++;

			for (int i = 0; i<resultMasks_Num; i++)  // 遍历masks上每一个对应位置上的像素点
			{

				if (results[i].first.data[pixelIndex])  // 如果mask对应像素有值
				{
					NumMatrix[i*spNum + id] ++;  // 第i个mask和第id个超像素的重叠数加1
				}
			}
		}
	}

	// Step 2
	for (int x = 0; x<width; x++)  // 遍历超像素上的每一个像素点
	{
		for (int y = 0; y<height; y++)
		{
            int pixelIndex = y*width + x;
			int id = finalSPixel[pixelIndex];
			
            if (id >= spNum || id<0)  // 若像素点的超像素标号处于正常范围之外，该点的所有mask都直接置为0
			{
                // std::cout<<"error id:"<<id<<std::endl;
				for (int i = 0; i<resultMasks_Num; i++) 
                    results[i].first.data[pixelIndex] = 0;
				continue;
			}

			for (int i = 0; i<resultMasks_Num; i++)
			{
				int n = PixelNumOfEachSPixel[id];  // 超像素标号为id的超像素的像素个数
				int m = NumMatrix[i*spNum + id];  // 第i个mask和第id个超像素的像素重叠数
				//if(n<filterNumThreshold) continue;

				float test = m*1.0f / n;
				if (test>0.75)//&&n>filterNumThreshold 超像素内某种标签的mask占了75%以上，就把mask上该超像素内所有像素都置为mask标签值
				{
                    results[i].first.data[pixelIndex] = PixelValueOfEachMask[i];
				}
				else
				{
                    results[i].first.data[pixelIndex] = 0;
				}
			}
		}
	}


	free(NumMatrix);
	free(PixelValueOfEachMask);
    free(PixelNumOfEachSPixel);
}

void TandemBackendImpl::generateClassColor()
{
    int a =1,b = 255;
    srand((int)time(NULL));
    classColorList = (unsigned char*)malloc(81*3*sizeof(unsigned char));
    for(int i = 0;i<81;i++)
    {
        classColorList[i] = a+rand()%(b-a+1);
    }
}

void TandemBackendImpl::visualMaskRCNNMask(cv::Mat img,char* name,std::vector<std::pair<cv::Mat,MaskRCNNUtils::BBoxInfo>> results)
{
  std::cout<<"visualMaskRCNNMaskNum:"<<results.size()<<std::endl;
  if(classColorList==nullptr)
  {
    generateClassColor();
  }
  
  if(results.size()>0)
  {
    cv::Mat visual_img;
    img.copyTo(visual_img);
    static int masknum = 0;
    
    cv::Mat mask = cv::Mat(visual_img.size(),CV_8UC1);
    for(int i = 0;i<results.size();i++)
    {   
      int label = results[i].second.label;
      printf("%d\n",label);
      cv::Scalar color(classColorList[label*3+0],classColorList[label*3+1],classColorList[label*3+2]);
      cv::Mat colormap = cv::Mat(visual_img.size(),CV_8UC3,color);
      mask = results[i].first*255;
      cv::add(visual_img,colormap,visual_img,mask);
      cv::rectangle(visual_img,cv::Point(results[i].second.box.x1,results[i].second.box.y1),cv::Point(results[i].second.box.x2,results[i].second.box.y2),color,1,1,0);
    }
    char imagename[100];
    sprintf(imagename,"visual/test_maskrcnn/debugmask_%d_%s.png",masknum,name);
    cv::imwrite(imagename,visual_img);
    masknum++;
  }
}


void TandemBackendImpl::visualMask(cv::Mat img,char* name,std::vector<std::pair<cv::Mat,MaskRCNNUtils::BBoxInfo>> results)
{
    std::cout<<"visualMaskNum:"<<results.size()<<std::endl;
    static int masknum = 0;
    cv::Mat visual_img;
    img.copyTo(visual_img);
    if(results.size()>0)
    {

        
        
        cv::Mat mask = cv::Mat(visual_img.size(),CV_8UC1);
        for(int i = 0;i<results.size();i++)
        {   
            int label = results[i].second.label;
            cv::Scalar color;
            fusion->GetInstanceColor(label,color);
            cv::Mat colormap = cv::Mat(visual_img.size(),CV_8UC3,color);
            mask = results[i].first/(results[i].second.label+1)*255;
            cv::add(visual_img,colormap,visual_img,mask);
            
            //在图像上加字符
            //第一个参数为要加字符的目标函数
            //第二个参数为要加的字符
            //第三个参数为字体
            //第四个参数为子的粗细
            //第五个参数为字符的颜色
            char instanceIndex[10];
            sprintf(instanceIndex,"%d",results[i].second.label);
            cv::Point p = cv::Point(results[i].second.box.x1,results[i].second.box.y1);
            cv::putText(visual_img, instanceIndex, p, cv::FONT_HERSHEY_TRIPLEX, 0.8, color, 2, CV_AA);
                


            cv::rectangle(visual_img,cv::Point(results[i].second.box.x1,results[i].second.box.y1),cv::Point(results[i].second.box.x2,results[i].second.box.y2),color,1,1,0);
        }
        
        
    }
    char imagename[100];
    sprintf(imagename,"visual/test_render/%d_0_debugmaskrcnninstance.png",masknum);
    cv::imwrite(imagename,visual_img);
    masknum ++;
}

void  TandemBackendImpl::visualRender(unsigned char* bgr, unsigned char* instance_bgr, std::vector<cv::Mat> RenderInstanceMasks,std::vector<RenderBBox> BBoxResults)
{

  static int bgr_num = 0;
  cv::Mat bgr_tmp(height,width,CV_8UC3);
  memcpy(bgr_tmp.data,bgr,width*height*3*sizeof(unsigned char));
  
  cv::Mat bgr_v_tmp(height,width,CV_8UC3);
  memcpy(bgr_v_tmp.data,bgr,width*height*3*sizeof(unsigned char));
  
  cv::Mat instance_bgr_tmp(height,width,CV_8UC3);
  memcpy(instance_bgr_tmp.data,instance_bgr,width*height*3*sizeof(unsigned char));

  int instanceNum = BBoxResults.size();
  char imagename[100];
  
  for(int i = 0;i<instanceNum;i++)
  {   
    int label = BBoxResults[i].Instanceindex-1;
    cv::Scalar color;
    fusion->GetInstanceColor(label,color);
  
    cv::Mat colormap = cv::Mat(bgr_tmp.size(),CV_8UC3,color);
 
    cv::Mat mask = RenderInstanceMasks[label]*255;

    cv::add(bgr_tmp,colormap,bgr_tmp,mask);
    // cv::rectangle(bgr_tmp,cv::Point(BBoxResults[i].x1,BBoxResults[i].y1),cv::Point(BBoxResults[i].x2,BBoxResults[i].y2),color,1,1,0);
    
    char instanceIndex[10];
    sprintf(instanceIndex,"%d",BBoxResults[i].Instanceindex-1);
    cv::Point p = cv::Point(BBoxResults[i].x1,BBoxResults[i].y1+5);
    // cv::putText(instance_bgr_tmp, instanceIndex, p, cv::FONT_HERSHEY_TRIPLEX, 0.8, color, 2, CV_AA);
        


    // cv::add(instance_bgr_tmp,colormap,instance_bgr_tmp,mask);
    // cv::rectangle(instance_bgr_tmp,cv::Point(BBoxResults[i].x1,BBoxResults[i].y1),cv::Point(BBoxResults[i].x2,BBoxResults[i].y2),color,1,1,0);
    
  }
  
  

  sprintf(imagename,"visual/test_render_rgb/%d_1_debugrender_v.png",bgr_num);
  cv::imwrite(imagename,bgr_v_tmp);

//   sprintf(imagename,"visual/test_render/%d_3_debugrender.png",bgr_num);
//   cv::imwrite(imagename,bgr_tmp);

  sprintf(imagename,"visual/test_render/%d_1_%d_mask_debugrender_instance.png",bgr_num,BBoxResults.size());
  cv::imwrite(imagename,instance_bgr_tmp);



  bgr_num++;

}

void TandemBackendImpl::visualRgb(cv::Mat rgb)
{
  static int rgb_num = 0;
  char imagename[500];
  if(rgb_num>=0)
  {
    sprintf(imagename,"visual/test_render_rgb/%d_0_debugrgb.png",rgb_num);
    cv::imwrite(imagename,rgb);
  }
  rgb_num++;
}

void TandemBackendImpl::visualMasks(std::vector<cv::Mat> InstanceMasks)
{
    int InstanceNum = InstanceMasks.size();
    std::cout<<"Render Masks Num:"<<InstanceNum<<std::endl;
    static int render_num = 0;
    for(int i = 0;i<InstanceNum;i++)
    {
        char imagename[100];
        sprintf(imagename,"visual/test_render_mask/debugrendermask_%d_%d.png",render_num,i);
        cv::imwrite(imagename,InstanceMasks[i]);
    }
    render_num++;
}

void TandemBackendImpl::ExtractInstanceBBox(std::vector<cv::Mat>& RenderInstanceMasks,std::vector<RenderBBox>& BBoxResults)
{
    BBoxResults.clear();
    int InstanceNum = RenderInstanceMasks.size();
    static int mask_index = 0;
    char imagename[100];
    
    for(int i=0;i<InstanceNum;i++)
    {
        RenderBBox bbox;
        std::vector<cv::Point> idx;
        cv::Mat mask;// = RenderInstanceMasks[i]*255;
        cv::Mat mask_f;
        RenderInstanceMasks[i].copyTo(mask);
        
        // double minValue, maxValue;
        // cv::Point  minIdx, maxIdx;    // 最小值坐标，最大值坐标     
        // cv::minMaxLoc(mask, &minValue, &maxValue, &minIdx, &maxIdx);
        // std::cout << "最大值：" << maxValue <<"最小值："<<minValue<<std::endl;


        // mask = mask*255;
        // LargestConnecttedComponent(mask,mask_f);
        // cv::findNonZero(mask_f/255, idx);
        
        cv::findNonZero(mask, idx);
        sprintf(imagename,"visual/test_render_mask/%d_0_%d_before.png",mask_index,i);
        cv::imwrite(imagename,mask*255);
        
        if(idx.size()>0 && idx.size()<640*480)
        {
            // printf("idx = %d\n",idx.size());
        
            
            
            // sprintf(imagename,"visual/test_render/%d_0_%d_after.png",mask_index,i);
            // cv::imwrite(imagename,mask_f);
        

            cv::Point maxx = *std::max_element(idx.begin(), idx.end(), cmpx);
            cv::Point minx = *std::min_element(idx.begin(), idx.end(), cmpx);

            cv::Point maxy = *std::max_element(idx.begin(), idx.end(), cmpy);
            cv::Point miny = *std::min_element(idx.begin(), idx.end(), cmpy);
            // cv::rectangle(mask,cvPoint(minx.x,miny.y),cvPoint(maxx.x,maxy.y),255,1,1,0);
            bbox.x1 = minx.x;
            bbox.y1 = miny.y;
            bbox.x2 = maxx.x;
            bbox.y2 = maxy.y;
            bbox.Instanceindex = i+1;
            BBoxResults.push_back(bbox);
        }
    }
    // if(BBoxResults.size()==0)
    // {
    //     std::cout<<"-------------------------------------------------------------------------------   "<<BBoxResults.size()<<std::endl;
    //     for(int i = 0;i<RenderInstanceMasks.size();i++)
    //     {
    //         sprintf(imagename,"visual/test_render_mask/%d_%d.png",mask_index,i);
    //         cv::imwrite(imagename,RenderInstanceMasks[i]*255);
    //     }
    // }
    mask_index++;

}

void TandemBackendImpl::LargestConnecttedComponent(cv::Mat srcImage, cv::Mat &dstImage)
{
    cv::Mat temp;
    cv::Mat labels;
    srcImage.copyTo(temp);

    //1. 标记连通域
    int n_comps = connectedComponents(temp, labels, 4, CV_16U);
    // printf("n_comps = %d\n",n_comps);
    std::vector<int> histogram_of_labels;
    for (int i = 0; i < n_comps; i++)//初始化labels的个数为0
    {
        histogram_of_labels.push_back(0);
    }

    int rows = labels.rows;
    int cols = labels.cols;
    for (int row = 0; row < rows; row++) //计算每个labels的个数
    {
        for (int col = 0; col < cols; col++)
        {
            histogram_of_labels.at(labels.at<unsigned short>(row, col)) += 1;
        }
    }
    histogram_of_labels.at(0) = 0; //将背景的labels个数设置为0

    //2. 计算最大的连通域labels索引
    int maximum = 0;
    int max_idx = 0;
    for (int i = 0; i < n_comps; i++)
    {
        if (histogram_of_labels.at(i) > maximum)
        {
            maximum = histogram_of_labels.at(i);
            max_idx = i;
        }
    }

    //3. 将最大连通域标记为1
    for (int row = 0; row < rows; row++) 
    {
        for (int col = 0; col < cols; col++)
        {
            if (labels.at<unsigned short>(row, col) == max_idx)
            {
                labels.at<unsigned short>(row, col) = 255;
            }
            else
            {
                labels.at<unsigned short>(row, col) = 0;
            }
        }
    }

    //4. 将图像更改为CV_8U格式
    labels.convertTo(dstImage, CV_8U);
}

/*
// void TandemBackendImpl::processInstance(
//                                                                                             std::vector<std::pair<cv::Mat,MaskRCNNUtils::BBoxInfo>> &maskrcnnResults,
//                                                                                             std::vector<RenderBBox> RenderResults,
//                                                                                             int InstanceNum)
// {
//     if(InstanceNum == 0)
//     {
//         for(int i = 0;i<maskrcnnResults.size();i++)
//         {
//             if(maskrcnnResults[i].second.prob > 0.8)
//             {
//                 fusion->InstanceTableAdd(maskrcnnResults[i].second.label);
//                 maskrcnnResults[i].first*= (i+1);
//                 maskrcnnResults[i].second.label = i;
//             }
//         }
//     }
//     else
//     {
//         // std::vector<int> comparedflag(maskrcnnResults.size(),-1);  //>=0 match index  ,-1 not match,   -2 match but not max
//         for(int j = 0;j<RenderResults.size();j++)
//         {
//             float MaxIoU = 0;
//             int MaxIoUIndex = -1;
//             for(int i = 0;i<maskrcnnResults.size();i++)
//             {
//                 float IoU = calculateIoU(maskrcnnResults[i].second.box,RenderResults[j]);
//                 if(IoU>maskrcnnResults[i].second.IoU)
//                 {
//                     maskrcnnResults[i].second.IoU = IoU;
//                     if(IoU>MaxIoU)
//                     {
//                         if(IoU>0.4)
//                         {
//                             if(MaxIoU!=0)
//                             {    
//                                 // comparedflag[MaxIoUIndex] = -2;                       
//                                 maskrcnnResults[MaxIoUIndex].second.matchIndex = -2;
//                             }
//                             // comparedflag[i] = j;
//                             MaxIoU = IoU;
//                             MaxIoUIndex = i;
                            
//                             maskrcnnResults[i].second.matchIndex = j;
//                         }
//                         else
//                         {
//                             maskrcnnResults[i].second.matchIndex = -2;    
//                         }
//                     }
//                     else if(IoU>0 && IoU<=MaxIoU)
//                     {
//                         // comparedflag[i] = -2;
//                         maskrcnnResults[i].second.matchIndex = -2;
//                     }
//                 }
//             }
//         }

//         for(int i = 0;i<maskrcnnResults.size();i++)
//         {
//             if(maskrcnnResults[i].second.matchIndex == -2)
//             {
//                 //delete ith maskrcnn result
//                 maskrcnnResults.erase(maskrcnnResults.begin()+i);
//                 i--;
//             }
//             else if(maskrcnnResults[i].second.matchIndex == -1)
//             {
//                 //add instance
//                 fusion->InstanceTableAdd(maskrcnnResults[i].second.label);
//                 maskrcnnResults[i].second.label = fusion->GetInstanceNum()-1;
//                 maskrcnnResults[i].first*= (fusion->GetInstanceNum());
//             }
//             else if(maskrcnnResults[i].second.matchIndex>=0) 
//             {
//                 //update instance table
//                 // maskrcnnResults[i].first*= ( RenderResults[j].Instanceindex);
//                 maskrcnnResults[i].first*= ( RenderResults[maskrcnnResults[i].second.matchIndex].Instanceindex);
//                 // printf("update instance Table:%d,%d\n",RenderResults[maskrcnnResults[i].second.matchIndex].Instanceindex,maskrcnnResults[i].second.label);
//                 fusion->UpdateInstanceTable(RenderResults[maskrcnnResults[i].second.matchIndex].Instanceindex,maskrcnnResults[i].second.label); 
//                 maskrcnnResults[i].second.label = RenderResults[maskrcnnResults[i].second.matchIndex].Instanceindex-1;
//             }
//         }





//         // for(int i = 0;i<maskrcnnResults.size();i++)
//         // {
//         //     if(maskrcnnResults[i].second.prob > 0.8)
//         //     {
//         //         int cmpflag = 0;
//         //         for(int j = 0;j<RenderResults.size();j++)
//         //         {
//         //             //如果匹配成功
//         //             float IoU = calculateIoU(maskrcnnResults[i].second.box,RenderResults[j]);
//         //             if( IoU> 0.4)
//         //             {
//         //                 // printf("%f %f %f %f   %f %f %f %f  %f\n",
//         //                 //                 maskrcnnResults[i].second.box.x1,
//         //                 //                 maskrcnnResults[i].second.box.x2,
//         //                 //                 maskrcnnResults[i].second.box.y1,
//         //                 //                 maskrcnnResults[i].second.box.y2,
//         //                 //                 RenderResults[j].x1,
//         //                 //                 RenderResults[j].x2, 
//         //                 //                 RenderResults[j].y1,
//         //                 //                 RenderResults[j].y2,
//         //                 //                 IoU);

//         //                 maskrcnnResults[i].first*= ( RenderResults[j].Instanceindex);
//         //                 fusion->UpdateInstanceTable(RenderResults[j].Instanceindex,maskrcnnResults[i].second.label); 
//         //                 maskrcnnResults[i].second.label = RenderResults[j].Instanceindex-1;
//         //                 RenderResults.erase(RenderResults.begin()+j);
//         //                 cmpflag = 1;
//         //                 break;  
//         //             }
//         //         }
//         //         if(cmpflag == 0)
//         //         {
//         //             fusion->InstanceTableAdd(maskrcnnResults[i].second.label);
//         //             maskrcnnResults[i].second.label = fusion->GetInstanceNum()-1;
//         //             maskrcnnResults[i].first*= (fusion->GetInstanceNum());
//         //         }
//         //     }
//         // }
    
//     }
// }
*/

float TandemBackendImpl::calculateIoU(MaskRCNNUtils::BBox maskrcnnBBox,RenderBBox renderBBox)
{
    float iou = 0;

    // def IOU(Reframe, GTframe):
    // 	# 得到第一个矩形的左上坐标及宽和高
    // 	x1 = Reframe[0]
    // 	y1 = Reframe[1]
    // 	width1 = Reframe[2] - Reframe[0]
    // 	height1 = Reframe[3] - Reframe[1]
    float x1 = maskrcnnBBox.x1;
    float y1 = maskrcnnBBox.y1;
    float width1 = maskrcnnBBox.x2-maskrcnnBBox.x1;
    float height1 = maskrcnnBBox.y2-maskrcnnBBox.y1;

    // 	# 得到第二个矩形的左上坐标及宽和高
    // 	x2 = GTframe[0]
    // 	y2 = GTframe[1]
    // 	width2 = GTframe[2] - GTframe[0]
    // 	height2 = GTframe[3] - GTframe[1]
    float x2 = renderBBox.x1;
    float y2 = renderBBox.y1;
    float width2 = renderBBox.x2-renderBBox.x1;
    float height2 = renderBBox.y2-renderBBox.y1;

    // 	# 计算重叠部分的宽和高
    // 	endx = max(x1+width1, x2+width2)
    // 	startx = min(x1, x2)
    // 	width = width1 + width2 - (endx - startx)
    float endx = std::max(x1+width1, x2+width2);
    float startx = std::min(x1, x2);
    float width = width1 + width2 - (endx - startx);


    // 	endy = max(y1+height1, y2+height2)
    // 	starty = min(y1, y2)
    // 	height = height1 + height2 - (endy - starty)

    float endy = std::max(y1+height1, y2+height2);
    float starty = std::min(y1, y2);
    float height = height1 + height2 - (endy - starty);


    // 	# 如果重叠部分为负, 即不重叠
    // 	if width <= 0 or height <= 0:
    // 		ratio = 0
    // 	else:
    // 		Area = width * height
    // 		Area1 = width1 * height1
    // 		Area2 = width2 * height2
    // 		ratio = Area*1. / (Area1+Area2-Area)
	if(width <= 0 || height <= 0)
    {
		iou = 0;
	}
    else
    {
        float Area = width * height;
		float Area1 = width1 * height1;
		float Area2 = width2 * height2;
		// float iou1 = Area*1.0 / (Area1);
        // float iou2 = Area*1.0 / (Area2);
        // iou = iou1>iou2?iou1:iou2;
        // iou = iou2;
        iou = Area*1.0 / (Area1+Area2-Area);
    }
		
    return  iou;
}

void TandemBackendImpl::computeCompareMap(std::vector<std::pair<cv::Mat,MaskRCNNUtils::BBoxInfo>> &maskrcnnResults,
                                                                                                         std::vector<RenderBBox>& RenderResults,
                                                                                                         float* compareMap)
{
    int maskrcnn_num = maskrcnnResults.size();
    int render_num = RenderResults.size();

    for(int i = 0;i<maskrcnn_num;i++)
    {
        for(int j=0;j<render_num;j++)
        {
            float IoU = calculateIoU(maskrcnnResults[i].second.box,RenderResults[j]);
            // std::cout<<"BBox"<<std::endl;
            // std::cout<<maskrcnnResults[i].second.box.x1<<"  "<<maskrcnnResults[i].second.box.x2<<"  "<<maskrcnnResults[i].second.box.y1<<"  "<<maskrcnnResults[i].second.box.y2<<"  "<<std::endl;
            // std::cout<<RenderResults[j].x1<<" "<<RenderResults[j].x2<<" "<<RenderResults[j].y1<<" "<<RenderResults[j].y2<<" "<<std::endl;
            // std::cout<<"IOU"<<std::endl;
            // std::cout<<IoU<<std::endl;
            compareMap[i*render_num+j] = IoU;
        }
    }

}

void TandemBackendImpl::processInstance(
                                                                                            std::vector<std::pair<cv::Mat,MaskRCNNUtils::BBoxInfo>> &maskrcnnResults,
                                                                                            std::vector<RenderBBox>& RenderResults,
                                                                                            int InstanceNum)
{

    if(InstanceNum == 0)
    {
        for(int i = 0;i<maskrcnnResults.size();i++)
        {
            if(maskrcnnResults[i].second.prob > 0.8)
            {
                fusion->InstanceTableAdd(maskrcnnResults[i].second.label);
                maskrcnnResults[i].first*= (i+1);
                maskrcnnResults[i].second.label = i;
            }
        }
    }
    else
    {
        int maskrcnn_num = maskrcnnResults.size();
        int render_num = RenderResults.size();
        float* compareMap = (float*)malloc(maskrcnn_num * render_num * sizeof(float));
        computeCompareMap(maskrcnnResults,RenderResults,compareMap);
        for(int i = 0;i<maskrcnn_num;i++)
        {
            int bestIoUIndex = -1;
            int bestIoU = 0;
            for(int j=0;j<render_num;j++)
            {

                // std::cout<<compareMap[i*render_num+j]<<std::endl;
                if(compareMap[i*render_num+j]>0.6 && compareMap[i*render_num+j] > bestIoU)
                {
                    bestIoU = compareMap[i*render_num+j];
                    bestIoUIndex = j;
                }
            }
            if(bestIoUIndex >-1)
            {
                maskrcnnResults[i].first*= ( RenderResults[bestIoUIndex].Instanceindex);
                fusion->UpdateInstanceTable(RenderResults[bestIoUIndex].Instanceindex,maskrcnnResults[i].second.label); 
                maskrcnnResults[i].second.label = RenderResults[bestIoUIndex].Instanceindex-1;
                // RenderResults.erase(RenderResults.begin()+j);
            }
            else
            {
                fusion->InstanceTableAdd(maskrcnnResults[i].second.label);
                maskrcnnResults[i].second.label = fusion->GetInstanceNum()-1;
                maskrcnnResults[i].first*= (fusion->GetInstanceNum());
            }
        }
        free(compareMap);
    }
}

void TandemBackendImpl::copyMaskToGpu(int resultMasks_Num_cpu)
{
    cudaFree(maskrcnnMask_gpu);
    cudaFree(scores_gpu);
    cudaMalloc((void**)&maskrcnnMask_gpu, sizeof(unsigned char)*width*height*resultMasks_Num_cpu);
    cudaMalloc((void**)&scores_gpu, resultMasks_Num_cpu*sizeof(float));
    
    cudaMemcpy(maskrcnnMask_gpu,maskrcnnMask_cpu,sizeof(unsigned char)*width*height*resultMasks_Num_cpu,cudaMemcpyHostToDevice);
    maskCleanOverlap(maskrcnnMask_gpu,resultMasks_Num_cpu,width,height);//(unsigned char* masks, const int masksNum,const int width, const int height)
    cudaMemcpy(scores_gpu,scores_cpu,sizeof(float)*resultMasks_Num_cpu,cudaMemcpyHostToDevice);
    
    free(maskrcnnMask_cpu);
    free(scores_cpu);
}

void TandemBackendImpl::setmaskrcnnMask(std::vector<std::pair<cv::Mat,MaskRCNNUtils::BBoxInfo>> &results)
{
    short resultMasks_Num_cpu = (short)results.size();
    maskrcnnMask_cpu = (unsigned char*) malloc(sizeof(unsigned char)*width*height*resultMasks_Num_cpu);
    scores_cpu = (float*) malloc(sizeof(float)*resultMasks_Num_cpu);
    for(int i =0 ;i<resultMasks_Num_cpu;i++)
    {
        memcpy(maskrcnnMask_cpu+i*width*height, results[i].first.data,sizeof(unsigned char)*width*height);
        memcpy(scores_cpu+i,&(results[i].second.prob),sizeof(float));
    }

}

void TandemBackendImpl::CallSequential() {
  int id_time;
  call_num++;

  /* --- 3.5 CURRENT: Call MVSNet Async --- */
  std::vector<unsigned char *> bgrs_current_ptr;
  for (auto const &e : bgrs_current) bgrs_current_ptr.push_back(e.data);
  std::vector<float *> cam_to_worlds_current_ptr;
  for (auto const &e : cam_to_worlds_current) cam_to_worlds_current_ptr.push_back((float *) e.data);

  mvsnet->CallAsync(
      height,
      width,
      view_num_current,
      corrected_ref_index_current,
      bgrs_current_ptr.data() + index_offset_current,
      (float *) intrinsic_matrix_current.data,
      cam_to_worlds_current_ptr.data() + index_offset_current,
      depth_min_current,
      depth_max_current,
      mvsnet_discard_percentage,
      false
  );
  has_to_wait_current = true;

  // Here we have the lock (the loop function has it)
    // std::cout<<render_instance_num<<"     "<<last_render_instance_num<<std::endl;;
    if(render_success)
    {
        if (has_to_wait_previous && fusion != nullptr) 
        {
            cv::Mat  bgr_tmp(height,width,CV_8UC3);
            cv::Mat  visual_img;
            cv::Mat  bgr_slicr;
            memcpy(bgr_tmp.data,bgrs_previous[corrected_ref_index_previous].data,width*height*3);
            bgr_tmp.copyTo(visual_img);
            bgr_tmp.copyTo(bgr_slicr);
            
            // visualRgb(visual_img);

            maskrcnn->infer(bgr_tmp);
            // int masks_num = maskrcnn->getMaskNum();
            
            // printf("masks_num:%d\n",masks_num);
            std::vector<std::pair<cv::Mat,MaskRCNNUtils::BBoxInfo>> maskrcnnResults = maskrcnn->results;

            cv::Mat depth_tmp(height,width,CV_8UC1);
            // // cv::imwrite("depth_from_mvsnet.png",depth_tmp);
            slicr->rgbdSuperPixelSeg(bgr_slicr,depth_tmp,finalSPixel,visual_img);
            // printf("masks_num:%d\n",maskrcnnResults.size());
            processInstance(maskrcnnResults,renderInstanceBBoxResults,fusion->GetInstanceNum());
            // printf("masks_num:%d\n",maskrcnnResults.size());
            int masks_num = maskrcnnResults.size();
            maskSuperPixelFilter_OverSeg(1200, finalSPixel, maskrcnnResults.size(),maskrcnnResults);
            // visualMask(visual_img,"",maskrcnnResults);

            // std::cout<<"Instance Num: "<<fusion->GetInstanceNum()<<std::endl;
            setmaskrcnnMask(maskrcnnResults);
            copyMaskToGpu(masks_num);

            /* --- 3. PREVIOUS: Integrate into Fusion --- */
            if (dr_timing) 
                id_time = dr_timer->start_timing("IntegrateScanAsync");

            fusion->IntegrateScanWithSemanticAsync(bgrs_previous[corrected_ref_index_previous].data, 
                                                                                                        output_previous->depth,
                                                                                                        maskrcnnMask_gpu, 
                                                                                                        masks_num,
                                                                                                        (float *) cam_to_worlds_previous[corrected_ref_index_previous].data) ;
            // std::cout<<"integrate pos :"<<cam_to_worlds_previous[corrected_ref_index_previous]<<std::endl;
            // fusion->IntegrateScanAsync(bgrs_previous[corrected_ref_index_previous].data, output_previous->depth, (float *) cam_to_worlds_previous[corrected_ref_index_previous].data);
            
            if (dr_timing) 
                dr_timer->end_timing("IntegrateScanAsync", id_time, !setting_debugout_runquiet);
            /* --- 4. PREVIOUS: RenderAsync for coarse tracker --- */
            std::vector<float const *> poses_to_render_previous;
            if (dense_tracking) 
                poses_to_render_previous.push_back(coarse_tracker_depth_map_use_next->cam_to_world);
            
            fusion->RenderWithMaskAsync(poses_to_render_previous);
            // fusion->RenderAsync(poses_to_render_previous);

            /* --- 5. PREVIOUS: Get render result --- */
            std::vector<unsigned char *> bgr_rendered_previous;
            std::vector<unsigned char *> instance_bgr_rendered_previous;
            std::vector<float *> depth_rendered_previous;
            
            fusion->GetRenderWithMaskResult(bgr_rendered_previous, instance_bgr_rendered_previous, depth_rendered_previous,InstanceMasks);
            // fusion->GetRenderResult(bgr_rendered_previous, depth_rendered_previous);
            

            ExtractInstanceBBox(InstanceMasks,renderInstanceBBoxResults);
            render_instance_num = renderInstanceBBoxResults.size();
            render_success = false;
            // if(last_render_instance_num-render_instance_num <=1 )
            // {
                last_render_instance_num = render_instance_num;
                // visualRender(bgr_rendered_previous[0],instance_bgr_rendered_previous[0],InstanceMasks,renderInstanceBBoxResults);
                render_success = true;
            // }
            // else
            // {
            //     std::cout<<"render fail"<<std::endl;
            //     last_render_instance_num = 0;
            // }
            InstanceMasks.clear();

            if (dense_tracking) 
            {
                memcpy(coarse_tracker_depth_map_use_next->depth, depth_rendered_previous[0], sizeof(float) * width * height);
                coarse_tracker_depth_map_use_next->is_valid = true; // atomic

                /* --- 5.5 PREVIOUS: Set Coarse Tracker --- */
                {
                    boost::unique_lock<boost::mutex> lock_coarse_tracker(mut_coarse_tracker);

                    // Ternary will only be false on first iter
                    TandemCoarseTrackingDepthMap *tmp = coarse_tracker_depth_map_valid ? coarse_tracker_depth_map_valid : coarse_tracker_depth_map_B;
                    coarse_tracker_depth_map_valid = coarse_tracker_depth_map_use_next;
                    coarse_tracker_depth_map_use_next = tmp;
                }
            }

            /* --- 6. PREVIOUS: Get mesh and push to output_previous wrapper --- */
            if (get_mesh && (call_num % mesh_extraction_freq) == 0) 
            {
                if (dr_timing) 
                    id_time = dr_timer->start_timing("fusion-mesh");
                fusion->ExtractMeshWithInstanceAsync(mesh_lower_corner, mesh_upper_corner, true);
                // fusion->ExtractMeshAsync(mesh_lower_corner, mesh_upper_corner);

                fusion->GetMeshSync();
                
                if (dr_timing) 
                    dr_timer->end_timing("fusion-mesh", id_time, !setting_debugout_runquiet);

                for (dso::IOWrap::Output3DWrapper *ow : outputWrapper) 
                    ow->pushDrMesh(fusion->dr_mesh_num, fusion->dr_mesh_vert, fusion->dr_mesh_cols);

                has_to_wait_previous = false;
            }
            delete output_previous;
        }

        // Now we swap *_previous <- *_current
        view_num_previous = view_num_current;
        index_offset_previous = index_offset_current;
        corrected_ref_index_previous = corrected_ref_index_current;
        bgrs_previous = bgrs_current;
        intrinsic_matrix_previous = intrinsic_matrix_current;
        cam_to_worlds_previous = cam_to_worlds_current;
        depth_min_previous = depth_min_current;
        depth_max_previous = depth_max_current;
        has_to_wait_previous = has_to_wait_current;
        unprocessed_data = false;

    }
    else
    {
        render_success = true;
    }
  
}

void TandemBackendImpl::CallAsync(
    int view_num_in,
    int index_offset_in,
    int corrected_ref_index_in,
    std::vector<cv::Mat> const &bgrs_in,
    cv::Mat const &intrinsic_matrix_in,
    std::vector<cv::Mat> const &cam_to_worlds_in,
    float depth_min_in,
    float depth_max_in,
    cv::Mat const &coarse_tracker_pose_in
) {
  using std::cerr;
  using std::cout;
  using std::endl;

  if (unprocessed_data) {
    cerr << "Wrong Call Order in TANDEM Backend. Will just return." << endl;
    return;
  }

  {
    boost::unique_lock<boost::mutex> lock(mut);
    {
      // Now we have the lock
      // We will process the MVSNet result for the *_previous data
      // The Loop will finish the processing of the *_previous data
      // The end of Loop will switch *_previous <- *_current

      /* --- 0. Copy input Data --- */
      view_num_current = view_num_in;
      index_offset_current = index_offset_in;
      corrected_ref_index_current = corrected_ref_index_in;
      bgrs_current = bgrs_in;
      intrinsic_matrix_current = intrinsic_matrix_in;
      cam_to_worlds_current = cam_to_worlds_in;
      depth_min_current = depth_min_in;
      depth_max_current = depth_max_in;
      if (dense_tracking) {
        coarse_tracker_depth_map_use_next->is_valid = false; // atomic
        memcpy(coarse_tracker_depth_map_use_next->cam_to_world, coarse_tracker_pose_in.data, sizeof(float) * 16);
      }

      if (has_to_wait_previous) {
        /* --- 1. PREVIOUS: Get MVSNet result --- */
        if (!mvsnet->Ready()) {
          std::cerr << "MVSNET IS NOT READY!!! WHY" << std::endl;
          exit(EXIT_FAILURE);
        }
        output_previous = mvsnet->GetResult();

        /* --- 2. PREVIOUS: Push depth map to output_previous wrapper --- */
        const float depth_max_value_previous = *std::max_element(output_previous->depth, output_previous->depth + width * height);
        for (dso::IOWrap::Output3DWrapper *ow : outputWrapper) {
          ow->pushDrKfImage(bgrs_previous[corrected_ref_index_previous].data);
          ow->pushDrKfDepth(output_previous->depth_dense, depth_min_previous, depth_max_value_previous);
        }
      }

    }

    unprocessed_data = true;
    newInputSignal.notify_all();
  }
}

bool TandemBackendImpl::Ready() {
  return !unprocessed_data && mvsnet->Ready();
}

TandemCoarseTrackingDepthMap::TandemCoarseTrackingDepthMap(int width, int height) {
  depth = (float *) malloc(sizeof(float) * width * height);
}

TandemCoarseTrackingDepthMap::~TandemCoarseTrackingDepthMap() {
  free(depth);
}

TandemBackend::TandemBackend(int width, int height, bool dense_tracking,
                             DrMvsnet *mvsnet, DrFusion *fusion,
                             float mvsnet_discard_percentage,
                             Timer *dr_timer,
                             std::vector<dso::IOWrap::Output3DWrapper *> const &outputWrapper,
                             int mesh_extraction_freq,
                             SampleMaskRCNN* maskrcnn,
                             gSLICrTools*  slicr
                             ) {
  impl = new TandemBackendImpl(width, height, dense_tracking, mvsnet, fusion, mvsnet_discard_percentage, dr_timer, outputWrapper, mesh_extraction_freq,
                                                                        maskrcnn,slicr);
}

TandemBackend::~TandemBackend() {
  delete impl;
}

bool TandemBackend::Ready() {
  return impl->Ready();
}

void TandemBackend::CallAsync(int view_num_in, int index_offset_in, int corrected_ref_index_in, const std::vector<cv::Mat> &bgrs_in, const cv::Mat &intrinsic_matrix_in, const std::vector<cv::Mat> &cam_to_worlds_in, float depth_min_in,
                              float depth_max_in, const cv::Mat &coarse_tracker_pose_in) {
  impl->CallAsync(
      view_num_in,
      index_offset_in,
      corrected_ref_index_in,
      bgrs_in,
      intrinsic_matrix_in,
      cam_to_worlds_in,
      depth_min_in,
      depth_max_in,
      coarse_tracker_pose_in
  );

}

boost::mutex &TandemBackend::GetTrackingDepthMapMutex() {
  return impl->GetTrackingDepthMapMutex();
}

TandemCoarseTrackingDepthMap const *TandemBackend::GetTrackingDepthMap() {
  return impl->GetTrackingDepthMap();
}

void TandemBackend::Wait() {
  impl->Wait();
}

void TandemBackendImpl::Wait() {
  boost::unique_lock<boost::mutex> lock(mut);
  while (unprocessed_data) {
    dataProcessedSignal.wait(lock);
  }
  mvsnet->Wait();
  if (!Ready()) {
    std::cerr << "TandemBackendImpl must be Ready() after Wait(). Something went wrong." << std::endl;
    exit(EXIT_FAILURE);
  }
}

float get_idepth_quantile(int n, const float *const idepth, float fraction) {
  std::vector<float> idepth_sorted(idepth, idepth + n);
  const int quantile_n = (int) (fraction * (float) n);
  auto m = idepth_sorted.begin() + quantile_n;
  std::nth_element(idepth_sorted.begin(), m, idepth_sorted.end());
  const float idepth_quantile = idepth_sorted[quantile_n];
  return 1.0f / idepth_quantile;
}
