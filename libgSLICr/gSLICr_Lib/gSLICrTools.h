#ifndef GSLICRTOOLS_H
#define GSLICRTOOLS_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include "gSLICr.h"
#include "segmentationCuda.h"
#include <Eigen/Core>

class gSLICrTools 
{
    public:
        gSLICrTools(int w,int h)
        {
            width = w;
            height = h;
            my_settings.img_size.x = width;
            my_settings.img_size.y = height;
            my_settings.no_segs = 2000;
            my_settings.spixel_size = 16;
            my_settings.coh_weight = 0.6f;
            my_settings.no_iters = 5;
            my_settings.color_space = gSLICr::XYZ; // gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB
            my_settings.seg_method = gSLICr::GIVEN_SIZE; // or gSLICr::GIVEN_NUM for given number
            my_settings.do_enforce_connectivity = true; // whether or not run the enforce connectivity step

            gSLICr_engine = new gSLICr::engines::core_engine(my_settings);
	        // in_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);
	        // out_img = new gSLICr::UChar4Image(my_settings.img_size, true, true);


        }
        void initCameraIntrinsics(float cx,float cy,float fx,float fy)
        {
            cam = Eigen::Vector4f(cx, cy,1.0 / fx, 1.0 / fy);
        }
        void rgbdSuperPixelSeg(cv::Mat rgb,cv::Mat depth,int* gSLICrMask,cv::Mat& visual_img);
        void imageCV2SLIC(const cv::Mat& inimg, gSLICr::UChar4Image* outimg);
        void imageSLIC2CV(const gSLICr::UChar4Image* inimg, cv::Mat& outimg);
        void gSLICrInterface(cv::Mat inputrgb,int* gSLICrMask,cv::Mat& boundry_draw_frame);

        // 利用Depth将RGB超像素分割的结果合并为更大的超像素
        void connectSuperPixel(float* spInfo);
        void mergeSuperPixel(unsigned short* depthMap, int* segMask, int *finalSPixel);

        uchar* matToUchar(cv::Mat img);
        uchar* matToUchar_oneChanel(cv::Mat img);
        ushort* matToUshort_oneChanel(cv::Mat img);

        void load_image(const gSLICr::UChar4Image* inimg, cv::Mat& outimg);

        int getResultSuperpixelNum()
        {
            return spNum;
        }

        gSLICr::objects::settings my_settings;
	    gSLICr::engines::core_engine* gSLICr_engine;
        int spNum = 1200;  // 超像素的个数

        //spInfo 是针对超像素块的
        //0 pixel_num   123 pos_sum   456 nor_sum 789 pos_avg   10-12 nor_avg   13 depth_sum 14 depth_avg   
        //15 distance_stand_deviation 16 num_after_cluster 17 connectNum   18-28 neighbor 29 finalID
        const int SPI_SIZE = 30;
        const int SPI_PNUM = 0;

        const int SPI_POS_SX = 1;
        const int SPI_POS_SY = 2;
        const int SPI_POS_SZ = 3;
        const int SPI_NOR_SX = 4;
        const int SPI_NOR_SY = 5;
        const int SPI_NOR_SZ = 6;

        const int SPI_POS_AX = 7;
        const int SPI_POS_AY = 8;
        const int SPI_POS_AZ = 9;
        const int SPI_NOR_AX = 10;
        const int SPI_NOR_AY = 11;
        const int SPI_NOR_AZ = 12;

        const int SPI_DEPTH_SUM = 13;
        const int SPI_DEPTH_AVG = 14;

        const int SPI_DIST_DEV = 15;
        const int SPI_NOR_DEV = 16;

        const int SPI_CONNECT_N = 17;
        const int SPI_NP_FIRST = 18;
        const int SPI_NP_MAX = 11;	//18+11=29

        const int SPI_FINAL = 29;

        //iterations in depthMap cluster
        //const int iterations = 1;

        int height = 480;
        int width = 640;
        Eigen::Vector4f cam;

};


#endif