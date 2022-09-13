#ifndef MASKRCNN_H
#define MASKRCNN_H
// #include "main.h"
#include <assert.h>
#include <chrono>
#include <ctime>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>
#include <vector>

#include "NvInfer.h"
#include "NvUffParser.h"

#include "buffers.h"
#include "common.h"
#include "logger.h"

#include <opencv2/opencv.hpp>

// max
#include <algorithm>

// MaskRCNN Parameter
#include "mrcnn_config.h"

struct MaskRCNNParams
{
    int batchSize{1};                  //!< Number of inputs in a batch
    int dlaCore{-1};                   //!< Specify the DLA core to run network on.
    bool int8{false};                  //!< Allow runnning the network in Int8 mode.
    bool fp16{false};                  //!< Allow running the network in FP16 mode.
    std::vector<std::string> dataDirs; //!< Directory paths where sample data files are stored
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;

    std::string uffFileName;
    float maskThreshold;
};


namespace MaskRCNNUtils
{
    struct RawDetection
    {
        float y1, x1, y2, x2, class_id, score;
    };

    struct Mask
    {
        float raw[MaskRCNNConfig::MASK_POOL_SIZE * 2 * MaskRCNNConfig::MASK_POOL_SIZE * 2];
    };

    struct BBox
    {
        float x1, y1, x2, y2;
    };

    struct BBoxInfo
    {
        BBox box;
        int label = -1;
        float prob = 0.0f;
        int matchIndex = -1;
        float IoU = 0;
        Mask* mask = nullptr;
    };

    template <typename T>
    struct PPM
    {
        std::string magic, fileName;
        int h, w, max;
        std::vector<T> buffer;
    };

    inline void setAllTensorScales(nvinfer1::INetworkDefinition* network, float inScales = 2.0f, float outScales = 4.0f);
    inline int64_t volume(const nvinfer1::Dims& d);
    void readPPMFile(const std::string& filename, PPM<uint8_t>& ppm);
    void writePPMFile(const std::string& filename, PPM<uint8_t>& ppm);
    template <typename T>
    void resizePPM(const PPM<T>& src, PPM<T>& dst, int target_height, int target_width, int channel);
    void padPPM(const PPM<uint8_t>& src, PPM<uint8_t>& dst, int top, int bottom, int left, int right);
    void preprocessPPM(PPM<uint8_t>& src, PPM<uint8_t>& dst, int target_h, int target_w);
    PPM<uint8_t> resizeMask(const BBoxInfo& box, const float mask_threshold);
    void maskPPM(PPM<uint8_t>& image, const PPM<uint8_t>& mask, const int start_x, const int start_y, const std::vector<int>& color);
    void addBBoxPPM(PPM<uint8_t>& ppm, const BBoxInfo& box, const PPM<uint8_t>& resized_mask);
    void readImg(const std::string& filename, cv::Mat& img);
    void resizeImg(cv::Mat& img,int target_heigh,int target_width);
    void padImg(cv::Mat& img,int top, int bottom, int left, int right);
    void writeImg(const std::string& filename, cv::Mat& img);
} // namespace MaskRCNNUtils







class SampleMaskRCNN
{

    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, samplesCommon::InferDeleter>;

public:

    SampleMaskRCNN(){}
    void init(const MaskRCNNParams& params)
    {
        mParams = params;
        mEngine = nullptr;
        srand((int) time(0));
    }

    SampleMaskRCNN(const MaskRCNNParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
        srand((int) time(0));
    }

    bool build(std::string serializedEngine,bool isload);

    bool infer(cv::Mat  inputimg);
    bool teardown();
    int getMaskNum()
    {
        return results.size();
    }
    std::vector<std::pair<cv::Mat,MaskRCNNUtils::BBoxInfo>> results;

private:
    MaskRCNNParams mParams;

    nvinfer1::Dims mInputDims;

    // original images
    std::vector<MaskRCNNUtils::PPM<uint8_t>> mOriginalPPMs;

    // processed images (resize + pad)
    std::vector<MaskRCNNUtils::PPM<uint8_t>> mPPMs;

    

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;

    bool verifyOutput(const samplesCommon::BufferManager& buffers);
    std::vector<MaskRCNNUtils::BBoxInfo> decodeOutput(void* detectionsHost, void* masksHost);
    void resizeImg(cv::Mat& src,cv::Mat& dst,int target_heigh,int target_width);
    void padImg(cv::Mat& src,cv::Mat& dst,int top, int bottom, int left, int right);
    void preprocessImg(cv::Mat& src, cv::Mat& dst, int target_h, int target_w);
    cv::Mat resizeMask_mat(const MaskRCNNUtils::BBoxInfo& box, const float mask_threshold,cv::Size size);
    static bool sortMask(const std::pair<cv::Mat,MaskRCNNUtils::BBoxInfo>& mask1, const std::pair<cv::Mat,MaskRCNNUtils::BBoxInfo>& mask2);
    cv::Mat originalimg;
    cv::Mat resizedimg;

    SampleUniquePtr<nvinfer1::IExecutionContext> context;
};



#endif
