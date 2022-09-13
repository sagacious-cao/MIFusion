
#include "maskrcnn.h"
// const std::string gSampleName = "TensorRT.sample_maskrcnn";
int main(int argc, char** argv)
{
    
    MaskRCNNParams params;
    params.dataDirs.push_back("data/maskrcnn/");
    params.inputTensorNames.push_back(MaskRCNNConfig::MODEL_INPUT);
    params.batchSize = 1;
    params.outputTensorNames.push_back(MaskRCNNConfig::MODEL_OUTPUTS[0]);
    params.outputTensorNames.push_back(MaskRCNNConfig::MODEL_OUTPUTS[1]);
    params.dlaCore = -1;
    params.int8 = false;
    params.fp16 = true;
    params.uffFileName = MaskRCNNConfig::MODEL_NAME;
    params.maskThreshold = MaskRCNNConfig::MASK_THRESHOLD;

    SampleMaskRCNN sample(params);

    // sample::gLogInfo << "Building and running a GPU inference engine for Mask RCNN" << std::endl;

    if (!sample.build("eigen.bin",false))
    {
        // return sample::gLogger.reportFail(sampleTest);
    }
    cv::Mat rgbImage = cv::imread ( "data/maskrcnn/001763.ppm" );
    if (!sample.infer(rgbImage))
    {
        // return sample::gLogger.reportFail(sampleTest);
    }
    if (!sample.teardown())
    {
        // return sample::gLogger.reportFail(sampleTest);
    }

    // return sample::gLogger.reportPass(sampleTest);
}
