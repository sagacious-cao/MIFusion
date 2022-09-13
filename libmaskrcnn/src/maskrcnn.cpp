#include "maskrcnn.h"

namespace MaskRCNNUtils
{
    inline void setAllTensorScales(nvinfer1::INetworkDefinition* network, float inScales, float outScales)
    {
        // Ensure that all layer inputs have a scale.
        for (int i = 0; i < network->getNbLayers(); i++)
        {
            auto layer = network->getLayer(i);
            for (int j = 0; j < layer->getNbInputs(); j++)
            {
                ITensor* input{layer->getInput(j)};
                // Optional inputs are nullptr here and are from RNN layers.
                if (input != nullptr && !input->dynamicRangeIsSet())
                {
                    ASSERT(input->setDynamicRange(-inScales, inScales));
                }
            }
        }



        // Ensure that all layer outputs have a scale.
        // Tensors that are also inputs to layers are ingored here
        // since the previous loop nest assigned scales to them.
        for (int i = 0; i < network->getNbLayers(); i++)
        {
            auto layer = network->getLayer(i);
            for (int j = 0; j < layer->getNbOutputs(); j++)
            {
                nvinfer1::ITensor* output{layer->getOutput(j)};
                // Optional outputs are nullptr here and are from RNN layers.
                if (output != nullptr && !output->dynamicRangeIsSet())
                {
                    // Pooling must have the same input and output scales.
                    if (layer->getType() == nvinfer1::LayerType::kPOOLING)
                    {
                        ASSERT(output->setDynamicRange(-inScales, inScales));
                    }
                    else
                    {
                        ASSERT(output->setDynamicRange(-outScales, outScales));
                    }
                }
            }
        }
    }

    inline int64_t volume(const nvinfer1::Dims& d)
    {
        return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
    }

    void readPPMFile(const std::string& filename, PPM<uint8_t>& ppm)
    {
        ppm.fileName = filename;
        std::ifstream infile(filename, std::ifstream::binary);
        assert(infile.is_open() && "Attempting to read from a file that is not open. ");
        infile >> ppm.magic >> ppm.w >> ppm.h >> ppm.max;
        infile.seekg(1, infile.cur);

        ppm.buffer.resize(ppm.w * ppm.h * 3, 0);

        infile.read(reinterpret_cast<char*>(ppm.buffer.data()), ppm.w * ppm.h * 3);
    }

    void writePPMFile(const std::string& filename, PPM<uint8_t>& ppm)
    {
        std::ofstream outfile("./" + filename, std::ofstream::binary);
        assert(!outfile.fail());
        outfile << "P6"
                << "\n"
                << ppm.w << " " << ppm.h << "\n"
                << ppm.max << "\n";

        outfile.write(reinterpret_cast<char*>(ppm.buffer.data()), ppm.w * ppm.h * 3);
    }

    template <typename T>
    void resizePPM(const PPM<T>& src, PPM<T>& dst, int target_height, int target_width, int channel)
    {
        auto clip = [](float in, float low, float high) -> float { return (in < low) ? low : (in > high ? high : in); };
        int original_height = src.h;
        int original_width = src.w;
        assert(dst.h == target_height);
        assert(dst.w == target_width);
        float ratio_h = static_cast<float>(original_height - 1.0f) / static_cast<float>(target_height - 1.0f);
        float ratio_w = static_cast<float>(original_width - 1.0f) / static_cast<float>(target_width - 1.0f);

        int dst_idx = 0;
        for (int y = 0; y < target_height; ++y)
        {
            for (int x = 0; x < target_width; ++x)
            {
                float x0 = static_cast<float>(x) * ratio_w;
                float y0 = static_cast<float>(y) * ratio_h;
                int left = static_cast<int>(clip(std::floor(x0), 0.0f, static_cast<float>(original_width - 1.0f)));
                int top = static_cast<int>(clip(std::floor(y0), 0.0f, static_cast<float>(original_height - 1.0f)));
                int right = static_cast<int>(clip(std::ceil(x0), 0.0f, static_cast<float>(original_width - 1.0f)));
                int bottom = static_cast<int>(clip(std::ceil(y0), 0.0f, static_cast<float>(original_height - 1.0f)));

                for (int c = 0; c < channel; ++c)
                {
                    // H, W, C ordering
                    T left_top_val = src.buffer[top * (original_width * channel) + left * (channel) + c];
                    T right_top_val = src.buffer[top * (original_width * channel) + right * (channel) + c];
                    T left_bottom_val = src.buffer[bottom * (original_width * channel) + left * (channel) + c];
                    T right_bottom_val = src.buffer[bottom * (original_width * channel) + right * (channel) + c];
                    float top_lerp = left_top_val + (right_top_val - left_top_val) * (x0 - left);
                    float bottom_lerp = left_bottom_val + (right_bottom_val - left_bottom_val) * (x0 - left);
                    float lerp = clip(std::round(top_lerp + (bottom_lerp - top_lerp) * (y0 - top)), 0.0f, 255.0f);
                    dst.buffer[dst_idx] = (static_cast<T>(lerp));
                    dst_idx++;
                }
            }
        }
    }

    void padPPM(const PPM<uint8_t>& src, PPM<uint8_t>& dst, int top, int bottom, int left, int right)
    {
        assert(dst.h == (src.h + top + bottom));
        assert(dst.w == (src.w + left + right));

        for (int y = 0; y < src.h; y++)
        {
            for (int x = 0; x < src.w; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    dst.buffer[(top + y) * dst.w * 3 + (left + x) * 3 + c] = src.buffer[y * src.w * 3 + x * 3 + c];
                }
            }
        }
    }

    void preprocessPPM(PPM<uint8_t>& src, PPM<uint8_t>& dst, int target_h, int target_w)
    {
        std::cout<<target_h<<"  "<<target_w<<std::endl;
        assert(target_h == target_w);
        int input_dim = target_h;
        // padding the input img to model's input_size:
        const int image_dim = std::max(src.h, src.w);
        int resize_h = src.h * input_dim / image_dim;
        int resize_w = src.w * input_dim / image_dim;
        assert(resize_h == input_dim || resize_w == input_dim);

        int y_offset = (input_dim - resize_h) / 2;
        int x_offset = (input_dim - resize_w) / 2;

        // resize
        PPM<uint8_t> resized_ppm;
        resized_ppm.h = resize_h;
        resized_ppm.w = resize_w;
        resized_ppm.max = src.max;
        resized_ppm.buffer.resize(resize_h * resize_w * 3, 0);
        resizePPM<uint8_t>(src, resized_ppm, resize_h, resize_w, 3);

        // pad
        dst.h = target_h;
        dst.w = target_w;
        dst.max = src.max;
        dst.buffer.resize(dst.h * dst.w * 3, 0);
        padPPM(resized_ppm, dst, y_offset, input_dim - resize_h - y_offset, x_offset, input_dim - resize_w - x_offset);
    }

    PPM<uint8_t> resizeMask(const BBoxInfo& box, const float mask_threshold)
    {
        PPM<uint8_t> result;
        if (!box.mask)
        {
            assert(result.buffer.size() == 0);
            return result;
        }

        const int h = box.box.y2 - box.box.y1;
        const int w = box.box.x2 - box.box.x1;

        PPM<float> raw_mask;
        raw_mask.h = MaskRCNNConfig::MASK_POOL_SIZE * 2;
        raw_mask.w = MaskRCNNConfig::MASK_POOL_SIZE * 2;
        raw_mask.buffer.resize(raw_mask.h * raw_mask.w, 0);
        raw_mask.max = std::numeric_limits<int>::lowest();
        for (int i = 0; i < raw_mask.h * raw_mask.w; i++)
            raw_mask.buffer[i] = box.mask->raw[i];

        PPM<float> resized_mask;
        resized_mask.h = h;
        resized_mask.w = w;
        resized_mask.buffer.resize(h * w, 0);
        resizePPM<float>(raw_mask, resized_mask, h, w, 1);

        result.h = h;
        result.w = w;
        result.buffer.resize(result.h * result.w, 0);
        for (int i = 0; i < h * w; i++)
        {
            if (resized_mask.buffer[i] > mask_threshold)
            {
                result.buffer[i] = 1;
            }
        }

        return result;
    }

    void maskPPM(PPM<uint8_t>& image, const PPM<uint8_t>& mask, const int start_x, const int start_y, const std::vector<int>& color)
    {

        float alpha = 0.6f;

        for (int y = 0; y < mask.h; ++y)
        {
            for (int x = 0; x < mask.w; ++x)
            {
                uint8_t mask_pixel = mask.buffer[y * mask.w + x];
                if (mask_pixel == 1)
                {
                    assert(0 <= start_y + y && start_y + y < image.h);
                    assert(0 <= start_x + x && start_x + x < image.w);

                    int cur_y = start_y + y;
                    int cur_x = start_x + x;

                    float p_r = static_cast<float>(image.buffer[(cur_y * image.w + cur_x) * 3]);
                    float p_g = static_cast<float>(image.buffer[(cur_y * image.w + cur_x) * 3 + 1]);
                    float p_b = static_cast<float>(image.buffer[(cur_y * image.w + cur_x) * 3 + 2]);

                    image.buffer[(cur_y * image.w + cur_x) * 3]
                        = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, p_r * (1 - alpha) + color[0] * alpha)));
                    image.buffer[(cur_y * image.w + cur_x) * 3 + 1]
                        = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, p_g * (1 - alpha) + color[1] * alpha)));
                    image.buffer[(cur_y * image.w + cur_x) * 3 + 2]
                        = static_cast<uint8_t>(std::max(0.0f, std::min(255.0f, p_b * (1 - alpha) + color[2] * alpha)));
                }
                else
                    assert(mask_pixel == 0);
            }
        }
    }
    void addBBoxPPM(PPM<uint8_t>& ppm, const BBoxInfo& box, const PPM<uint8_t>& resized_mask)
    {
        const int x1 = box.box.x1;
        const int y1 = box.box.y1;
        const int x2 = box.box.x2;
        const int y2 = box.box.y2;
        std::vector<int> color = {rand() % 256, rand() % 256, rand() % 256};

        for (int x = x1; x <= x2; x++)
        {
            // bbox top border
            ppm.buffer[(y1 * ppm.w + x) * 3] = color[0];
            ppm.buffer[(y1 * ppm.w + x) * 3 + 1] = color[1];
            ppm.buffer[(y1 * ppm.w + x) * 3 + 2] = color[2];
            // bbox bottom border
            ppm.buffer[(y2 * ppm.w + x) * 3] = color[0];
            ppm.buffer[(y2 * ppm.w + x) * 3 + 1] = color[1];
            ppm.buffer[(y2 * ppm.w + x) * 3 + 2] = color[2];
        }

        for (int y = y1; y <= y2; y++)
        {
            // bbox left border
            ppm.buffer[(y * ppm.w + x1) * 3] = color[0];
            ppm.buffer[(y * ppm.w + x1) * 3 + 1] = color[1];
            ppm.buffer[(y * ppm.w + x1) * 3 + 2] = color[2];
            // bbox right border
            ppm.buffer[(y * ppm.w + x2) * 3] = color[0];
            ppm.buffer[(y * ppm.w + x2) * 3 + 1] = color[1];
            ppm.buffer[(y * ppm.w + x2) * 3 + 2] = color[2];
        }

        if (resized_mask.buffer.size() != 0)
        {
            maskPPM(ppm, resized_mask, x1, y1, color);
        }
    }


    void readImg(const std::string& filename, cv::Mat& img)
    {
        img = cv::imread(filename.c_str());
        cv::cvtColor(img, img, CV_RGB2BGR);
    }

    void resizeImg(cv::Mat& img,int target_heigh,int target_width)
    {
        cv::resize(img,img,cv::Size(target_width,target_heigh),CV_BILATERAL);
    }

    void padImg(cv::Mat& img,int top, int bottom, int left, int right)
    {
        cv::copyMakeBorder(img,img,top,bottom,left,right,cv::BORDER_CONSTANT);
    }

    void writeImg(const std::string& filename, cv::Mat& img)
    {
        cv::cvtColor(img,img,CV_RGB2BGR);
        cv::imwrite(filename.c_str(),img);
    }

} // namespace MaskRCNNUtils

bool SampleMaskRCNN::build(std::string serializedEngine,bool isload)
{
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
    if(!isload)
    {
        auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
        if (!builder)
        {
            return false;
        }

        auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
        if (!network)
        {
            return false;
        }

        auto parser = SampleUniquePtr<nvuffparser::IUffParser>(nvuffparser::createUffParser());
        if (!parser)
        {
            return false;
        }

        // auto constructed = constructNetwork(builder, network, parser);
        parser->registerInput(mParams.inputTensorNames[0].c_str(), MaskRCNNConfig::IMAGE_SHAPE, nvuffparser::UffInputOrder::kNCHW);
        for (size_t i = 0; i < mParams.outputTensorNames.size(); i++)
            parser->registerOutput(mParams.outputTensorNames[i].c_str());

        auto parsed = parser->parse(locateFile(mParams.uffFileName, mParams.dataDirs).c_str(), *network, DataType::kFLOAT);
        if (!parsed)
        {
            return false;
        }

        SampleUniquePtr<IBuilderConfig> config{builder->createBuilderConfig()};

        builder->setMaxBatchSize(mParams.batchSize);
        config->setMaxWorkspaceSize(8_GiB);
        if (mParams.fp16)
        {
            config->setFlag(BuilderFlag::kFP16);
        }

        // Only for speed test
        if (mParams.int8)
        {
            samplesCommon::setAllDynamicRanges(network.get());
            config->setFlag(BuilderFlag::kINT8);
        }

        // CUDA stream used for profiling by the builder.
        auto profileStream = samplesCommon::makeCudaStream();
        if (!profileStream)
        {
            return false;
        }
        config->setProfileStream(*profileStream);

        SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
        if (!plan)
        {
            return false;
        }

        SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
        if (!runtime)
        {
            return false;
        }

        std::string serialize_str;
        serialize_str.resize( plan->size() );
        std::cout << plan->size()<<std::endl;
        memcpy((void*)serialize_str.data(), plan->data(), plan->size());
        std::ofstream serialize_output_stream(serializedEngine,std::ios_base::out | std::ios_base::binary);
        serialize_output_stream << serialize_str;
        serialize_output_stream.close();

        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
        if (!mEngine)
        {
            return false;
        }

        ASSERT(network->getNbInputs() == 1);
        mInputDims = network->getInput(0)->getDimensions();
        ASSERT(mInputDims.nbDims == 3);
        ASSERT(network->getNbOutputs() == 2);
    }
    else
    {
        //读取engine
        std::cout<<"start build eigen"<<std::endl;
        auto tStart = std::chrono::high_resolution_clock::now();
        
        std::cout<<serializedEngine<<std::endl;
        std::ifstream fin(serializedEngine,std::ios_base::in | std::ios_base::binary);
        // 将文件中的内容读取至cached_engine字符串
        std::string cached_engine = "";
        while (fin.peek() != EOF)
        { // 使用fin.peek()防止文件读取时无限循环
            std::stringstream buffer;
            buffer << fin.rdbuf();
            cached_engine.append(buffer.str());
        }
        fin.close();

        // 将序列化得到的结果进行反序列化，以执行后续的inference
        // IRuntime* runtime = createInferRuntime(sample::gLogger.getTRTLogger());
        IRuntime* runtime = createInferRuntime(sample::gLogger.getTRTLogger());
        std::cout<<"size:"<< cached_engine.size()<<std::endl;
        // mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(plan->data(),                    plan->size()),                                      samplesCommon::InferDeleter());
        mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(cached_engine.data(), cached_engine.size()), samplesCommon::InferDeleter());
        auto tEnd = std::chrono::high_resolution_clock::now();
        float totalHost = std::chrono::duration<float, std::milli>(tEnd - tStart).count();
        std::cout << "build eigen time is " << (totalHost) << "ms"<<std::endl;
        if (!mEngine)
        {
            return false;
        }
    }
    context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    return true;
}

bool SampleMaskRCNN::infer(cv::Mat  inputimg)
{
    inputimg.copyTo(originalimg);
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine, mParams.batchSize);
    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    const int inputC = MaskRCNNConfig::IMAGE_SHAPE.d[0];
    const int inputH = MaskRCNNConfig::IMAGE_SHAPE.d[1];
    const int inputW = MaskRCNNConfig::IMAGE_SHAPE.d[2];
    preprocessImg(inputimg,resizedimg,inputH, inputW);
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    float pixelMean[3]{123.7, 116.8, 103.9};    
    for (int c = 0; c < inputC; ++c)
        // The color image to input should be in RGB order
        for (unsigned j = 0, volChl = inputH * inputW; j < volChl; ++j)
            hostDataBuffer[c * volChl + j] = float(resizedimg.data[j * inputC + c]) - pixelMean[c];

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();
    bool status;
    auto tStart = std::chrono::high_resolution_clock::now();
    status = context->execute(mParams.batchSize, buffers.getDeviceBindings().data());
    auto tEnd = std::chrono::high_resolution_clock::now();
    float totalHost = std::chrono::duration<float, std::milli>(tEnd - tStart).count();
    // std::cout << "语义分割速度:" << totalHost<< " ms/frame" << std::endl;
    if (!status)
    {
        return false;
    }
    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();
    // Post-process detections and verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }
    
    bool debugmask = false;
    if(debugmask)
    {
        if(results.size()>0)
        {
            static int masknum = 0;
            cv::Mat contours;
            cv::Mat visual_img;
            originalimg.copyTo(visual_img);
            cv::Mat red = cv::Mat(visual_img.size(),CV_8UC3,cv::Scalar(0,0,255));
            cv::Mat mask = cv::Mat(visual_img.size(),CV_8UC1);
            for(int i = 0;i<results.size();i++)
            {   
                mask = results[i].first/results[i].second.label*255;
                std::cout<<visual_img.size()<<red.size()<<visual_img.size()<<mask.size()<<std::endl;
                cv::add(visual_img,red,visual_img,mask);
            }
            char imagename[50];
            sprintf(imagename,"test_maskrcnn/debugmask_%d.png",masknum);
            cv::imwrite(imagename,visual_img);
            masknum++;
        }

    }
    

    return true;
}

bool SampleMaskRCNN::teardown()
{
    //! Clean up the libprotobuf files as the parsing is complete
    //! \note It is not safe to use any other part of the protocol buffers library after
    //! ShutdownProtobufLibrary() has been called.
    nvuffparser::shutdownProtobufLibrary();
    return true;
}

std::vector<MaskRCNNUtils::BBoxInfo> SampleMaskRCNN::decodeOutput(void* detectionsHost, void* masksHost)
{
    int input_dim_h = MaskRCNNConfig::IMAGE_SHAPE.d[1], input_dim_w = MaskRCNNConfig::IMAGE_SHAPE.d[2];
    assert(input_dim_h == input_dim_w);
    int image_height = originalimg.rows;
    int image_width = originalimg.cols;
    // resize the DsImage with scale
    const int image_dim = std::max(image_height, image_width);
    int resizeH = (int) image_height * input_dim_h / (float) image_dim;
    int resizeW = (int) image_width * input_dim_w / (float) image_dim;
    // keep accurary from (float) to (int), then to float
    float window_x = (1.0f - (float) resizeW / input_dim_w) / 2.0f;
    float window_y = (1.0f - (float) resizeH / input_dim_h) / 2.0f;
    float window_width = (float) resizeW / input_dim_w;
    float window_height = (float) resizeH / input_dim_h;

    float final_ratio_x = (float) image_width / window_width;
    float final_ratio_y = (float) image_height / window_height;
    std::vector<MaskRCNNUtils::BBoxInfo> binfo;

    // int detectionOffset = MaskRCNNUtils::volume(MaskRCNNConfig::MODEL_DETECTION_SHAPE); // (100,6)
    // int maskOffset = MaskRCNNUtils::volume(MaskRCNNConfig::MODEL_MASK_SHAPE);           // (100, 81, 28, 28)

    MaskRCNNUtils::RawDetection* detections = reinterpret_cast<MaskRCNNUtils::RawDetection*>((float*) detectionsHost);
    MaskRCNNUtils::Mask* masks = reinterpret_cast<MaskRCNNUtils::Mask*>((float*) masksHost);
    for (int det_id = 0; det_id < MaskRCNNConfig::DETECTION_MAX_INSTANCES; det_id++)
    {
        MaskRCNNUtils::RawDetection cur_det = detections[det_id];
        int label = (int) cur_det.class_id;
        if (label <= 0)
            continue;

        MaskRCNNUtils::BBoxInfo det;
        det.label = label;
        det.prob = cur_det.score;

        det.box.x1 = std::min(std::max((cur_det.x1 - window_x) * final_ratio_x, 0.0f), (float) image_width);
        det.box.y1 = std::min(std::max((cur_det.y1 - window_y) * final_ratio_y, 0.0f), (float) image_height);
        det.box.x2 = std::min(std::max((cur_det.x2 - window_x) * final_ratio_x, 0.0f), (float) image_width);
        det.box.y2 = std::min(std::max((cur_det.y2 - window_y) * final_ratio_y, 0.0f), (float) image_height);

        if (det.box.x2 <= det.box.x1 || det.box.y2 <= det.box.y1)
            continue;

        det.mask = masks + det_id * MaskRCNNConfig::NUM_CLASSES + label;

        binfo.push_back(det);
    }

    return binfo;
}

cv::Mat SampleMaskRCNN::resizeMask_mat(const MaskRCNNUtils::BBoxInfo& box, const float mask_threshold,cv::Size size)
{
    cv::Mat result;
    if (!box.mask)
    {
        assert(result.empty() == 1);
        return result;
    }

    const int h = box.box.y2 - box.box.y1;
    const int w = box.box.x2 - box.box.x1;

    cv::Mat raw_mask = cv::Mat(MaskRCNNConfig::MASK_POOL_SIZE * 2,MaskRCNNConfig::MASK_POOL_SIZE * 2, CV_32FC1, box.mask->raw);
    cv::Mat resized_mask;
    cv::resize(raw_mask,resized_mask,cv::Size(w,h),CV_BILATERAL);
    cv::threshold(resized_mask, result, mask_threshold, 1, CV_THRESH_BINARY);
    result.convertTo(result,CV_8UC1);

    cv::Mat background = cv::Mat::zeros(size,CV_8U);
    cv::Rect rect(box.box.x1,box.box.y1,result.cols,result.rows);
    result.copyTo(background(rect));
    
    return background;
}

bool SampleMaskRCNN::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    void* detectionsHost = buffers.getHostBuffer(mParams.outputTensorNames[0]);
    void* masksHost = buffers.getHostBuffer(mParams.outputTensorNames[1]);
    results.clear();
    std::vector<MaskRCNNUtils::BBoxInfo> binfo = decodeOutput(detectionsHost, masksHost);
    // free(detectionsHost);
    // free(masksHost);
    // std::cout<<"maskrcnn obj:"<<binfo.size()<<std::endl;

    for (size_t roi_id = 0; roi_id < binfo.size(); roi_id++)
    {
        if(binfo[roi_id].prob>0.8)
        {
            const auto _mask = resizeMask_mat(binfo[roi_id], mParams.maskThreshold,originalimg.size()); // mask threshold
            cv::Mat resized_mask;
            cv::resize(_mask,resized_mask,originalimg.size(),CV_INTER_NN);
            // resized_mask = resized_mask*binfo[roi_id].label;
            resized_mask = resized_mask;
            int nonZerosNum = countNonZero(resized_mask);
            if(nonZerosNum*1.0/(originalimg.cols*originalimg.rows)<0.2)
            {
                results.push_back(std::pair<cv::Mat,MaskRCNNUtils::BBoxInfo>(resized_mask,binfo[roi_id]));
                // std::cout<<binfo[roi_id].prob<<"   "<<binfo[roi_id].label<<"   "<<MaskRCNNConfig::CLASS_NAMES[binfo[roi_id].label]<<std::endl;    
            } 
        }
    }

    std::sort(results.begin(),results.end(),sortMask);
    return true;
}

void SampleMaskRCNN::resizeImg(cv::Mat& src,cv::Mat& dst,int target_heigh,int target_width)
{
    cv::resize(src,dst,cv::Size(target_width,target_heigh),CV_BILATERAL);
}

void SampleMaskRCNN::padImg(cv::Mat& src,cv::Mat& dst,int top, int bottom, int left, int right)
{
    cv::copyMakeBorder(src,dst,top,bottom,left,right,cv::BORDER_CONSTANT);
}

void SampleMaskRCNN::preprocessImg(cv::Mat& src, cv::Mat& dst, int target_h, int target_w)
{
    assert(target_h == target_w);
    int input_dim = target_h;
    // padding the input img to model's input_size:
    const int image_dim = std::max(src.rows, src.cols);
    int resize_h = src.rows * input_dim / image_dim;
    int resize_w = src.cols * input_dim / image_dim;
    assert(resize_h == input_dim || resize_w == input_dim);

    int y_offset = (input_dim - resize_h) / 2;
    int x_offset = (input_dim - resize_w) / 2;
    cv::Mat resized_img;
    resizeImg(src,resized_img,resize_h,resize_w);
    padImg(resized_img,dst,y_offset, input_dim - resize_h - y_offset, x_offset, input_dim - resize_w - x_offset);
}

bool SampleMaskRCNN::sortMask(const std::pair<cv::Mat,MaskRCNNUtils::BBoxInfo> &mask1, const std::pair<cv::Mat,MaskRCNNUtils::BBoxInfo> &mask2)
{
    cv::Scalar mask1_sum = cv::sum(mask1.first);
    cv::Scalar mask2_sum = cv::sum(mask2.first);
    return mask1_sum(0) > mask2_sum(0);
}
