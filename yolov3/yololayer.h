#ifndef _YOLO_LAYER_H
#define _YOLO_LAYER_H

#include <assert.h>
#include <cmath>
#include <string.h>
#include <cublas_v2.h>
#include "NvInfer.h"
#include "Utils.h"
#include <iostream>

#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define BBOX_CONF_THRESH 0.5
#define BATCH_SIZE 8
#define WEIGHT_PATH ("../car.wts")

namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 1;
    static constexpr int INPUT_H = 416;
    static constexpr int INPUT_W = 416;
    static constexpr const char* YOLOV3_PLUGIN_NAME = "YoloV3Layer_TRT";
   

    struct YoloKernel
    {
        int width;
        int height;
        float anchors[CHECK_COUNT*2];
    };

    #define VEHICLE

    

    // static constexpr YoloKernel yolo1 = {
    //     INPUT_W / 32,
    //     INPUT_H / 32,
    //     #if defined(PERSON)
        
    //      {28,25,  14,34,  18,38}
    //     #elif defined(VEHICLE)
        
    //     {10,13,  16,30,  33,23}
    //     #else
    //     {10,13,  16,30,  33,23}
           
    //     #endif

    // };
    // static constexpr YoloKernel yolo2 = {
    //     INPUT_W / 16,
    //     INPUT_H / 16,
    //     #if defined(PERSON)
       
    //     {33,23,  30,71,  24,55}
    //     #elif defined(VEHICLE)
    //     {30,61,  62,45,  59,119}
    //     #else
    //     {30,61,  62,45,  59,119}
    //     #endif
    // };
    // static constexpr YoloKernel yolo3 = {
    //     INPUT_W / 8,
    //     INPUT_H / 8,
    //     #if defined(PERSON)
    //     {12,14,  11,25,  16,38}
    //     #elif defined(VEHICLE)
    //     {116,90,  156,198,  373,326}
    //     #else
    //     {116,90,  156,198,  373,326}
    //     #endif
    // };

    static constexpr int LOCATIONS = 4;
    struct alignas(float) Detection{
        //x y w h
        float bbox[LOCATIONS];
        float det_confidence;
        float class_id;
        float class_confidence;
    };
}


namespace nvinfer1
{
    class YoloV3LayerPlugin: public IPluginV2IOExt
    {
        public:
            explicit YoloV3LayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel);
            YoloV3LayerPlugin(const void* data, size_t length);

            ~YoloV3LayerPlugin();

            int getNbOutputs() const override
            {
                return 1;
            }

            Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;

            int initialize() override;

            virtual void terminate() override {};

            virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0;}

            virtual int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override;

            virtual size_t getSerializationSize() const override;

            virtual void serialize(void* buffer) const override;

            bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) const override {
                return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
            }

            const char* getPluginType() const override;

            const char* getPluginVersion() const override;

            void destroy() override;

            IPluginV2IOExt* clone() const override;

            void setPluginNamespace(const char* pluginNamespace) override;

            const char* getPluginNamespace() const override;

            DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

            bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

            bool canBroadcastInputAcrossBatch(int inputIndex) const override;

            void attachToContext(
                    cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

            void configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) override;

            void detachFromContext() override;

        private:
            void forwardGpu(const float *const * inputs,float * output, cudaStream_t stream,int batchSize = 1);
              int mThreadCount = 256;
        const char* mPluginNamespace;
        int mKernelCount;
        int mClassCount;
        int mYoloV5NetWidth;
        int mYoloV5NetHeight;
        int mMaxOutObject;
        std::vector<Yolo::YoloKernel> mYoloKernel;
		void** mAnchor;
    };

    class YoloV3PluginCreator : public IPluginCreator
    {
        public:
            YoloV3PluginCreator();

            ~YoloV3PluginCreator() override = default;

            const char* getPluginName() const override;

            const char* getPluginVersion() const override;

            const PluginFieldCollection* getFieldNames() override;

            IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) override;

            IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

            void setPluginNamespace(const char* libNamespace) override
            {
                mNamespace = libNamespace;
            }

            const char* getPluginNamespace() const override
            {
                return mNamespace.c_str();
            }

        private:
            std::string mNamespace;
            static PluginFieldCollection mFC;
            static std::vector<PluginField> mPluginAttributes;
    };
    REGISTER_TENSORRT_PLUGIN(YoloV3PluginCreator);
};

#endif 
