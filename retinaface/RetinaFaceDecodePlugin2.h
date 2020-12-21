

#pragma once
#include <string>
#include <vector>
#include "NvInfer.h"

#define RETINAFACE_DECODE_PLUGIN "RetinaFace_Decode_TRT"
#define RETINAFACE_DECODE_VERSION "1";

class DetectionEntry{
    float score;
    float bbox[4];
    float landmarks[8];
};

namespace nvinfer1
{
    class RetinaFaceDecodePlugin2 final : public IPluginV2IOExt
    {
    public:
        RetinaFaceDecodePlugin2(const int net_width, const int net_height, const int num_landmarks);
        RetinaFaceDecodePlugin2(const void *data, size_t length);

        ~RetinaFaceDecodePlugin2();

        int getNbOutputs() const override
        {
            return 1;
        }

        Dims getOutputDimensions(int index, const Dims *inputs, int nbInputDims) override;

        int initialize() override;

        virtual void terminate() override{};

        virtual size_t getWorkspaceSize(int maxBatchSize) const override { return 0; }

        virtual int enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) override;

        virtual size_t getSerializationSize() const override;

        virtual void serialize(void *buffer) const override;

        bool supportsFormatCombination(int pos, const PluginTensorDesc *inOut, int nbInputs, int nbOutputs) const override
        {
            return inOut[pos].format == TensorFormat::kLINEAR && inOut[pos].type == DataType::kFLOAT;
        }

        const char *getPluginType() const override;

        const char *getPluginVersion() const override;

        void destroy() override;

        IPluginV2IOExt *clone() const override;

        void setPluginNamespace(const char *pluginNamespace) override;

        const char *getPluginNamespace() const override;

        DataType getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const override;

        bool isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const override;

        bool canBroadcastInputAcrossBatch(int inputIndex) const override;

        void attachToContext(
            cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) override;

        void configurePlugin(const PluginTensorDesc *in, int nbInput, const PluginTensorDesc *out, int nbOutput) override;

        void detachFromContext() override;

        int input_size_;

    private:
        void forwardGpu(const float *const *inputs, float *output, cudaStream_t stream, int batchSize = 1);
        
        const char *mPluginNamespace;
        struct PluginData;
        PluginData* plugin_data{nullptr};
    };

    class RetinaFaceDecodePlugin2Creator : public IPluginCreator
    {
    public:
        RetinaFaceDecodePlugin2Creator();

        ~RetinaFaceDecodePlugin2Creator() override = default;

        const char *getPluginName() const override;

        const char *getPluginVersion() const override;

        const PluginFieldCollection *getFieldNames() override;

        IPluginV2IOExt *createPlugin(const char *name, const PluginFieldCollection *fc) override;

        IPluginV2IOExt *deserializePlugin(const char *name, const void *serialData, size_t serialLength) override;

        void setPluginNamespace(const char *libNamespace) override
        {
            mNamespace = libNamespace;
        }

        const char *getPluginNamespace() const override
        {
            return mNamespace.c_str();
        }

    private:
        std::string mNamespace;
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
    };
    REGISTER_TENSORRT_PLUGIN(RetinaFaceDecodePlugin2Creator);
}; // namespace nvinfer1
