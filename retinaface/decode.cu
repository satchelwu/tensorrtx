#include "decode.h"
#include "stdio.h"
#include "common.hpp"

namespace nvinfer1
{
    RetinaFaceDecodePlugin::RetinaFaceDecodePlugin(const int net_width, const int net_height, const int num_landmarks)
    :net_width(net_width), net_height(net_height), num_landmarks(num_landmarks)
    {

    }

    RetinaFaceDecodePlugin::~RetinaFaceDecodePlugin()
    {

    }

    // create the plugin at runtime from a byte stream
    RetinaFaceDecodePlugin::RetinaFaceDecodePlugin(const void* data, size_t length)
    {
        const char* d = (const char*)data;
        read<int>(d, net_width);
        read<int>(d, net_height);
        read<int>(d, num_landmarks);
    }

    void RetinaFaceDecodePlugin::serialize(void* buffer) const
    {
        char* d = (char*)buffer;
        write<int>(d, net_width);
        write<int>(d, net_height);
        write<int>(d, num_landmarks);
    }

    size_t RetinaFaceDecodePlugin::getSerializationSize() const
    {  
        return sizeof(net_width) + sizeof(net_height) + sizeof(num_landmarks);
    }

    int RetinaFaceDecodePlugin::initialize()
    { 
        return 0;
    }

    Dims RetinaFaceDecodePlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        //output the result to channel
        int totalCount = 0;
        
        totalCount += net_height / 8 * net_width / 8 * 2 * sizeof(RetinaFace::Detection) / sizeof(float);
        totalCount += net_height / 16 * net_width / 16 * 2 * sizeof(RetinaFace::Detection) / sizeof(float);
        totalCount += net_height / 32 * net_width / 32 * 2 * sizeof(RetinaFace::Detection) / sizeof(float);

        return Dims3(totalCount + 1, 1, 1);
    }

    // Set plugin namespace
    void RetinaFaceDecodePlugin::setPluginNamespace(const char* pluginNamespace)
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* RetinaFaceDecodePlugin::getPluginNamespace() const
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType RetinaFaceDecodePlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool RetinaFaceDecodePlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool RetinaFaceDecodePlugin::canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    void RetinaFaceDecodePlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void RetinaFaceDecodePlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
    {
    }

    // Detach the plugin object from its execution context.
    void RetinaFaceDecodePlugin::detachFromContext() {}

    const char* RetinaFaceDecodePlugin::getPluginType() const
    {
        return "Decode_TRT";
    }

    const char* RetinaFaceDecodePlugin::getPluginVersion() const
    {
        return "1";
    }

    void RetinaFaceDecodePlugin::destroy()
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* RetinaFaceDecodePlugin::clone() const
    {
        RetinaFaceDecodePlugin *p = new RetinaFaceDecodePlugin(this->net_width, this->net_height, this->num_landmarks);
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __device__ float Logist(float data){ return 1./(1. + expf(-data)); };

    __global__ void CalDetection(const float *input, float *output, int net_width, int net_height, int num_landmarks, int num_elem, int step, int anchor, int output_elem) {

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_elem) return;

        int h = net_height / step;
        int w = net_width / step;
        int total_grid = h * w;
        int bn_idx = idx / total_grid;
        idx = idx - bn_idx * total_grid;
        int y = idx / w;
        int x = idx % w;
        const float* cur_input = input + bn_idx * (4 + 2 + num_landmarks) * 2 * total_grid;
        const float *bbox_reg = &cur_input[0];
        const float *cls_reg = &cur_input[2 * 4 * total_grid];
        const float *lmk_reg = &cur_input[2 * 4 * total_grid + 2 * 2 * total_grid];

        for (int k = 0; k < 2; ++k) {
            float conf1 = cls_reg[idx + k * total_grid * 2];
            float conf2 = cls_reg[idx + k * total_grid * 2 + total_grid];
            conf2 = expf(conf2) / (expf(conf1) + expf(conf2));
            if (conf2 <= 0.02) continue;

            float *res_count = output + bn_idx * output_elem;
            int count = (int)atomicAdd(res_count, 1);
            char* data = (char *)res_count + sizeof(float) + count * sizeof(RetinaFace::Detection);
            RetinaFace::Detection* det = (RetinaFace::Detection*)(data);

            float prior[4];
            prior[0] = ((float)x + 0.5) / w;
            prior[1] = ((float)y + 0.5) / h;
            prior[2] = (float)anchor * (k + 1) / net_width;
            prior[3] = (float)anchor * (k + 1) / net_height;

            //Location
            det->bbox[0] = prior[0] + bbox_reg[idx + k * total_grid * 4] * 0.1 * prior[2];
            det->bbox[1] = prior[1] + bbox_reg[idx + k * total_grid * 4 + total_grid] * 0.1 * prior[3];
            det->bbox[2] = prior[2] * expf(bbox_reg[idx + k * total_grid * 4 + total_grid * 2] * 0.2);
            det->bbox[3] = prior[3] * expf(bbox_reg[idx + k * total_grid * 4 + total_grid * 3] * 0.2);
            det->bbox[0] -= det->bbox[2] / 2;
            det->bbox[1] -= det->bbox[3] / 2;
            det->bbox[2] += det->bbox[0];
            det->bbox[3] += det->bbox[1];
            det->bbox[0] *= net_width;
            det->bbox[1] *= net_height;
            det->bbox[2] *= net_width;
            det->bbox[3] *= net_height;
            det->class_confidence = conf2;
            for (int i = 0; i < num_landmarks; i += 2) {
                det->landmark[i] = prior[0] + lmk_reg[idx + k * total_grid * num_landmarks + total_grid * i] * 0.1 * prior[2];
                det->landmark[i+1] = prior[1] + lmk_reg[idx + k * total_grid * num_landmarks + total_grid * (i + 1)] * 0.1 * prior[3];
                det->landmark[i] *= net_width;
                det->landmark[i+1] *= net_height;
            }
        }
    }

    void RetinaFaceDecodePlugin::forwardGpu(const float *const * inputs, float * output, cudaStream_t stream, int batchSize)
    {
        int num_elem = 0;
        int base_step = 8;
        // int base_anchor = 16;
        std::vector<int> base_anchors = {16, 48, 192};
        int thread_count;

        int totalCount = 1;
        totalCount += net_height / 8 * net_width / 8 * 2 * sizeof(RetinaFace::Detection) / sizeof(float);
        totalCount += net_height / 16 * net_width / 16 * 2 * sizeof(RetinaFace::Detection) / sizeof(float);
        totalCount += net_height / 32 * net_width / 32 * 2 * sizeof(RetinaFace::Detection) / sizeof(float);
        for(int idx = 0 ; idx < batchSize; ++idx) {
            cudaMemset(output + idx * totalCount, 0, sizeof(float));
        }

        for (unsigned int i = 0; i < 3; ++i)
        {
            num_elem = batchSize * net_height / base_step * net_width / base_step;
            thread_count = (num_elem < thread_count_) ? num_elem : thread_count_;
            CalDetection<<< (num_elem + thread_count - 1) / thread_count, thread_count>>>
                (inputs[i], output, net_width, net_height, num_landmarks, num_elem, base_step, base_anchors[i], totalCount);
            base_step *= 2;
            // base_anchor *= 4;
        }
    }

    int RetinaFaceDecodePlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        //GPU
        //CUDA_CHECK(cudaStreamSynchronize(stream));
        forwardGpu((const float *const *)inputs, (float *)outputs[0], stream, batchSize);
        return 0;
    };

    PluginFieldCollection RetinaFaceDecodePluginCreator::mFC;
    std::vector<nvinfer1::PluginField> RetinaFaceDecodePluginCreator::mPluginAttributes;

    RetinaFaceDecodePluginCreator::RetinaFaceDecodePluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* RetinaFaceDecodePluginCreator::getPluginName() const
    {
        return "Decode_TRT";
    }

    const char* RetinaFaceDecodePluginCreator::getPluginVersion() const
    {
        return "1";
    }

    const PluginFieldCollection* RetinaFaceDecodePluginCreator::getFieldNames()
    {
        return &mFC;
    }

    IPluginV2IOExt* RetinaFaceDecodePluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    {
        
        int net_width = 0, net_height = 0;
        int num_landmarks = 0;

        for(int index = 0; index < fc->nbFields; ++index)
        {
            PluginField pf = fc->fields[index];
            if(!strcmp(pf.name, "net_width"))
            {
                net_width = *(int*)pf.data; 
            }
            else if(!strcmp(pf.name, "net_height"))
            {
                net_height = *(int*)pf.data;
            }
            else if(!strcmp(pf.name, "num_landmarks"))
            {
                num_landmarks = *(int*)pf.data;
            }
        }
        std::cout << "net width：" << net_width << std::endl;
        std::cout << "net height: " << net_height << std::endl;
        std::cout << "num_landmarks: " << num_landmarks << std::endl;
        RetinaFaceDecodePlugin* obj = new RetinaFaceDecodePlugin(net_width, net_height, num_landmarks);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* RetinaFaceDecodePluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
    {
        // This object will be deleted when the network is destroyed, which will
        // call PReluPlugin::destroy()
        RetinaFaceDecodePlugin* obj = new RetinaFaceDecodePlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

};
