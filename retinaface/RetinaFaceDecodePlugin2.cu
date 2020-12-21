#include "RetinaFaceDecodePlugin2.h"
#include "common.hpp"
#include <vector>
#include "cuda_runtime_api.h"

namespace nvinfer1
{
    struct  RetinaFaceDecodePlugin2::PluginData{
        int net_width;
        int net_height;
        int num_landmarks;
        int feature_map_data_length;
        int* feature_map_size_gpu_data;
        float* anchor_gpu_data;
        int blob_size;
        int thread_count = 256;
        std::vector<float> anchors;
        std::vector<int> feature_maps;
    };

    RetinaFaceDecodePlugin2::RetinaFaceDecodePlugin2(const int net_width, const int net_height, const int num_landmarks) 
    :plugin_data(new PluginData({net_width, net_height, num_landmarks, nullptr, nullptr, 0, 256})) 
    {
        std::vector<int> strides = {8, 16, 32};
        this->plugin_data->feature_maps.resize(strides.size() * 2);
        for(int stride_index = 0; stride_index < strides.size(); ++stride_index)
        {
            feature_map_size_cpu_data[2 * stride_index] = net_width / strides[stride_index];
            feature_map_size_cpu_data[2 * stride_index + 1] = net_height / strides[stride_index];
        }
        this->plugin_data->feature_map_data_length

        cudaMalloc((void**)&plugin_data->feature_map_size_gpu_data, sizeof(int) * feature_map_size_cpu_data.size());
        cudaMemcpy(plugin_data->feature_map_size_gpu_data, feature_map_size_cpu_data.data(), sizeof(int) * feature_map_size_cpu_data.size(), cudaMemcpyHostToDevice);
        
        this->plugin_data->anchors = {16, 32, 48, 96, 192, 384};
        cudaMalloc((void**)&plugin_data->anchor_gpu_data, sizeof(float) * anchors.size());
        cudaMemcpy(plugin_data->anchor_gpu_data, anchors.data(), sizeof(float) * anchors.size(), cudaMemcpyHostToDevice);
        
        plugin_data->blob_size = 0;
        for(int index = 0; index < strides.size(); ++index)
        {
            plugin_data->blob_size += feature_map_size_cpu_data[2 * index] * feature_map_size_cpu_data[2 * index + 1] * feature_map_size_cpu_data.size() / strides.size();
        }
        // std::vector<float> anchor_data(plugin_data->blob_size * 6);// bbox + net
    }
    
    RetinaFaceDecodePlugin2::RetinaFaceDecodePlugin2(const void *data, size_t length) 
    {
        const char* d = (const char*)data;
        read<int>(d, plugin_data->net_width);
        read<int>(d, plugin_data->net_height);
        read<int>(d, plugin_data->num_landmarks);
        
    }
    
    RetinaFaceDecodePlugin2::~RetinaFaceDecodePlugin2() 
    {
        cudaFree(plugin_data->feature_map_size_gpu_data);
        plugin_data->feature_map_size_gpu_data = nullptr;
        cudaFree(plugin_data->anchor_gpu_data);
        plugin_data->anchor_gpu_data = nullptr;
        delete plugin_data;
        plugin_data = nullptr;
    }
    
    int RetinaFaceDecodePlugin2::initialize() 
    {
        return 0;
    }
    
    __global__ void RetinafaceDecode(int batch_size,  int blob_size, const float const* bboxes, const float const* classes, const float const* landmarks, const int landmark_entry_length, 
    float* output, const int* feature_map_sizes, const int num_feature_map_size, const float* anchors, const int num_anchors, const int anchor_count_per_feature)
    {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if(idx >= batch_size * blob_size)
            return;
        int batch_index =  idx / blob_size;
        int index_in_batch = idx % blob_size;
        
        float* bbox_ptr = bbox + (batch_index * blob_size  + index_in_batch ) * anchor_count_per_feature * 4;
        float* class_ptr = classes + (batch_index * blob_size  + index_in_batch) * anchor_count_per_feature * 2;
        float* landmarks_ptr = landmarks_ptr + (batch_index * blob_size  + index_in_batch) *  anchor_count_per_feature * landmark_entry_length;
        // match anchor and feature map size;

        int anchor_index = 0;
        int width = 0, height = 0;
        int x = 0, y = 0;
        int temp = 0;
        for(int index = 0; index < num_feature_map_size / 2; ++index)
        {
            temp += feature_map_sizes[2 * index] * feature_map_sizes[2 * index + 1]
            if(index_in_batch  < feature_map_sizes[2 * index] * feature_map_sizes[2 * index + 1] )
            {
                width = feature_map_sizes[2 * index];
                height = feature_map_sizes[2 * index + 1];
                anchor_index = index;
                x = (index_in_batch - temp) % width;
                y = (index_in_batch - temp) / width;
                break;
            }
        }

        float ax, ay,aw, ah;
        for(int index = 0; index < anchor_count_per_feature ; ++index)
        {
            float conf1 = classes[2 * index];
            float conf2 = classes[2 * index + 1];
            conf2 = expf(conf2) / (expf(conf2) + expf(conf1));
            if(conf2 <= 0.02)
                continue;
            ax = (x + 0.5) / w;
            ay = (y + 0.5) / h;
            aw = anchors[anchor_index * anchor_count_per_feature + index] / w;
            ah = anchors[anchor_index * anchor_count_per_feature + index] / h;
            ax + bbox_ptr[4 * index] * 0.1 * aw;
            ay + bbox_ptr[4 * index + 1] * 0.1 * ah;
            aw * expf(bbox_ptr[4 * index + 2]) * 0.2;
            ah * expf(bbox_ptr[4 * index + 3]) * 0.2;

            for(int landmark_index = 0; landmark_entry_length / 2; ++landmark_index)
            {
                landmarks_ptr[landmark_entry_length * index + landmark_index * 2] * 0.1 * aw;
                landmarks_ptr[landmark_entry_length * index + landmark_index * 2 + 1] * 0.1 * ah;
            }
        }   
    }

    int RetinaFaceDecodePlugin2::enqueue(int batchSize, const void *const *inputs, void **outputs, void *workspace, cudaStream_t stream) 
    {

        int num_elem = batchSize * this->plugin_data->blob_size;
        int thread_count = (num_elem < this->plugin_data->thread_count) ? num_elem ? this->plugin_data->thread_count;
        int block_size = (num_elem + thread_count - 1) / thread_count;
        RetinaFaceDocode<<<block_size, thread_count>>> (batchSize, this->plugin_data->blob_size, inputs[0], 
        inputs[1], intpus[2], this->plugin_data->num_landmarks, output[0], this->plugin_data->feature_map_size_gpu_data, );
        
        return 0;
    }
    
    size_t RetinaFaceDecodePlugin2::getSerializationSize() const 
    {
        return sizeof(net_width) + sizeof(net_height) + sizeof(num_landmarks);
    }
    
    void RetinaFaceDecodePlugin2::serialize(void *buffer) const 
    {
        char* d = (char*)buffer;
        write<int>(d, net_width);
        write<int>(d, net_height);
        write<int>(d, num_landmarks);
    }
    
    const char* RetinaFaceDecodePlugin2::getPluginType() const 
    {
        return RETINAFACE_DECODE_PLUGIN;
    }
    
    const char* RetinaFaceDecodePlugin2::getPluginVersion() const 
    {
        return RETINAFACE_DECODE_VERSION;
    }
    
    void RetinaFaceDecodePlugin2::destroy() 
    {
        delete this;
    }
    
    IPluginV2IOExt* RetinaFaceDecodePlugin2::clone() const 
    {
        RetinaFaceDecodePlugin2* plugin = new RetinaFaceDecodePlugin2(this->net_width, this->net_height, this->num_landmarks);
        plugin->setPluginNamespace(this->getPluginNamespace());
        return plugin;
    }
    
    void RetinaFaceDecodePlugin2::setPluginNamespace(const char *pluginNamespace) 
    {
        this->mPluginNamespace = pluginNamespace;
    }
    
    const char* RetinaFaceDecodePlugin2::getPluginNamespace() const 
    {
        return this->mPluginNamespace;
    }
    
    DataType RetinaFaceDecodePlugin2::getOutputDataType(int index, const nvinfer1::DataType *inputTypes, int nbInputs) const 
    {
        // assert(nbInputs == 1);
        return DataType::kFLOAT;

    }
    
    bool RetinaFaceDecodePlugin2::isOutputBroadcastAcrossBatch(int outputIndex, const bool *inputIsBroadcasted, int nbInputs) const 
    {
        return false;
    }
    
    bool RetinaFaceDecodePlugin2::canBroadcastInputAcrossBatch(int inputIndex) const 
    {
        return false;
    }
    
    void RetinaFaceDecodePlugin2::attachToContext(cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) 
    {
        
    }
    
    void RetinaFaceDecodePlugin2::configurePlugin(const PluginTensorDesc *in, int nbInput, const PluginTensorDesc *out, int nbOutput) 
    {
        
    }
    
    void RetinaFaceDecodePlugin2::detachFromContext() 
    {
        
    }
    
    void RetinaFaceDecodePlugin2::forwardGpu(const float *const *inputs, float *output, cudaStream_t stream, int batchSize) 
    {
        
    }
}