#include "yololayer.h"

using namespace Yolo;



namespace nvinfer1
{
    YoloV3LayerPlugin::YoloV3LayerPlugin(int classCount, int netWidth, int netHeight, int maxOut, const std::vector<Yolo::YoloKernel>& vYoloKernel)
    {
        mClassCount = classCount;
        mYoloV5NetWidth = netWidth;
        mYoloV5NetHeight = netHeight;
        mMaxOutObject = maxOut;
        mYoloKernel = vYoloKernel;
        mKernelCount = vYoloKernel.size();

        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t AnchorLen = sizeof(float)* CHECK_COUNT * 2;
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen));
            const auto& yolo = mYoloKernel[ii];
            CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }
    }
    
    YoloV3LayerPlugin::~YoloV3LayerPlugin()
    {
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaFree(mAnchor[ii]));
        }
        CUDA_CHECK(cudaFreeHost(mAnchor));
    }

    // create the plugin at runtime from a byte stream
    YoloV3LayerPlugin::YoloV3LayerPlugin(const void* data, size_t length)
    {
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        read(d, mClassCount);
        read(d, mThreadCount);
        read(d, mKernelCount);
        read(d, mYoloV5NetWidth);
        read(d, mYoloV5NetHeight);
        read(d, mMaxOutObject);
        mYoloKernel.resize(mKernelCount);
        auto kernelSize = mKernelCount * sizeof(YoloKernel);
        memcpy(mYoloKernel.data(), d, kernelSize);
        d += kernelSize;
        CUDA_CHECK(cudaMallocHost(&mAnchor, mKernelCount * sizeof(void*)));
        size_t AnchorLen = sizeof(float)* CHECK_COUNT * 2;
        for (int ii = 0; ii < mKernelCount; ii++)
        {
            CUDA_CHECK(cudaMalloc(&mAnchor[ii], AnchorLen));
            const auto& yolo = mYoloKernel[ii];
            CUDA_CHECK(cudaMemcpy(mAnchor[ii], yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
        }
        assert(d == a + length);
    }

    void YoloV3LayerPlugin::serialize(void* buffer) const
    {
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mClassCount);
        write(d, mThreadCount);
        write(d, mKernelCount);
        write(d, mYoloV5NetWidth);
        write(d, mYoloV5NetHeight);
        write(d, mMaxOutObject);
        auto kernelSize = mKernelCount * sizeof(YoloKernel);
        memcpy(d, mYoloKernel.data(), kernelSize);
        d += kernelSize;

        assert(d == a + getSerializationSize());
    }
    
    size_t YoloV3LayerPlugin::getSerializationSize() const
    {  
        return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount) + sizeof(Yolo::YoloKernel) * mYoloKernel.size() + sizeof(mYoloV5NetWidth) + sizeof(mYoloV5NetHeight) + sizeof(mMaxOutObject);
    }

    int YoloV3LayerPlugin::initialize()
    { 
        return 0;
    }
    
    Dims YoloV3LayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
      //output the result to channel
      int totalsize = mMaxOutObject * sizeof(Detection) / sizeof(float);

      return Dims3(totalsize + 1, 1, 1);
    }

    // Set plugin namespace
    void YoloV3LayerPlugin::setPluginNamespace(const char* pluginNamespace)
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* YoloV3LayerPlugin::getPluginNamespace() const
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType YoloV3LayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool YoloV3LayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool YoloV3LayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    void YoloV3LayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void YoloV3LayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
    {
    }

    // Detach the plugin object from its execution context.
    void YoloV3LayerPlugin::detachFromContext() {}

    const char* YoloV3LayerPlugin::getPluginType() const
    {
        return Yolo::YOLOV3_PLUGIN_NAME;
    }

    const char* YoloV3LayerPlugin::getPluginVersion() const
    {
        return "1";
    }

    void YoloV3LayerPlugin::destroy()
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* YoloV3LayerPlugin::clone() const
    {
        YoloV3LayerPlugin *p = new YoloV3LayerPlugin(mClassCount, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject, mYoloKernel);
        
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __device__ float Logist(float data){ return 1.0f / (1.0f + expf(-data)); };

    __global__ void CalDetection(const float *input, float *output,int noElements, 
            int netWidth,int netHeight,int maxoutobject, int yoloWidth, int yoloHeight, const float anchors[CHECK_COUNT*2],int classes,int outputElem) {
 
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= noElements) return;

        int total_grid = yoloWidth * yoloHeight;
        int bnIdx = idx / total_grid;
        idx = idx - total_grid*bnIdx;
        int info_len_i = 5 + classes;
        const float* curInput = input + bnIdx * (info_len_i * total_grid * CHECK_COUNT);

        for (int k = 0; k < 3; ++k) {
            int class_id = 0;
            float max_cls_prob = 0.0;
            for (int i = 5; i < info_len_i; ++i) {
                float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
                if (p > max_cls_prob) {
                    max_cls_prob = p;
                    class_id = i - 5;
                }
            }
            float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
            if (max_cls_prob < IGNORE_THRESH || box_prob < IGNORE_THRESH) continue;

            float *res_count = output + bnIdx*outputElem;
            int count = (int)atomicAdd(res_count, 1);
            if (count >= maxoutobject) return;
            char* data = (char * )res_count + sizeof(float) + count*sizeof(Detection);
            Detection* det =  (Detection*)(data);

            int row = idx / yoloWidth;
            int col = idx % yoloWidth;

            //Location
            det->bbox[0] = (col + Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * netWidth / yoloWidth;
            det->bbox[1] = (row + Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * netHeight / yoloHeight;
            det->bbox[2] = expf(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]) * anchors[2*k];
            det->bbox[3] = expf(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]) * anchors[2*k + 1];
            det->det_confidence = box_prob;
            det->class_id = class_id;
            det->class_confidence = max_cls_prob;
        }
    }

    void YoloV3LayerPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {
        
        int outputElem = 1 + mMaxOutObject * sizeof(Detection) / sizeof(float);
        for (int idx = 0; idx < batchSize; ++idx) {
            CUDA_CHECK(cudaMemset(output + idx * outputElem, 0, sizeof(float)));
        }

        int numElem = 0;
        for (unsigned int i = 0;i< mYoloKernel.size();++i)
        {
            const auto& yolo = mYoloKernel[i];
            numElem = yolo.width*yolo.height*batchSize;
            if (numElem < mThreadCount)
                mThreadCount = numElem;
            CalDetection<<< (yolo.width*yolo.height*batchSize + mThreadCount - 1) / mThreadCount, mThreadCount>>>
                (inputs[i],output, numElem, mYoloV5NetWidth, mYoloV5NetHeight, mMaxOutObject,  yolo.width, yolo.height,  (float *)mAnchor[i], mClassCount ,outputElem);
        }

        // CUDA_CHECK(cudaFree(devAnchor));
    }


    int YoloV3LayerPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        //assert(batchSize == 1);
        //GPU
        //CUDA_CHECK(cudaStreamSynchronize(stream));
        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);

        return 0;
    }

    PluginFieldCollection YoloV3PluginCreator::mFC{};
    std::vector<PluginField> YoloV3PluginCreator::mPluginAttributes;

    YoloV3PluginCreator::YoloV3PluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* YoloV3PluginCreator::getPluginName() const
    {
            return YOLOV3_PLUGIN_NAME;
    }

    const char* YoloV3PluginCreator::getPluginVersion() const
    {
            return "1";
    }

    const PluginFieldCollection* YoloV3PluginCreator::getFieldNames()
    {
            return &mFC;
    }

    IPluginV2IOExt* YoloV3PluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    {
        int class_count = 80;
        int input_w = 416;
        int input_h = 416;
        int max_output_object_count = 1000;
        std::vector<Yolo::YoloKernel> yolo_kernels(3);

        const PluginField* fields = fc->fields;
        for (int i = 0; i < fc->nbFields; i++) {
            if (strcmp(fields[i].name, "netdata") == 0) {
                assert(fields[i].type == PluginFieldType::kFLOAT32);
                int *tmp = (int*)(fields[i].data);
                class_count = tmp[0];
                input_w = tmp[1];
                input_h = tmp[2];
                max_output_object_count = tmp[3];
            } else if (strstr(fields[i].name, "yolodata") != NULL) {
                assert(fields[i].type == PluginFieldType::kFLOAT32);
                int *tmp = (int*)(fields[i].data);
                YoloKernel kernel;
                kernel.width = tmp[0];
                kernel.height = tmp[1];
                for (int j = 0; j < fields[i].length - 2; j++) {
                    kernel.anchors[j] = tmp[j + 2];
                }
                yolo_kernels[2 - (fields[i].name[8] - '1')] = kernel;
            }
        }
        auto obj = new YoloV3LayerPlugin(class_count, input_w, input_h, max_output_object_count, yolo_kernels);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* YoloV3PluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
    {
        // This object will be deleted when the network is destroyed, which will
        // call MishPlugin::destroy()
        YoloV3LayerPlugin* obj = new YoloV3LayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

}
