#include "ax_model_runner_ax650.hpp"
#ifdef BUILD_WITH_AX650
#include "string.h"
#include "fstream"
#include "memory"
// #include "utilities/file.hpp"
#include <ax_sys_api.h>
#include <ax_engine_api.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "sample_log.h"

#define AX_CMM_ALIGN_SIZE 128

const char *AX_CMM_SESSION_NAME = "npu";

typedef enum
{
    AX_ENGINE_ABST_DEFAULT = 0,
    AX_ENGINE_ABST_CACHED = 1,
} AX_ENGINE_ALLOC_BUFFER_STRATEGY_T;

typedef std::pair<AX_ENGINE_ALLOC_BUFFER_STRATEGY_T, AX_ENGINE_ALLOC_BUFFER_STRATEGY_T> INPUT_OUTPUT_ALLOC_STRATEGY;

bool file_exist(const std::string &path)
{
    auto flag = false;

    std::fstream fs(path, std::ios::in | std::ios::binary);
    flag = fs.is_open();
    fs.close();

    return flag;
}

class MMap
{
private:
    void *_add;
    int _size;

public:
    MMap() {}
    MMap(const char *file)
    {
        _add = _mmap(file, &_size);
    }
    ~MMap()
    {
        munmap(_add, _size);
    }

    size_t size()
    {
        return _size;
    }

    void *data()
    {
        return _add;
    }

    static void *_mmap(const char *model_file, int *model_size)
    {
        auto *file_fp = fopen(model_file, "r");
        if (!file_fp)
        {
            ALOGE("Read Run-Joint model(%s) file failed.\n", model_file);
            return nullptr;
        }
        fseek(file_fp, 0, SEEK_END);
        *model_size = ftell(file_fp);
        fclose(file_fp);
        int fd = open(model_file, O_RDWR, 0644);
        void *mmap_add = mmap(NULL, *model_size, PROT_WRITE, MAP_SHARED, fd, 0);
        return mmap_add;
    }
};

bool read_file(const std::string &path, std::vector<char> &data)
{
    std::fstream fs(path, std::ios::in | std::ios::binary);

    if (!fs.is_open())
    {
        return false;
    }

    fs.seekg(std::ios::end);
    auto fs_end = fs.tellg();
    fs.seekg(std::ios::beg);
    auto fs_beg = fs.tellg();

    auto file_size = static_cast<size_t>(fs_end - fs_beg);
    auto vector_size = data.size();

    data.reserve(vector_size + file_size);
    data.insert(data.end(), std::istreambuf_iterator<char>(fs), std::istreambuf_iterator<char>());

    fs.close();

    return true;
}

void free_io_index(AX_ENGINE_IO_BUFFER_T *io_buf, int index)
{
    for (int i = 0; i < index; ++i)
    {
        AX_ENGINE_IO_BUFFER_T *pBuf = io_buf + i;
        AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
    }
}

void free_io(AX_ENGINE_IO_T *io)
{
    for (size_t j = 0; j < io->nInputSize; ++j)
    {
        AX_ENGINE_IO_BUFFER_T *pBuf = io->pInputs + j;
        AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
    }
    for (size_t j = 0; j < io->nOutputSize; ++j)
    {
        AX_ENGINE_IO_BUFFER_T *pBuf = io->pOutputs + j;
        AX_SYS_MemFree(pBuf->phyAddr, pBuf->pVirAddr);
    }
    delete[] io->pInputs;
    delete[] io->pOutputs;
}

static inline int prepare_io(AX_ENGINE_IO_INFO_T *info, AX_ENGINE_IO_T *io_data, INPUT_OUTPUT_ALLOC_STRATEGY strategy)
{
    memset(io_data, 0, sizeof(*io_data));
    io_data->pInputs = new AX_ENGINE_IO_BUFFER_T[info->nInputSize];
    io_data->nInputSize = info->nInputSize;

    auto ret = 0;
    for (uint i = 0; i < info->nInputSize; ++i)
    {
        auto meta = info->pInputs[i];
        auto buffer = &io_data->pInputs[i];
        if (strategy.first == AX_ENGINE_ABST_CACHED)
        {
            ret = AX_SYS_MemAllocCached((AX_U64 *)(&buffer->phyAddr), &buffer->pVirAddr, meta.nSize, AX_CMM_ALIGN_SIZE, (const AX_S8 *)(AX_CMM_SESSION_NAME));
        }
        else
        {
            ret = AX_SYS_MemAlloc((AX_U64 *)(&buffer->phyAddr), &buffer->pVirAddr, meta.nSize, AX_CMM_ALIGN_SIZE, (const AX_S8 *)(AX_CMM_SESSION_NAME));
        }

        if (ret != 0)
        {
            free_io_index(io_data->pInputs, i);
            fprintf(stderr, "Allocate input{%d} { phy: %p, vir: %p, size: %lu Bytes }. fail \n", i, (void *)buffer->phyAddr, buffer->pVirAddr, (long)meta.nSize);
            return ret;
        }
        // fprintf(stderr, "Allocate input{%d} { phy: %p, vir: %p, size: %lu Bytes }. \n", i, (void*)buffer->phyAddr, buffer->pVirAddr, (long)meta.nSize);
    }

    io_data->pOutputs = new AX_ENGINE_IO_BUFFER_T[info->nOutputSize];
    io_data->nOutputSize = info->nOutputSize;
    for (uint i = 0; i < info->nOutputSize; ++i)
    {
        auto meta = info->pOutputs[i];
        auto buffer = &io_data->pOutputs[i];
        buffer->nSize = meta.nSize;
        if (strategy.second == AX_ENGINE_ABST_CACHED)
        {
            ret = AX_SYS_MemAllocCached((AX_U64 *)(&buffer->phyAddr), &buffer->pVirAddr, meta.nSize, AX_CMM_ALIGN_SIZE, (const AX_S8 *)(AX_CMM_SESSION_NAME));
        }
        else
        {
            ret = AX_SYS_MemAlloc((AX_U64 *)(&buffer->phyAddr), &buffer->pVirAddr, meta.nSize, AX_CMM_ALIGN_SIZE, (const AX_S8 *)(AX_CMM_SESSION_NAME));
        }
        if (ret != 0)
        {
            fprintf(stderr, "Allocate output{%d} { phy: %p, vir: %p, size: %lu Bytes }. fail \n", i, (void *)buffer->phyAddr, buffer->pVirAddr, (long)meta.nSize);
            free_io_index(io_data->pInputs, io_data->nInputSize);
            free_io_index(io_data->pOutputs, i);
            return ret;
        }
        // fprintf(stderr, "Allocate output{%d} { phy: %p, vir: %p, size: %lu Bytes }.\n", i, (void*)buffer->phyAddr, buffer->pVirAddr, (long)meta.nSize);
    }

    return 0;
}

struct ax_joint_runner_ax650_handle_t
{
    AX_ENGINE_HANDLE handle;
    AX_ENGINE_IO_INFO_T *io_info;
    AX_ENGINE_IO_T io_data;

    int algo_width, algo_height;
    int algo_colorformat;
};

int ax_runner_ax650::init(const char *model_file)
{
    if (m_handle)
    {
        return -1;
    }
    m_handle = new ax_joint_runner_ax650_handle_t;

    // 1. init engine
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    AX_SYS_Init();
    auto ret = AX_ENGINE_Init(&npu_attr);
    if (0 != ret)
    {
        return ret;
    }

    // 2. load model
    std::shared_ptr<MMap> model_buffer(new MMap(model_file));
    if (!model_buffer->data())
    {
        ALOGE("mmap");
        return -1;
    }

    // 3. create handle

    ret = AX_ENGINE_CreateHandle(&m_handle->handle, model_buffer->data(), model_buffer->size());
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_CreateHandle");
        return ret;
    }
    fprintf(stdout, "Engine creating handle is done.\n");

    // 4. create context
    ret = AX_ENGINE_CreateContext(m_handle->handle);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_CreateContext");
        return ret;
    }
    fprintf(stdout, "Engine creating context is done.\n");

    // 5. set io

    ret = AX_ENGINE_GetIOInfo(m_handle->handle, &m_handle->io_info);
    if (0 != ret)
    {
        ALOGE("AX_ENGINE_GetIOInfo");
        return ret;
    }
    fprintf(stdout, "Engine get io info is done. \n");

    // 6. alloc io

    ret = prepare_io(m_handle->io_info, &m_handle->io_data, std::make_pair(AX_ENGINE_ABST_DEFAULT, AX_ENGINE_ABST_CACHED));
    if (0 != ret)
    {
        ALOGE("prepare_io");
        return ret;
    }
    fprintf(stdout, "Engine alloc io is done. \n");

    m_handle->algo_width = m_handle->io_info->pInputs[0].pShape[2];

    switch (m_handle->io_info->pInputs[0].pExtraMeta->eColorSpace)
    {
    case AX_ENGINE_CS_NV12:
        m_handle->algo_colorformat = (int)AX_FORMAT_YUV420_SEMIPLANAR;
        m_handle->algo_height = m_handle->io_info->pInputs[0].pShape[1] / 1.5;
        ALOGI("NV12 MODEL");
        break;
    case AX_ENGINE_CS_RGB:
        m_handle->algo_colorformat = (int)AX_FORMAT_RGB888;
        m_handle->algo_height = m_handle->io_info->pInputs[0].pShape[1];
        ALOGI("RGB MODEL");
        break;
    case AX_ENGINE_CS_BGR:
        m_handle->algo_colorformat = (int)AX_FORMAT_BGR888;
        m_handle->algo_height = m_handle->io_info->pInputs[0].pShape[1];
        ALOGI("BGR MODEL");
        break;
    default:
        ALOGE("now ax-pipeline just only support NV12/RGB/BGR input format,you can modify by yourself");
        // return deinit_joint();
        return -1;
    }

    for (size_t i = 0; i < m_handle->io_info->nOutputSize; i++)
    {
        ax_runner_tensor_t tensor;
        tensor.nIdx = i;
        tensor.sName = std::string(m_handle->io_info->pOutputs[i].pName);
        tensor.nSize = m_handle->io_info->pOutputs[i].nSize;
        for (size_t j = 0; j < m_handle->io_info->pOutputs[i].nShapeSize; j++)
        {
            tensor.vShape.push_back(m_handle->io_info->pOutputs[i].pShape[j]);
        }
        tensor.phyAddr = m_handle->io_data.pOutputs[i].phyAddr;
        tensor.pVirAddr = m_handle->io_data.pOutputs[i].pVirAddr;
        mtensors.push_back(tensor);
    }

    for (size_t i = 0; i < m_handle->io_info->nInputSize; i++)
    {
        ax_runner_tensor_t tensor;
        tensor.nIdx = i;
        tensor.sName = std::string(m_handle->io_info->pInputs[i].pName);
        tensor.nSize = m_handle->io_info->pInputs[i].nSize;
        for (size_t j = 0; j < m_handle->io_info->pInputs[i].nShapeSize; j++)
        {
            tensor.vShape.push_back(m_handle->io_info->pInputs[i].pShape[j]);
        }
        tensor.phyAddr = m_handle->io_data.pInputs[i].phyAddr;
        tensor.pVirAddr = m_handle->io_data.pInputs[i].pVirAddr;
        minput_tensors.push_back(tensor);
    }

    return ret;
}

void ax_runner_ax650::deinit()
{
    if (m_handle && m_handle->handle)
    {
        free_io(&m_handle->io_data);
        AX_ENGINE_DestroyHandle(m_handle->handle);
    }
    delete m_handle;
    m_handle = nullptr;
    AX_ENGINE_Deinit();
}

int ax_runner_ax650::get_algo_width() { return m_handle->algo_width; }
int ax_runner_ax650::get_algo_height() { return m_handle->algo_height; }
ax_color_space_e ax_runner_ax650::get_color_space()
{
    switch (m_handle->algo_colorformat)
    {
    case AX_FORMAT_RGB888:
        return ax_color_space_e::axdl_color_space_rgb;
    case AX_FORMAT_BGR888:
        return ax_color_space_e::axdl_color_space_bgr;
    case AX_FORMAT_YUV420_SEMIPLANAR:
        return ax_color_space_e::axdl_color_space_nv12;
    default:
        return axdl_color_space_unknown;
    }
}

int ax_runner_ax650::inference(ax_image_t *pstFrame)
{
    unsigned char *dst = (unsigned char *)minput_tensors[0].pVirAddr;
    unsigned char *src = (unsigned char *)pstFrame->pVir;

    switch (m_handle->algo_colorformat)
    {
    case AX_FORMAT_RGB888:
    case AX_FORMAT_BGR888:
        for (size_t i = 0; i < pstFrame->nHeight; i++)
        {
            memcpy(dst + i * pstFrame->nWidth * 3, src + i * pstFrame->tStride_W * 3, pstFrame->nWidth * 3);
        }
        break;
    case AX_FORMAT_YUV420_SEMIPLANAR:
    case AX_FORMAT_YUV420_SEMIPLANAR_VU:
        for (size_t i = 0; i < pstFrame->nHeight * 1.5; i++)
        {
            memcpy(dst + i * pstFrame->nWidth, src + i * pstFrame->tStride_W, pstFrame->nWidth);
        }
        break;
    default:
        break;
    }

    // memcpy(minput_tensors[0].pVirAddr, pstFrame->pVir, minput_tensors[0].nSize);
    return inference();
}
int ax_runner_ax650::inference()
{
    return AX_ENGINE_RunSync(m_handle->handle, &m_handle->io_data);
}
#else
int ax_runner_ax650::init(const char *model_file)
{
    return -1;
}

void ax_runner_ax650::deinit()
{
}

int ax_runner_ax650::get_algo_width() { return -1; }
int ax_runner_ax650::get_algo_height() { return -1; }
ax_color_space_e ax_runner_ax650::get_color_space()
{
    return axdl_color_space_unknown;
}

int ax_runner_ax650::inference(ax_image_t *pstFrame)
{
    return -1;
}

int ax_runner_ax650::inference()
{
    return -1;
}
#endif // BUILD_WITH_AX650