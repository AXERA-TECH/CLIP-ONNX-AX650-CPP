#pragma once
#include "CLIPImageEncoder.hpp"
#include "ax_model_runner_ax650.hpp"

class CLIPImageEncoderAX650 : public CLIPImageEncoder
{
private:
    std::shared_ptr<ax_runner_base> m_encoder;
    cv::Mat input;

public:
    bool load_image_encoder(std::string encoder_path) override
    {
        m_encoder.reset(new ax_runner_ax650);
        m_encoder->init(encoder_path.c_str());
        input_height = m_encoder->get_input(0).vShape[2];
        input_width = m_encoder->get_input(0).vShape[3];
        ALOGI("input size %d %d", input_height, input_width);
        // input = cv::Mat(input_height, input_width, CV_8UC3, m_encoder->get_input(0).pVirAddr);

        LEN_IMAGE_FEATURE = m_encoder->get_output(0).vShape[1];
        ALOGI("image feature len %d", LEN_IMAGE_FEATURE);
        image_features_input = std::vector<float>(1024 * LEN_IMAGE_FEATURE);
        return true;
    }
    bool encode(cv::Mat image, std::vector<float> &image_features) override
    {
        if (!m_encoder.get())
        {
            ALOGE("encoder not init");
            return false;
        }
        cv::resize(image, input, cv::Size(input_width, input_height));
        cv::cvtColor(input, input, cv::COLOR_BGR2RGB);

        float *inputPtr = (float *)m_encoder->get_input(0).pVirAddr;

        uchar *img_data = input.data;

        int letterbox_cols = input_width;
        int letterbox_rows = input_height;
        for (int c = 0; c < 3; c++)
        {
            for (int h = 0; h < letterbox_rows; h++)
            {
                for (int w = 0; w < letterbox_cols; w++)
                {
                    int in_index = h * letterbox_cols * 3 + w * 3 + c;
                    int out_index = c * letterbox_rows * letterbox_cols + h * letterbox_cols + w;
                    inputPtr[out_index] = (float(img_data[in_index]) - _mean_val[c]) * _std_val[c];
                }
            }
        }

        auto ret = m_encoder->inference();

        image_features.resize(LEN_IMAGE_FEATURE);
        // m_encoder->mem_sync_output(0);
        memcpy(image_features.data(), m_encoder->get_output(0).pVirAddr, LEN_IMAGE_FEATURE * sizeof(float));
        return true;
    }
};
