#pragma once
#include <opencv2/opencv.hpp>

#include "sample_log.h"

class CLIPImageEncoder
{
protected:
    float _mean_val[3] = {0.48145466f * 255.f, 0.4578275f * 255.f, 0.40821073f * 255.f};
    float _std_val[3] = {1 / (0.26862954f * 255.f), 1 / (0.26130258f * 255.f), 1 / (0.27577711f * 255.f)};

    std::vector<float> image_features_input;

    int LEN_IMAGE_FEATURE = 512;
    int input_height, input_width;

public:
    virtual bool load_image_encoder(std::string encoder_path) = 0;
    virtual bool encode(cv::Mat image, std::vector<float> &image_features) = 0;

    int get_image_feature_size()
    {
        return LEN_IMAGE_FEATURE;
    }
};
