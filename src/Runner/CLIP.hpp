#pragma once
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <thread>

#include "../string_utility.hpp"
#include "sample_log.h"

#include "CLIPTextEncoderAX650.hpp"
#include "CLIPImageEncoderAX650.hpp"
#ifdef ENABLE_ONNXRUNTIME
#include "CLIPTextEncoderOnnx.hpp"
#include "CLIPImageEncoderOnnx.hpp"
#endif

class CLIP
{
protected:
    std::shared_ptr<CLIPTextEncoder> m_text_encoder;
    std::shared_ptr<CLIPImageEncoder> m_image_encoder;

    static void softmax(const std::vector<std::vector<float>> &input, std::vector<std::vector<float>> &result)
    {
        result.reserve(input.size());

        for (const auto &row : input)
        {
            std::vector<float> rowResult;
            rowResult.reserve(row.size());

            float maxVal = *std::max_element(row.begin(), row.end());

            float sumExp = 0.0;
            for (float val : row)
            {
                float expVal = std::exp(val - maxVal);
                rowResult.emplace_back(expVal);
                sumExp += expVal;
            }

            for (float &val : rowResult)
            {
                val /= sumExp;
            }

            result.emplace_back(std::move(rowResult));
        }
    }

    static void postprocess(
        const std::vector<std::vector<float>> &imageFeatures, const std::vector<std::vector<float>> &textFeatures,
        std::vector<std::vector<float>> &logits_per_image, std::vector<std::vector<float>> &logits_per_text)
    {
        std::vector<std::vector<float>> logitsPerImage;
        logitsPerImage.reserve(imageFeatures.size());

        for (const auto &_row : imageFeatures)
        {
            float norm = 0.0;
            for (float val : _row)
            {
                norm += val * val;
            }
            norm = std::sqrt(norm);
            std::vector<float> normRow;
            normRow.reserve(_row.size());
            for (float val : _row)
            {
                normRow.push_back(val / norm);
            }

            std::vector<float> row;
            row.reserve(textFeatures.size());
            for (const auto &textRow : textFeatures)
            {
                float sum = 0.0;
                for (size_t i = 0; i < normRow.size(); i++)
                {
                    sum += normRow[i] * textRow[i];
                }
                row.push_back(100 * sum);
            }
            logitsPerImage.push_back(std::move(row));
        }

        std::vector<std::vector<float>> logitsPerText(logitsPerImage[0].size(), std::vector<float>(logitsPerImage.size()));

        for (size_t i = 0; i < logitsPerImage.size(); i++)
        {
            for (size_t j = 0; j < logitsPerImage[i].size(); j++)
            {
                logitsPerText[j][i] = logitsPerImage[i][j];
            }
        }

        softmax(logitsPerImage, logits_per_image);
        softmax(logitsPerText, logits_per_text);
    }

public:
    CLIP()
    {
    }

    int get_image_feature_size()
    {
        if (m_image_encoder == nullptr)
        {
            ALOGE("image encoder is null");
            return -1;
        }
        return m_image_encoder->get_image_feature_size();
    }

    int get_text_feature_size()
    {
        if (m_text_encoder == nullptr)
        {
            ALOGE("text encoder is null");
            return -1;
        }
        return m_text_encoder->get_text_feature_size();
    }

    bool load_tokenizer(std::string vocab_path, bool isCN)
    {
        if (m_text_encoder == nullptr)
        {
            ALOGE("text encoder is null");
            return -1;
        }
        return m_text_encoder->load_tokenizer(vocab_path, isCN);
    }

    bool load_text_encoder(std::string encoder_path)
    {
        if (m_text_encoder == nullptr)
        {
            if (string_utility<std::string>::ends_with(encoder_path, ".onnx"))
            {
#ifdef ENABLE_ONNXRUNTIME
                m_text_encoder.reset(new CLIPTextEncoderOnnx);
#else
                ALOGE("don't support onnx model");
#endif
            }
            else if (string_utility<std::string>::ends_with(encoder_path, ".axmodel"))
            {
                m_text_encoder.reset(new CLIPTextEncoderAX650);
            }
            else
            {
                ALOGE("%s is not implemented", encoder_path.c_str());
                return false;
            }
        }
        return m_text_encoder->load_text_encoder(encoder_path);
    }

    bool load_image_encoder(std::string encoder_path)
    {
        if (m_image_encoder == nullptr)
        {
            if (string_utility<std::string>::ends_with(encoder_path, ".onnx"))
            {
#ifdef ENABLE_ONNXRUNTIME
                m_image_encoder.reset(new CLIPImageEncoderOnnx);
#else
                ALOGE("don't support onnx model");
#endif
            }
            else if (string_utility<std::string>::ends_with(encoder_path, ".axmodel"))
            {
                m_image_encoder.reset(new CLIPImageEncoderAX650);
            }
            else
            {
                ALOGE("%s is not implemented", encoder_path.c_str());
                return false;
            }
        }
        return m_image_encoder->load_image_encoder(encoder_path);
    }
    bool encode(cv::Mat image, std::vector<float> &image_features)
    {
        if (m_image_encoder == nullptr)
        {
            ALOGE("image encoder is null");
            return false;
        }
        return m_image_encoder->encode(image, image_features);
    }

    bool encode(std::vector<std::string> &texts, std::vector<std::vector<float>> &text_features)
    {
        if (m_text_encoder == nullptr)
        {
            ALOGE("text encoder is null");
            return false;
        }
        return m_text_encoder->encode(texts, text_features);
    }

    void decode(std::vector<std::vector<float>> &image_features, std::vector<std::vector<float>> &text_features,
                std::vector<std::vector<float>> &logits_per_image, std::vector<std::vector<float>> &logits_per_text)
    {
        postprocess(image_features, text_features, logits_per_image, logits_per_text);
    }
};
