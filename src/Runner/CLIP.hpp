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

    static void softmax(const std::vector<std::vector<float>> &input, std::vector<std::vector<float>> &output)
    {
        output.clear();
        output.reserve(input.size());

        for (const auto &row : input)
        {
            float maxVal = *std::max_element(row.begin(), row.end());
            std::vector<float> expRow;
            float sum = 0.0f;
            for (float val : row)
            {
                float e = std::exp(val - maxVal); // 防止溢出
                expRow.push_back(e);
                sum += e;
            }

            for (float &val : expRow)
                val /= sum;

            output.push_back(std::move(expRow));
        }
    }

    static void postprocess(
        const std::vector<std::vector<float>> &imageFeatures,
        const std::vector<std::vector<float>> &textFeatures,
        std::vector<std::vector<float>> &logits_per_image,
        std::vector<std::vector<float>> &logits_per_text)
    {
        const float logit_scale = 100.0f;

        // Step 1: Normalize image features
        std::vector<std::vector<float>> imageNormed;
        imageNormed.reserve(imageFeatures.size());
        for (const auto &img : imageFeatures)
        {
            float norm = 0.0f;
            for (float v : img)
                norm += v * v;
            norm = std::sqrt(norm);

            std::vector<float> normVec;
            normVec.reserve(img.size());
            for (float v : img)
                normVec.push_back(v / norm);

            imageNormed.push_back(std::move(normVec));
        }

        // Step 2: Normalize text features
        std::vector<std::vector<float>> textNormed;
        textNormed.reserve(textFeatures.size());
        for (const auto &txt : textFeatures)
        {
            float norm = 0.0f;
            for (float v : txt)
                norm += v * v;
            norm = std::sqrt(norm);

            std::vector<float> normVec;
            normVec.reserve(txt.size());
            for (float v : txt)
                normVec.push_back(v / norm);

            textNormed.push_back(std::move(normVec));
        }

        // Step 3: Compute logits_per_image = image @ text^T
        std::vector<std::vector<float>> logitsPerImage;
        logitsPerImage.reserve(imageNormed.size());

        for (const auto &imgVec : imageNormed)
        {
            std::vector<float> row;
            row.reserve(textNormed.size());
            for (const auto &txtVec : textNormed)
            {
                float dot = 0.0f;
                for (size_t i = 0; i < imgVec.size(); ++i)
                    dot += imgVec[i] * txtVec[i];
                row.push_back(logit_scale * dot);
            }
            logitsPerImage.push_back(std::move(row));
        }

        // Step 4: Transpose logitsPerImage to get logitsPerText
        std::vector<std::vector<float>> logitsPerText(textNormed.size(), std::vector<float>(imageNormed.size()));
        for (size_t i = 0; i < logitsPerImage.size(); ++i)
            for (size_t j = 0; j < logitsPerImage[i].size(); ++j)
                logitsPerText[j][i] = logitsPerImage[i][j];

        // Step 5: Apply softmax
        softmax(logitsPerImage, logits_per_image); // shape (N_img, N_txt)
        softmax(logitsPerText, logits_per_text);   // shape (N_txt, N_img)
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
