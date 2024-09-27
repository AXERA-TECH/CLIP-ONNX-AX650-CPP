#pragma once
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <thread>

#include "onnxruntime_cxx_api.h"

#include "sample_log.h"
#include "Tokenizer.hpp"

#ifndef EN_TEXT_TOKEN_LEN
#define EN_TEXT_TOKEN_LEN 77
#endif

#ifndef ZH_TEXT_TOKEN_LEN
#define ZH_TEXT_TOKEN_LEN 52
#endif

class CLIPTextEncoder
{
protected:
    std::shared_ptr<TokenizerBase> tokenizer;

    std::vector<float> text_features_input;
    std::vector<int> text_tokens_input;

    bool _isCN = false;
    int LEN_TEXT_FEATURE = 512;
    int LEN_TEXT_TOKEN = EN_TEXT_TOKEN_LEN;

public:
    virtual bool load_text_encoder(std::string encoder_path) = 0;
    virtual bool encode(std::vector<std::string> &texts, std::vector<std::vector<float>> &text_features) = 0;
    int get_text_feature_size()
    {
        return LEN_TEXT_FEATURE;
    }

    bool load_tokenizer(std::string vocab_path, bool isCN)
    {
        std::ifstream fs(vocab_path);
        if (!fs.good())
        {
            ALOGE("vocab file open failed %s", vocab_path.c_str());
            return false;
        }
        fs.close();

        _isCN = isCN;
        if (isCN)
        {
            LEN_TEXT_TOKEN = ZH_TEXT_TOKEN_LEN;
            tokenizer.reset(new TokenizerClipChinese);
        }
        else
        {
            tokenizer.reset(new TokenizerClip);
        }
        ALOGI("text token len %d", LEN_TEXT_TOKEN);
        text_tokens_input = std::vector<int>(1024 * LEN_TEXT_TOKEN);
        return tokenizer->load_tokenize(vocab_path);
    }
};

class CLIPTextEncoderOnnx : public CLIPTextEncoder
{
protected:
    std::string device{"cpu"};
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::shared_ptr<Ort::Session> TextEncoderSession;
    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    const char
        *TextEncInputNames[1]{"texts"},
        *TextEncOutputNames[1]{"text_features"};

public:
    CLIPTextEncoderOnnx()
    {
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "CLIP_DECODER");
        session_options = Ort::SessionOptions();
        session_options.SetInterOpNumThreads(std::thread::hardware_concurrency());
        session_options.SetIntraOpNumThreads(std::thread::hardware_concurrency());
        // 设置图像优化级别
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }

    bool load_text_encoder(std::string encoder_path) override
    {
        TextEncoderSession.reset(new Ort::Session(env, encoder_path.c_str(), session_options));
        if (TextEncoderSession->GetInputCount() != 1 || TextEncoderSession->GetOutputCount() != 1)
        {
            ALOGE("Model not loaded (invalid input/output count)");
            return false;
        }
        auto shape = TextEncoderSession->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        LEN_TEXT_FEATURE = shape[1];
        ALOGI("text feature len %d", LEN_TEXT_FEATURE);
        text_features_input = std::vector<float>(1024 * LEN_TEXT_FEATURE);
        return true;
    }

    bool encode(std::vector<std::string> &texts, std::vector<std::vector<float>> &text_features) override
    {
        std::vector<std::vector<int>> text_token;
        text_token.resize(texts.size());
        for (size_t i = 0; i < texts.size(); i++)
        {
            tokenizer->encode_text(texts[i], text_token[i]);
        }

        if (text_token.size() * LEN_TEXT_TOKEN > text_tokens_input.size())
        {
            text_tokens_input.resize(text_token.size() * LEN_TEXT_TOKEN);
        }

        memset(text_tokens_input.data(), 0, text_token.size() * LEN_TEXT_TOKEN * sizeof(int));
        auto text_tokens_input_ptr = text_tokens_input.data();
        for (size_t i = 0; i < text_token.size(); i++)
        {
            if (text_token[i].size() > LEN_TEXT_TOKEN)
            {
                ALOGW("text_features index %ld ,bigger than %d\n", i, LEN_TEXT_TOKEN);
                return false;
            }
            memcpy(text_tokens_input_ptr + i * LEN_TEXT_TOKEN, text_token[i].data(), text_token[i].size() * sizeof(int));
        }

        if (_isCN)
        {
            std::vector<int64_t> text_token_shape = {1, LEN_TEXT_TOKEN};
            text_features.resize(text_token.size());

            std::vector<int64_t> text_tokens_input_64(texts.size() * LEN_TEXT_TOKEN);
            for (size_t i = 0; i < text_tokens_input_64.size(); i++)
            {
                text_tokens_input_64[i] = text_tokens_input[i];
            }

            for (size_t i = 0; i < text_token.size(); i++)
            {
                auto inputTensor = Ort::Value::CreateTensor<int64_t>(
                    memory_info_handler, text_tokens_input_64.data() + i * LEN_TEXT_TOKEN, LEN_TEXT_TOKEN, text_token_shape.data(), text_token_shape.size());

                Ort::RunOptions runOptions;
                auto OutputTensors = TextEncoderSession->Run(runOptions, TextEncInputNames, &inputTensor,
                                                             1, TextEncOutputNames, 1);
                auto &text_features_tensor = OutputTensors[0];
                auto text_features_tensor_ptr = text_features_tensor.GetTensorMutableData<float>();

                text_features[i].resize(LEN_TEXT_FEATURE);
                memcpy(text_features[i].data(), text_features_tensor_ptr, LEN_TEXT_FEATURE * sizeof(float));
            }
        }
        else
        {
            std::vector<int64_t> text_token_shape = {(int64_t)text_token.size(), LEN_TEXT_TOKEN};

            auto inputTensor = (Ort::Value::CreateTensor<int>(
                memory_info_handler, text_tokens_input.data(), text_tokens_input.size(), text_token_shape.data(), text_token_shape.size()));

            Ort::RunOptions runOptions;
            auto OutputTensors = TextEncoderSession->Run(runOptions, TextEncInputNames, &inputTensor,
                                                         1, TextEncOutputNames, 1);
            auto &text_features_tensor = OutputTensors[0];
            auto text_features_tensor_ptr = text_features_tensor.GetTensorMutableData<float>();
            auto output_shape = text_features_tensor.GetTensorTypeAndShapeInfo().GetShape();

            text_features.resize(output_shape[0]);

            for (size_t i = 0; i < text_features.size(); i++)
            {
                text_features[i].resize(output_shape[1]);
                memcpy(text_features[i].data(), text_features_tensor_ptr + i * output_shape[1], output_shape[1] * sizeof(float));
            }
        }

        return true;
    }
};
