#pragma once
#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <thread>

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
