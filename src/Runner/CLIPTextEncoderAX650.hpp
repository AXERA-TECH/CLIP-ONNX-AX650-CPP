#pragma once
#include "CLIPTextEncoder.hpp"
#include "ax_model_runner_ax650.hpp"

class CLIPTextEncoderAX650 : public CLIPTextEncoder
{
private:
    std::shared_ptr<ax_runner_base> m_encoder;

public:
    bool load_text_encoder(std::string encoder_path) override
    {
        m_encoder.reset(new ax_runner_ax650);
        m_encoder->init(encoder_path.c_str());
        LEN_TEXT_FEATURE = m_encoder->get_output(0).vShape[m_encoder->get_output(0).vShape.size() - 1];
        ALOGI("text feature len %d", LEN_TEXT_FEATURE);
        return true;
    }

    template <typename T>
    void fill_ids(T *data, int len, std::vector<int> &text_token)
    {
        memset(data, 0, len * sizeof(T));
        for (int i = 0; i < len; i++)
        {
            data[i] = text_token[i];
        }
    }

    bool encode(std::vector<std::string> &texts, std::vector<std::vector<float>> &text_features) override
    {
        if (m_encoder == nullptr)
        {
            return false;
        }
        text_features.resize(texts.size());
        for (size_t i = 0; i < texts.size(); i++)
        {
            std::vector<int> text_token;
            tokenizer->encode_text(texts[i], text_token);
            if (text_token.size() > LEN_TEXT_TOKEN)
            {
                ALOGW("the text of \"%s\" token bigger than %d\n", texts[i].c_str(), LEN_TEXT_TOKEN);
                return false;
            }

            // if (_isCN)
            // {
            //     fill_ids((int64_t *)m_encoder->get_input(0).pVirAddr, text_token.size(), text_token);
            // }
            // else
            // {
            fill_ids((int32_t *)m_encoder->get_input(0).pVirAddr, text_token.size(), text_token);
            // }
            m_encoder->inference();
            text_features[i].resize(LEN_TEXT_FEATURE);
            // m_encoder->mem_sync_output(0);
            memcpy(text_features[i].data(), m_encoder->get_output(0).pVirAddr, LEN_TEXT_FEATURE * sizeof(float));
        }

        return true;
    }
};
