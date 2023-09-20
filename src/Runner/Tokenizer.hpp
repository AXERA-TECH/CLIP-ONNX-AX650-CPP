#pragma once
#include "map"
#include "vector"
#include "string"
#include "fstream"
#include "iostream"

class TokenizerBase
{
protected:
    std::map<std::string, int> tokenizer_token2idx;

public:
    virtual bool load_tokenize(std::string vocab_path) = 0;
    virtual void encode_text(std::string text, std::vector<int> &idx) = 0;
};

class TokenizerClip : public TokenizerBase
{
protected:
    std::vector<std::string> stringSplit(const std::string &str, char delim)
    {
        std::vector<std::string> elems;
        auto lastPos = str.find_first_not_of(delim, 0);
        auto pos = str.find_first_of(delim, lastPos);
        while (pos != std::string::npos || lastPos != std::string::npos)
        {
            elems.push_back(str.substr(lastPos, pos - lastPos));
            lastPos = str.find_first_not_of(delim, pos);
            pos = str.find_first_of(delim, lastPos);
        }
        return elems;
    }

    void tokenize(std::string token, std::vector<int> &idx)
    {
        idx.push_back(49406);
        {
            std::vector<std::string> tokens = stringSplit(token, ' ');
            for (auto t : tokens)
            {
                idx.push_back(tokenizer_token2idx[t + "</w>"]);
            }
        }
        idx.push_back(49407);

        // memset(feat, 0, sizeof(CLIP_TEXT_FEATURE_T));
        // memcpy(feat->feature, idx.data(), idx.size() * sizeof(int));
    }

public:
    bool load_tokenize(std::string vocab_path) override
    {
        std::ifstream infile;
        infile.open(vocab_path.data());
        if (!infile.good())
        {
            return false;
        }

        std::string s;
        int idx = 0;
        while (getline(infile, s))
        {
            tokenizer_token2idx.insert(std::pair<std::string, int>(s, idx));
            idx++;
        }
        infile.close();
        return true;
    }

    void encode_text(std::string text, std::vector<int> &idx) override
    {
        idx.clear();
        return tokenize(text, idx);
    }
};

class TokenizerClipChinese : public TokenizerClip
{
public:
    bool load_tokenize(std::string vocab_path) override
    {
        std::ifstream infile;
        infile.open(vocab_path.data());
        if (!infile.good())
        {
            return false;
        }

        std::string s;
        int idx = 0;
        while (getline(infile, s))
        {
            // printf("%s\n", s.c_str());
            tokenizer_token2idx.insert(std::pair<std::string, int>(s, idx));
            idx++;
        }
        infile.close();
        return true;
    }

    void encode_text(std::string text, std::vector<int> &idx) override
    {
#define CLS 101
#define SEP 102
        ALOGD("%s\n", text.c_str());
        idx.clear();
        idx.push_back(CLS);
        {
            std::vector<std::string> tokens = stringSplit(text, ' ');
            for (auto t : tokens)
            {
                if (tokenizer_token2idx.count(t) > 0)
                {
                    idx.push_back(tokenizer_token2idx[t]);
                }
                else
                {
                    for (size_t i = 0; i < t.length();)
                    {
                        int cplen = 1;
                        if ((t[i] & 0xf8) == 0xf0)
                            cplen = 4; // 占用4个字节，前5位为11110
                        else if ((t[i] & 0xf0) == 0xe0)
                            cplen = 3; // 占用3个字节，前4位为1110
                        else if ((t[i] & 0xe0) == 0xc0)
                            cplen = 2; // 占用2个字节，前3位为110
                        // 个人感觉这行代码好像没什么用，如果三种情况都不符合，那么cplen就为初始化的0，是符合utf-8编码定义的
                        if ((i + cplen) > t.length())
                            cplen = 1;
                        auto tmp = t.substr(i, cplen);
                        i += cplen;
                        idx.push_back(tokenizer_token2idx[tmp]);
                        // std::cout << idx[idx.size() - 1] << std::endl;
                    }
                }
            }
        }
        idx.push_back(SEP);
        return;
    }
};
