// #include "Runner/CLIPAX650.hpp"
// #include "Runner/CLIPOnnx.hpp"
#include <memory>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "string_utility.hpp"
#include "cmdline.hpp"
#include "sample_log.h"
#include "ax_model_runner_ax650.hpp"

class ClipDecoder
{
public:
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

    static void forward(
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
};

class CLIPAX650
{
private:
    std::shared_ptr<ax_runner_base> m_encoder;
    cv::Mat input;
    int input_height;
    int input_width;
    int LEN_IMAGE_FEATURE;

public:
    bool load_image_encoder(std::string encoder_path)
    {
        m_encoder.reset(new ax_runner_ax650);
        m_encoder->init(encoder_path.c_str());
        input_height = m_encoder->get_algo_height();
        input_width = m_encoder->get_algo_width();
        ALOGI("input size %d %d", input_height, input_width);
        input = cv::Mat(input_height, input_width, CV_8UC3, m_encoder->get_input(0).pVirAddr);

        LEN_IMAGE_FEATURE = m_encoder->get_output(0).vShape[1];
        ALOGI("image feature len %d", LEN_IMAGE_FEATURE);
        return true;
    }
    void encode(cv::Mat image, std::vector<float> &image_features)
    {
        if (!m_encoder.get())
        {
            ALOGE("encoder not init");
            return;
        }
        cv::resize(image, input, cv::Size(input_width, input_height));
        cv::cvtColor(input, input, cv::COLOR_BGR2RGB);
        auto ret = m_encoder->inference();

        image_features.resize(LEN_IMAGE_FEATURE);
        memcpy(image_features.data(), m_encoder->get_output(0).pVirAddr, LEN_IMAGE_FEATURE * sizeof(float));
    }
};

bool _file_exist(const std::string &path)
{
    auto flag = false;

    std::fstream fs(path, std::ios::in | std::ios::binary);
    flag = fs.is_open();
    fs.close();

    return flag;
}

bool _file_read(const std::string &path, std::vector<char> &data)
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

bool _file_dump(const std::string &path, char *data, int size)
{
    std::fstream fs(path, std::ios::out | std::ios::binary);

    if (!fs.is_open() || fs.fail())
    {
        fprintf(stderr, "[ERR] cannot open file %s \n", path.c_str());
    }

    fs.write(data, size);

    return true;
}

void process_texts(std::vector<std::string> &lines, std::vector<std::string> &texts, std::vector<std::vector<float>> &text_features)
{
    for (size_t i = 0; i < lines.size(); i++)
    {
        auto &s = lines[i];
        std::vector<std::string> ctxs = string_utility<std::string>::split(s, ":");
        if (ctxs.size() == 2)
        {
            if (_file_exist(ctxs[1]))
            {
                std::vector<char> tmp_c_text_feature;
                _file_read(ctxs[1], tmp_c_text_feature);

                std::vector<float> tmp_text_features;
                tmp_text_features.resize(tmp_c_text_feature.size() / sizeof(float));
                memcpy(tmp_text_features.data(), tmp_c_text_feature.data(), tmp_c_text_feature.size());

                texts.push_back(ctxs[0]);
                text_features.push_back(tmp_text_features);
                ALOGI("read text feature [%s] from %s", ctxs[0].c_str(), ctxs[1].c_str());
            }
            else
            {
                ALOGE("text file not exist, %s", ctxs[1].c_str());
            }
        }
        else
        {
            ALOGE("text format error, %s", s.c_str());
        }
        /* code */
    }
}

int main(int argc, char *argv[])
{
    std::string image_src;
    std::string text_src;
    std::string image_encoder_model_path;

    cmdline::parser cmd;
    cmd.add<std::string>("ienc", 0, "encoder model(onnx model or axmodel)", true, image_encoder_model_path);
    cmd.add<std::string>("image", 'i', "image file or folder(jpg png etc....)", true, image_src);
    cmd.add<std::string>("text", 't', "txt file with content like \n\n%s    dog:text_feat/dog.bin\n    bird:text_feat/bird.bin\n    cat:text_feat/cat.bin\n", true, text_src);

    cmd.parse_check(argc, argv);

    image_encoder_model_path = cmd.get<std::string>("ienc");

    std::shared_ptr<CLIPAX650> mClip;
    if (string_utility<std::string>::ends_with(image_encoder_model_path, ".axmodel"))
    {
        mClip.reset(new CLIPAX650);
    }
    else
    {
        fprintf(stderr, "no impl for %s\n", image_encoder_model_path.c_str());
        return -1;
    }

    mClip->load_image_encoder(image_encoder_model_path);
    {
        ALOGI("if you dont want to load text encoder, the '--text' args must be set like '--text dog:dog.bin', or set a txt file with content like \n\n%s    dog:dog.bin\n    bird:bird.bin\n    cat:cat.bin\n", MACRO_RED);
        ALOGI("and make sure the '.bin' file exist\n");
    }

    image_src = cmd.get<std::string>("image");
    text_src = cmd.get<std::string>("text");

    std::vector<std::string> lines, texts;
    std::vector<std::vector<float>> text_features;
    if (string_utility<std::string>::ends_with(text_src, ".txt"))
    {
        std::ifstream infile;
        infile.open(text_src);
        if (!infile.good())
        {
            ALOGE("");
            return -1;
        }

        std::string s;
        while (getline(infile, s))
        {
            lines.push_back(s);
        }
        infile.close();
    }
    else
    {
        lines.push_back(text_src);
    }

    process_texts(lines, texts, text_features);

    // auto time_start = std::chrono::high_resolution_clock::now();
    // mClip->encode(texts, text_features);
    // auto time_end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> diff = time_end - time_start;
    // std::cout << "encode text Inference Cost time : " << diff.count() << "s" << std::endl;

    std::vector<std::vector<float>> image_features;
    std::vector<std::string> image_paths;
    cv::Mat src = cv::imread(image_src);
    if (src.data)
    {
        std::vector<float> feat;
        mClip->encode(src, feat);
        image_features.push_back(feat);
        image_paths.push_back(image_src);
    }
    else
    {
        if (!string_utility<std::string>::ends_with(image_src, "/") &&
            !string_utility<std::string>::ends_with(image_src, "\\"))
        {
            image_src += "/";
        }
        std::vector<std::string> image_list;
        cv::glob(image_src + "*.*", image_list);

        for (size_t i = 0; i < image_list.size(); i++)
        {
            std::string image_path = image_list[i];
            src = cv::imread(image_path);
            if (!src.data)
                continue;
            std::vector<float> feat;
            auto time_start = std::chrono::high_resolution_clock::now();
            mClip->encode(src, feat);
            auto time_end = std::chrono::high_resolution_clock::now();
            auto diff = time_end - time_start;
            std::cout << "image encode cost time : " << std::chrono::duration<double>(diff).count() << "s" << std::endl;
            image_features.push_back(feat);
            image_paths.push_back(image_path);
        }
    }

    std::vector<std::vector<float>> logits_per_image, logits_per_text;
    auto time_start = std::chrono::high_resolution_clock::now();
    ClipDecoder::forward(image_features, text_features, logits_per_image, logits_per_text);
    auto time_end = std::chrono::high_resolution_clock::now();
    auto diff = time_end - time_start;
    std::cout << "postprocess cost time : " << std::chrono::duration<double>(diff).count() << "s" << std::endl;

    printf("\n");
    if (texts.size() > 1)
    {
        printf("per image:\n");
        printf("%32s|", "image path\\text");
        for (size_t i = 0; i < texts.size(); i++)
        {
            printf("%32s|", texts[i].c_str());
        }
        printf("\n");
        for (size_t i = 0; i < logits_per_image.size(); i++)
        {
            printf("%32s|", image_paths[i].c_str());
            for (size_t j = 0; j < logits_per_image[i].size(); j++)
            {
                printf("%32.2f|", logits_per_image[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    printf("\n");
    printf("per text:\n");
    printf("%32s|", "text\\image path");
    for (size_t i = 0; i < image_paths.size(); i++)
    {
        printf("%32s|", image_paths[i].c_str());
    }
    printf("\n");
    for (size_t i = 0; i < logits_per_text.size(); i++)
    {
        printf("%32s|", texts[i].c_str());
        for (size_t j = 0; j < logits_per_text[i].size(); j++)
        {
            printf("%32.2f|", logits_per_text[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    return 0;
}
