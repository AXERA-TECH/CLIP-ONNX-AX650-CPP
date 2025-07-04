#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "QLabel"
#include "QGridLayout"
#include "myqlabel.h"

#include "clip/string_utility.hpp"
#include "clip/cqdm.h"

#include "internal_func.hpp"

MainWindow::MainWindow(
    std::string image_src,
    std::string vocab_path,
    std::string image_encoder_model_path,
    std::string text_encoder_model_path,
    int language,
    QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow)
{
    mClip = std::make_shared<CLIP>();

    if (!mClip->load_image_encoder(image_encoder_model_path))
    {
        ALOGE("load image encoder failed");
        exit(-1);
    }
    if (!text_encoder_model_path.empty())
    {
        mClip->load_text_encoder(text_encoder_model_path);
    }
    else
    {
        // ALOGI("if you dont want to load text encoder, the '--text' args must be set like '--text dog:dog.bin', or set a txt file with content like \n\n%s    dog:dog.bin\n    bird:bird.bin\n    cat:cat.bin\n", MACRO_RED);
        // ALOGI("and make sure the '.bin' file exist\n");
        exit(-1);
    }

    if (!mClip->load_tokenizer(vocab_path, language == 1))
    {
        ALOGE("load tonkenizer failed");
        exit(-1);
    }

    if (!string_utility<std::string>::ends_with(image_src, "/") &&
        !string_utility<std::string>::ends_with(image_src, "\\"))
    {
        image_src += "/";
    }
    std::vector<std::string> image_list;
    cv::glob(image_src + "*.jpg", image_list);

    std::vector<std::string> image_list_png;
    cv::glob(image_src + "*.png", image_list_png);

    std::vector<std::string> image_list_jpeg;
    cv::glob(image_src + "*.jpeg", image_list_jpeg);

    image_list.insert(image_list.end(), image_list_png.begin(), image_list_png.end());
    image_list.insert(image_list.end(), image_list_jpeg.begin(), image_list_jpeg.end());

    ALOGI("totally image count %d", image_list.size());

    image_features.resize(image_list.size());
    image_paths.resize(image_list.size());

    auto tqdm = create_cqdm(image_list.size(), 40);

    int load_cnt = 0, gen_cnt = 0;
    std::vector<int> remove_ids;
    for (size_t i = 0; i < image_list.size(); i++)
    {
        std::string image_path = image_list[i];
        std::string feat_path = image_path + ".feat";
        std::vector<float> feat(mClip->get_image_feature_size(), 0);
        do
        {
            if (_file_exist(feat_path))
            {
                std::vector<char> tmp;
                _file_read(feat_path, tmp);
                if (tmp.size() != (mClip->get_image_feature_size() * 4))
                {
                    remove_ids.push_back(i);
                    ALOGE("%s not match model %s,please remove it and restart process", feat_path.c_str(), image_encoder_model_path.c_str());
                    break;
                }
                memcpy(feat.data(), tmp.data(), tmp.size());
                load_cnt++;
            }
            else
            {
                auto src = cv::imread(image_path);
                if (!src.data)
                {
                    remove_ids.push_back(i);
                    ALOGE("load image failed, %s", image_path.c_str());
                    break;
                }
                mClip->encode(src, feat);
                _file_dump(feat_path, (char *)feat.data(), feat.size() * 4);
                gen_cnt++;
            }
        } while (false);

        image_features[i] = feat;
        image_paths[i] = image_path;

        update_cqdm(&tqdm, i);
    }

    for (size_t i = 0; i < remove_ids.size(); i++)
    {
        auto idx = remove_ids[remove_ids.size() - i - 1];
        ALOGI("remove feature for %s", image_paths[idx].c_str());
        image_features.erase(image_features.begin() + idx);
        image_paths.erase(image_paths.begin() + idx);
    }

    ALOGI("load feat %d, gen feat %d", load_cnt, gen_cnt);

    ui->setupUi(this);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::add_image_text_label(QString image_path, QString text, int max_row)
{
    // two labels vertical layout
    myQLabel *image_label = new myQLabel(this);
    QLabel *text_label = new QLabel(this);

    // set image
    if (image_path.isEmpty())
    {
        image_label->setText("No image");
    }
    else
    {
        QImage image(image_path);
        if (image.bits())
            image_label->SetImage(image);
        else
            image_label->setText("No image");
    }
    // set text
    text_label->setText(text);

    // set color
    image_label->setStyleSheet("background-color: rgb(60, 60, 60);");
    text_label->setStyleSheet("background-color: rgb(90, 90, 90);");

    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->addWidget(image_label);
    layout->addWidget(text_label);
    layout->setStretchFactor(image_label, 999);
    layout->setStretchFactor(text_label, 1);

    ui->gridLayout->addLayout(layout, current_row, current_col);
    current_col++;
    if (current_col == max_row)
    {
        current_col = 0;
        current_row++;
    }
}

void MainWindow::clear_image_text_label()
{
    // clear all layouts
    QLayoutItem *child;
    while ((child = ui->gridLayout->takeAt(0)) != 0)
    {
        // clear all widgets in layout
        QLayout *layout = child->layout();
        if (layout != nullptr)
        {
            QLayoutItem *child2;
            while ((child2 = layout->takeAt(0)) != 0)
            {
                delete child2->widget();
                delete child2;
            }
        }
        delete child;
    }
    repaint();
}

void MainWindow::on_btn_search_clicked()
{
    if (ui->txt_context->text().isEmpty())
    {
        return;
    }
    clear_image_text_label();
    current_col = 0;
    current_row = 0;

    std::vector<std::string> texts;
    texts.push_back(ui->txt_context->text().toStdString());

    std::vector<std::vector<float>> text_features;
    auto time_start = std::chrono::high_resolution_clock::now();
    mClip->encode(texts, text_features);
    auto time_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = time_end - time_start;
    std::cout << "encode text Inference Cost time : " << diff.count() << "s" << std::endl;

    std::vector<std::vector<float>> logits_per_image, logits_per_text;
    time_start = std::chrono::high_resolution_clock::now();
    mClip->decode(image_features, text_features, logits_per_image, logits_per_text);
    time_end = std::chrono::high_resolution_clock::now();
    diff = time_end - time_start;
    std::cout << "matmul Inference Cost time : " << diff.count() << "s" << std::endl;

    struct path_and_score
    {
        std::string path;
        float score;
    };

    std::vector<path_and_score> results;

    if (logits_per_text.size() > 0)
    {
        for (size_t i = 0; i < logits_per_text[0].size(); i++)
        {
            if (logits_per_text[0][i] > 0)
            {
                results.push_back({image_paths[i], logits_per_text[0][i]});
            }
        }
    }
    ALOGI("there are %d results score bigger than 0", results.size());
    // sort by score
    std::sort(results.begin(), results.end(), [](const path_and_score &a, const path_and_score &b)
              { return a.score > b.score; });

    // count score>0.01
    double threshold = 0.01;
    bool isOk = false;
    threshold = ui->txt_threshold->text().toDouble(&isOk);
    if (!isOk)
    {
        threshold = 0.01;
    }
    ALOGI("threshold = %f", threshold);

    int count = 0;
    for (size_t i = 0; i < results.size(); i++)
    {
        if (results[i].score > threshold)
        {
            count++;
        }
    }
    count = MIN(count, max_display);
    int maxCols = ceil(sqrt((float)count));
    // int nRows = ((count % nCols) > 0) ? (count / nCols + 1) : (count / nCols);

    // show
    for (size_t i = 0; i < count && i < max_display; i++)
    {
        add_image_text_label(QString::fromStdString(results[i].path), QString::fromStdString(std::to_string(results[i].score)), maxCols);
    }
}

void MainWindow::on_txt_context_returnPressed()
{
    on_btn_search_clicked();
}
