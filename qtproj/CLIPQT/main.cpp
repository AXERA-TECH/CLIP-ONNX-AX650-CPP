#include "mainwindow.h"

#include <QApplication>
#include "style/DarkStyle.h"

#include "clip/cmdline.hpp"
#include "axcl.h"

int main(int argc, char *argv[])
{
    std::string image_src;
    std::string vocab_path;
    std::string image_encoder_model_path;
    std::string text_encoder_model_path;
    int language = 0;

    cmdline::parser cmd;
    cmd.add<std::string>("ienc", 0, "encoder model(onnx model or axmodel)", true, image_encoder_model_path);
    cmd.add<std::string>("tenc", 0, "text encoder model(onnx model or axmodel)", true, text_encoder_model_path);
    cmd.add<std::string>("image", 'i', "image file or folder(jpg png etc....)", true, image_src);
    cmd.add<std::string>("vocab", 'v', "vocab path", true, vocab_path);
    cmd.add<int>("language", 'l', "language choose, 0:english 1:chinese", true, 0);

    cmd.parse_check(argc, argv);

    axclInit(0);

    vocab_path = cmd.get<std::string>("vocab");
    image_encoder_model_path = cmd.get<std::string>("ienc");
    text_encoder_model_path = cmd.get<std::string>("tenc");
    image_src = cmd.get<std::string>("image");
    language = cmd.get<int>("language");

    QApplication a(argc, argv);
    QApplication::setStyle(new DarkStyle);
    MainWindow w(image_src,
                 vocab_path,
                 image_encoder_model_path,
                 text_encoder_model_path,
                 language);
    w.show();
    auto ret = a.exec();
    axclFinalize();
    return ret;
}
