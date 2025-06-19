# CLIP

https://github.com/AXERA-TECH/CLIP-ONNX-AX650-CPP/assets/46700201/7fefc9dd-9168-462d-bae9-bb013731f5c6

## Build with axcl
```
mkdir build
cd build
cmake -DOpenCV_DIR=${opencv_cmake_file_dir} ..
make -j4
```
aarch64-none-gnu library:\
[opencv](https://github.com/ZHEQIUSHUI/SAM-ONNX-AX650-CPP/releases/download/ax_models/libopencv-4.6-aarch64-none.zip)

### Get model

[cnclip](https://hf-mirror.com/AXERA-TECH/cnclip)
[clip](https://hf-mirror.com/AXERA-TECH/clip)

## Run with axcl 
```
./main --ienc cnclip/cnclip_vit_l14_336px_vision_u16u8.axmodel --tenc cnclip/cnclip_vit_l14_336px_text_u16.axmodel -v ../cn_vocab.txt -l 1 -i cn_clip_models/images/ -t text.txt 

input size: 1
    name:    image [unknown] [unknown] 
        1 x 3 x 336 x 336


output size: 1
    name: unnorm_image_features 
        1 x 768

[I][              load_image_encoder][  18]: input size 336 336
[I][              load_image_encoder][  22]: image feature len 768

input size: 1
    name:     text [unknown] [unknown] 
        1 x 52


output size: 1
    name: unnorm_text_features 
        1 x 768

[I][               load_text_encoder][  16]: text feature len 768
[I][                  load_tokenizer][  60]: text token len 52
[I][                   process_texts][  70]: encode text [bird] cost time : 0.005517
[I][                   process_texts][  70]: encode text [cat] cost time : 0.005364
[I][                   process_texts][  70]: encode text [dog] cost time : 0.005266
image encode cost time : 0.0956998s
image encode cost time : 0.0952134s
image encode cost time : 0.0940913s
postprocess cost time : 4.2296e-05s

per image:
                         image path\text|                                    bird|                                     cat|                                     dog|
          cn_clip_models/images/bird.jpg|                                    1.00|                                    0.00|                                    0.00|
           cn_clip_models/images/cat.jpg|                                    0.00|                                    1.00|                                    0.00|
     cn_clip_models/images/dog-chai.jpeg|                                    0.00|                                    0.00|                                    1.00|


per text:
                         text\image path|          cn_clip_models/images/bird.jpg|           cn_clip_models/images/cat.jpg|     cn_clip_models/images/dog-chai.jpeg|
                                    bird|                                    1.00|                                    0.00|                                    0.00|
                                     cat|                                    0.00|                                    1.00|                                    0.00|
                                     dog|                                    0.00|                                    0.00|                                    1.00|
```
## Reference
[CLIP](https://github.com/openai/CLIP)\
[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)\
[CLIP-ImageSearch-NCNN](https://github.com/EdVince/CLIP-ImageSearch-NCNN)
