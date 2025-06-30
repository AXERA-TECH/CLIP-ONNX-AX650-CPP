# CLIP

https://github.com/AXERA-TECH/CLIP-ONNX-AX650-CPP/assets/46700201/7fefc9dd-9168-462d-bae9-bb013731f5c6

## Build with axcl
```
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-none-linux-gnu.toolchain.cmake -DOpenCV_DIR=${opencv_cmake_file_dir} -DBSP_MSP_DIR=/path/to/msp/out/ ..
make -j4
```
aarch64-none-gnu library:\
[opencv](https://github.com/ZHEQIUSHUI/SAM-ONNX-AX650-CPP/releases/download/ax_models/libopencv-4.6-aarch64-none.zip)

### Get model

[cnclip](https://hf-mirror.com/AXERA-TECH/cnclip)
[clip](https://hf-mirror.com/AXERA-TECH/clip)

## Run with axcl 
```shell
# ./main --ienc cnclip/cnclip_vit_l14_336px_vision_u16u8.axmodel --tenc cnclip/cnclip_vit_l14_336px_text_u16.axmodel -i ../images/ -t text.txt -v ../cn_vocab.txt -l 1
[I][              load_image_encoder][  18]: input size 336 336
[I][              load_image_encoder][  22]: image feature len 768
[I][               load_text_encoder][  16]: text feature len 768
[I][                  load_tokenizer][  60]: text token len 52
[I][                   process_texts][  71]: encode text [bird] cost time : 0.004662
[I][                   process_texts][  71]: encode text [cat] cost time : 0.004529
[I][                   process_texts][  71]: encode text [dog] cost time : 0.004484
image encode cost time : 0.101203s
image encode cost time : 0.0918216s
image encode cost time : 0.094925s
postprocess cost time : 0.000195515s

per image:
                         image path\text|                                    bird|                                     cat|                                     dog|
                      ../images/bird.jpg|                                    1.00|                                    0.00|                                    0.00|
                       ../images/cat.jpg|                                    0.00|                                    1.00|                                    0.00|
                 ../images/dog-chai.jpeg|                                    0.00|                                    0.00|                                    1.00|


per text:
                         text\image path|                      ../images/bird.jpg|                       ../images/cat.jpg|                 ../images/dog-chai.jpeg|
                                    bird|                                    1.00|                                    0.00|                                    0.00|
                                     cat|                                    0.00|                                    1.00|                                    0.00|
                                     dog|                                    0.00|                                    0.01|                                    0.99|
```
## Reference
[CLIP](https://github.com/openai/CLIP)\
[Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)\
[CLIP-ImageSearch-NCNN](https://github.com/EdVince/CLIP-ImageSearch-NCNN)
