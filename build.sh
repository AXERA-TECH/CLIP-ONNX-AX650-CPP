#!/bin/bash

build_chip=$1  # ax650 ax630c ax620q
BSP_MSP_DIR=$PWD/ax20e_bsp/ax620e_arm32/msp/out


if [ "${build_chip}" = "ax630c" ] || [ "${build_chip}" = "ax650" ] 
then
    echo "aarch64"

    mkdir build_aarch64
    cd build_aarch64

    opencv_aarch64_url=https://github.com/ZHEQIUSHUI/assets/releases/download/ax650/libopencv-4.5.5-aarch64.zip
    if [ ! -f "libopencv-4.5.5-aarch64.zip" ]; then
        # Download the file
        echo "Downloading $opencv_aarch64_url"
        wget "$opencv_aarch64_url" -O "libopencv-4.5.5-aarch64.zip"
    else
        echo "libopencv-4.5.5-aarch64.zip already exists"
    fi

    # Check if the folder exists
    if [ ! -d "libopencv-4.5.5-aarch64" ]; then
        # Extract the file
        echo "Extracting unzip libopencv-4.5.5-aarch64.zip"
        unzip libopencv-4.5.5-aarch64.zip
    else
        echo "libopencv-4.5.5-aarch64 already exists"
    fi


    onnxruntime_aarch64_url=https://github.com/ZHEQIUSHUI/SAM-ONNX-AX650-CPP/releases/download/ax_models/onnxruntime-aarch64-none-gnu-1.16.0.zip
    if [ ! -f "onnxruntime-aarch64-none-gnu-1.16.0.zip" ]; then
        # Download the file
        echo "Downloading $onnxruntime_aarch64_url"
        wget "$onnxruntime_aarch64_url" -O "onnxruntime-aarch64-none-gnu-1.16.0.zip"
    else 
        echo "onnxruntime-aarch64-none-gnu-1.16.0.zip already exists"
    fi

    # Check if the folder exists
    if [ ! -d "onnxruntime-aarch64-none-gnu-1.16.0" ]; then
        # Extract the file
        echo "Extracting unzip onnxruntime-aarch64-none-gnu-1.16.0.zip"
        unzip onnxruntime-aarch64-none-gnu-1.16.0.zip
    else
        echo "onnxruntime-aarch64-none-gnu-1.16.0 already exists"
    fi

    # 下载失败可以使用其他方式下载并放到在 $build_dir 目录，参考如下命令解压
    URL="https://developer.arm.com/-/media/Files/downloads/gnu-a/9.2-2019.12/binrel/gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu.tar.xz"
    FOLDER="gcc-arm-9.2-2019.12-x86_64-aarch64-none-linux-gnu"

    aarch64-none-linux-gnu-gcc -v
    if [ $? -ne 0 ]; then
        # Check if the file exists
        if [ ! -f "$FOLDER.tar.xz" ]; then
            # Download the file
            echo "Downloading $URL"
            wget "$URL" -O "$FOLDER.tar.xz"
        else
            echo "$FOLDER.tar.xz already exists"
        fi

        # Check if the folder exists
        if [ ! -d "$FOLDER" ]; then
            # Extract the file
            echo "Extracting $FOLDER.tar.xz"
            tar -xf "$FOLDER.tar.xz"
        else
            echo "$FOLDER already exists"
        fi

        export PATH=$PATH:$PWD/$FOLDER/bin/
        aarch64-none-linux-gnu-gcc -v
        if [ $? -ne 0 ]; then
            echo "Error: aarch64-none-linux-gnu-gcc not found"
            exit 1
        fi
    else
        echo "aarch64-none-linux-gnu-gcc already exists"
    fi

    if [ "${build_chip}" = "ax630c"]
    then
        BSP_MSP_DIR=$BSP_MSP_DIR/arm64_glibc
    fi
    echo "bsp dir: ${BSP_MSP_DIR}"

    cmake -DBSP_MSP_DIR=${BSP_MSP_DIR} \
    -DCMAKE_TOOLCHAIN_FILE=../toolchains/aarch64-none-linux-gnu.toolchain.cmake \
    -DBUILD_WITH_AX650=ON \
    -DONNXRUNTIME_DIR=$PWD/onnxruntime-aarch64-none-gnu-1.16.0 \
    -DOpenCV_DIR=$PWD/libopencv-4.5.5-aarch64/lib/cmake/opencv4 \
    -DCMAKE_BUILD_TYPE=Release \
    ..

    make -j16
    make install

elif [ "${build_chip}" = "ax620q" ] 
then
    echo "uclibc"

    mkdir build_uclibc
    cd build_uclibc

    BSP_MSP_DIR=$BSP_MSP_DIR/arm_uclibc

    opencv_uclibc_url=https://github.com/AXERA-TECH/ax-samples/releases/download/v0.6/opencv-arm-uclibc-linux.zip
    if [ ! -f "opencv-arm-uclibc-linux.zip" ]; then
        # Download the file
        echo "Downloading $opencv_uclibc_url"
        wget "$opencv_uclibc_url" -O "opencv-arm-uclibc-linux.zip"
    else
        echo "opencv-arm-uclibc-linux.zip already exists"
    fi

    # Check if the folder exists
    if [ ! -d "opencv-arm-uclibc-linux" ]; then
        # Extract the file
        echo "Extracting unzip opencv-arm-uclibc-linux.zip"
        unzip opencv-arm-uclibc-linux.zip
    else
        echo "opencv-arm-uclibc-linux already exists"
    fi

    # onnxruntime 没有编译


    URL="https://github.com/AXERA-TECH/ax620q_bsp_sdk/releases/download/v2.0.0/arm-AX620E-linux-uclibcgnueabihf_V3_20240320.tgz"
    FOLDER="arm-AX620E-linux-uclibcgnueabihf"

    arm-AX620E-linux-uclibcgnueabihf-gcc -v
    if [ $? -ne 0 ]; then
        # Check if the file exists
        if [ ! -f "arm-AX620E-linux-uclibcgnueabihf_V3_20240320.tgz" ]; then
            # Download the file
            echo "Downloading $URL"
            wget "$URL"
        else
            echo "arm-AX620E-linux-uclibcgnueabihf_V3_20240320.tgz already exists"
        fi

        # Check if the folder exists
        if [ ! -d "$FOLDER" ]; then
            # Extract the file
            echo "Extracting arm-AX620E-linux-uclibcgnueabihf_V3_20240320.tgz"
            tar -xf "arm-AX620E-linux-uclibcgnueabihf_V3_20240320.tgz"
        else
            echo "$FOLDER already exists"
        fi

        export PATH=$PATH:$PWD/$FOLDER/bin/
        arm-AX620E-linux-uclibcgnueabihf-gcc -v
        if [ $? -ne 0 ]; then
            echo "Error: arm-AX620E-linux-uclibcgnueabihf-gcc not found"
            exit 1
        fi
    else
        echo "arm-AX620E-linux-uclibcgnueabihf-gcc already exists"
    fi

    cmake -DBSP_MSP_DIR=${BSP_MSP_DIR} \
    -DCMAKE_TOOLCHAIN_FILE=../toolchains/arm-AX620E-linux-uclibcgnueabihf.cmake \
    -DBUILD_WITH_AX620E=ON \
    -DOpenCV_DIR=$PWD/opencv-arm-uclibc-linux/lib/cmake/opencv4 \
    -DCMAKE_BUILD_TYPE=Release \
    ..

    make -j16
    make install

else
    echo "Error: build type is invalid"
    exit 1
fi
