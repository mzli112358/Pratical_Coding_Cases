#!/bin/bash

echo "========================================"
echo "   YOLO骨架识别应用 - Linux/Mac启动器"
echo "========================================"
echo

echo "正在检查Python环境..."
python3 --version
if [ $? -ne 0 ]; then
    echo "错误: 未找到Python3，请先安装Python 3.7+"
    exit 1
fi

echo
echo "正在检查依赖包..."
python3 -c "import cv2, numpy, ultralytics" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "正在安装依赖包..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "依赖包安装失败，请手动运行: pip3 install -r requirements.txt"
        exit 1
    fi
fi

echo
echo "启动应用..."
python3 run_app.py
