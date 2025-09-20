@echo off
chcp 65001
echo ========================================
echo    YOLO骨架识别应用 - Windows启动器
echo ========================================
echo.

echo 正在检查Python环境...
python --version
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.7+
    pause
    exit /b 1
)

echo.
echo 正在检查依赖包...
python -c "import cv2, numpy, ultralytics" 2>nul
if errorlevel 1 (
    echo 正在安装依赖包...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 依赖包安装失败，请手动运行: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo.
echo 启动应用...
python run_app.py

pause
