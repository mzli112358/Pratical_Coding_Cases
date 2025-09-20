# OpenPose 安装和使用指南

## 问题解决

由于 `mediapipe` 在 Python 3.13 上存在兼容性问题，我们创建了一个基于 OpenCV 的替代方案。

## 环境设置

### 1. 创建虚拟环境

```bash
conda create -n openpose python=3.11 -y
conda activate openpose
```

### 2. 安装依赖

```bash
pip install -r requirements_opencv.txt
```

或者手动安装：

```bash
pip install opencv-python numpy matplotlib pillow
```

## 使用方法

### 基本使用

```bash
# 使用默认摄像头（ID=0）
python openpose_realtime_opencv.py

# 指定摄像头ID
python openpose_realtime_opencv.py --camera 1

# 使用预训练模型（如果有的话）
python openpose_realtime_opencv.py --model path/to/model.pb
```

### 操作说明

- **按 'q' 键**：退出程序
- **按 's' 键**：保存当前帧
- **按 'c' 键**：切换检测模式（OpenCV检测/简单轮廓检测）

## 功能特点

### 1. OpenCV 人体检测模式
- 使用 HOG 特征检测人体
- 在检测到的人体周围画框
- 适合快速人体检测

### 2. 简单轮廓检测模式
- 基于边缘检测和轮廓分析
- 显示人体轮廓
- 适合姿态分析

## 技术实现

### 依赖库
- **OpenCV**: 计算机视觉处理
- **NumPy**: 数值计算
- **Matplotlib**: 图像显示（可选）
- **Pillow**: 图像处理（可选）

### 核心功能
1. **实时摄像头捕获**
2. **人体检测和姿态估计**
3. **实时图像显示**
4. **图像保存功能**
5. **多检测模式切换**

## 故障排除

### 1. NumPy 版本兼容性问题
如果遇到 NumPy 版本冲突：
```bash
pip uninstall numpy opencv-python -y
pip install numpy==1.24.3 opencv-python==4.8.1.78
```

### 2. 摄像头无法打开
- 检查摄像头是否被其他程序占用
- 尝试不同的摄像头ID（0, 1, 2...）
- 确保摄像头驱动正常

### 3. 性能优化
- 降低摄像头分辨率
- 调整检测参数
- 使用更快的检测模式

## 文件说明

- `openpose_realtime_opencv.py`: 主要的OpenPose程序（OpenCV版本）
- `requirements_opencv.txt`: 依赖包列表
- `INSTALL_GUIDE.md`: 本安装指南

## 注意事项

1. 确保摄像头权限已开启
2. 在光线充足的环境下使用效果更好
3. 程序需要摄像头支持，如果没有摄像头会报错
4. 建议在Python 3.11环境中运行以获得最佳兼容性
