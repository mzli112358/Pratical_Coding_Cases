# YOLO骨架识别系统

基于YOLOv8-pose的人体姿态估计和骨架识别系统，支持实时摄像头、视频文件和图像的人体关键点检测。

## 功能特性

- 🎯 **实时骨架检测**: 支持摄像头实时人体姿态估计
- 📷 **图像骨架识别**: 单张图像的人体关键点检测
- 🎬 **视频骨架识别**: 视频文件中的人体姿态分析
- 🎨 **可视化骨架**: 彩色关键点和骨架连接线
- ⚡ **高性能**: 基于YOLOv8-pose模型，检测速度快
- 🔧 **易于使用**: 命令行界面，支持多种参数配置

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 实时摄像头骨架检测

```bash
python skeleton_detection.py --mode camera
```

### 2. 图像骨架检测

```bash
# 检测图像并显示结果
python skeleton_detection.py --mode image --input path/to/image.jpg

# 检测图像并保存结果
python skeleton_detection.py --mode image --input path/to/image.jpg --output result.jpg
```

### 3. 视频骨架检测

```bash
# 检测视频并显示结果
python skeleton_detection.py --mode video --input path/to/video.mp4

# 检测视频并保存结果
python skeleton_detection.py --mode video --input path/to/video.mp4 --output result.mp4
```

## 参数说明

- `--mode`: 检测模式，可选值：`image`、`video`、`camera`
- `--input`: 输入文件路径（图像或视频模式必需）
- `--output`: 输出文件路径（可选）
- `--model`: 自定义模型文件路径（可选，默认使用YOLOv8n-pose）
- `--camera`: 摄像头索引（默认0）
- `--confidence`: 置信度阈值（默认0.5）

## 关键点说明

系统检测17个人体关键点，按照COCO格式：

1. **头部**: 鼻子(0)、左眼(1)、右眼(2)、左耳(3)、右耳(4)
2. **上身**: 左肩(5)、右肩(6)、左肘(7)、右肘(8)、左腕(9)、右腕(10)
3. **下身**: 左臀(11)、右臀(12)、左膝(13)、右膝(14)、左踝(15)、右踝(16)

## 骨架连接

系统会自动绘制以下骨架连接：
- 头部连接：鼻子-眼睛-耳朵
- 躯干连接：肩膀-手臂-手腕
- 腿部连接：臀部-膝盖-脚踝

## 快捷键

在实时检测模式下：
- 按 `q` 键退出程序
- 按 `s` 键保存当前帧

## 示例代码

```python
from skeleton_detection import SkeletonDetector

# 创建检测器
detector = SkeletonDetector()

# 检测图像
detector.detect_skeleton_image('input.jpg', 'output.jpg')

# 检测视频
detector.detect_skeleton_video('input.mp4', 'output.mp4')

# 实时检测
detector.detect_skeleton_camera()
```

## 技术特点

- **模型**: YOLOv8-pose (轻量级姿态估计模型)
- **关键点**: 17个COCO格式人体关键点
- **连接**: 自动绘制骨架连接线
- **可视化**: 彩色关键点和连接线
- **性能**: 实时检测，支持多种输入源

## 注意事项

1. 首次运行会自动下载YOLOv8n-pose模型
2. 确保摄像头权限已开启
3. 建议在光线充足的环境下使用
4. 检测效果取决于人体姿态的清晰度

## 系统要求

- Python 3.8+
- OpenCV 4.5+
- PyTorch 1.9+
- 支持CUDA的GPU（可选，用于加速）

## 故障排除

### 常见问题

1. **摄像头无法打开**
   - 检查摄像头是否被其他程序占用
   - 尝试不同的摄像头索引（--camera 1, 2, 3...）

2. **模型下载失败**
   - 检查网络连接
   - 手动下载模型文件

3. **检测效果不佳**
   - 调整置信度阈值（--confidence 0.3-0.7）
   - 确保人体在画面中清晰可见
   - 避免遮挡和模糊

## 更新日志

- v1.0.0: 初始版本，支持基本的骨架检测功能
- 支持实时摄像头、图像和视频检测
- 彩色关键点和骨架可视化
- 命令行界面和参数配置
