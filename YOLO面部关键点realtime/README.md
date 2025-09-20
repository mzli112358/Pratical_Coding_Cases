# YOLO面部关键点实时检测系统

基于YOLOv8-pose的面部关键点检测系统，专门用于实时检测和分析面部关键点，支持摄像头、图像和视频文件的面部关键点识别。

## 🌟 功能特性

- 🎯 **实时面部关键点检测**: 支持摄像头实时面部关键点识别
- 📷 **图像面部检测**: 单张图像的面部关键点分析
- 🎬 **视频面部检测**: 视频文件中的面部关键点追踪
- 🎨 **可视化面部关键点**: 彩色关键点和面部连接线
- ⚡ **高性能检测**: 基于YOLOv8-pose模型，检测速度快
- 🔧 **易于使用**: 交互式演示脚本和命令行界面
- 🎛️ **实时调节**: 支持实时切换置信度阈值

## 📦 安装依赖

```bash
pip install -r requirements.txt
```

### 系统要求

- Python 3.8+
- OpenCV 4.5+
- PyTorch 1.9+
- 支持CUDA的GPU（可选，用于加速）

## 🚀 快速开始

### 方法1: 使用演示脚本（推荐新手）

```bash
python demo.py
```

然后按照提示选择检测模式：
1. 实时摄像头检测
2. 图像文件检测  
3. 视频文件检测

### 方法2: 使用命令行界面

#### 1. 实时摄像头面部关键点检测

```bash
# 使用默认摄像头
python face_keypoint_detector.py --mode camera

# 指定摄像头索引
python face_keypoint_detector.py --mode camera --camera 1

# 调整置信度阈值
python face_keypoint_detector.py --mode camera --confidence 0.3
```

#### 2. 图像面部关键点检测

```bash
# 检测图像并显示结果
python face_keypoint_detector.py --mode image --input path/to/image.jpg

# 检测图像并保存结果
python face_keypoint_detector.py --mode image --input path/to/image.jpg --output result.jpg

# 调整置信度阈值
python face_keypoint_detector.py --mode image --input image.jpg --confidence 0.7
```

#### 3. 视频面部关键点检测

```bash
# 检测视频并显示结果
python face_keypoint_detector.py --mode video --input path/to/video.mp4

# 检测视频并保存结果
python face_keypoint_detector.py --mode video --input path/to/video.mp4 --output result.mp4

# 调整置信度阈值
python face_keypoint_detector.py --mode video --input video.mp4 --confidence 0.4
```

## 🎮 操作说明

### 实时检测模式快捷键

- **`q`**: 退出程序
- **`s`**: 保存当前帧为图片
- **`c`**: 切换置信度阈值（0.3 ↔ 0.5）

### 视频检测模式

- **`q`**: 退出视频播放

## 🎯 面部关键点说明

系统检测5个主要面部关键点，基于COCO格式：

| 索引 | 关键点名称 | 颜色 | 说明 |
|------|------------|------|------|
| 0 | 鼻子 | 黄色 | 面部中心点 |
| 1 | 左眼 | 蓝色 | 左眼中心 |
| 2 | 右眼 | 蓝色 | 右眼中心 |
| 3 | 左耳 | 绿色 | 左耳位置 |
| 4 | 右耳 | 绿色 | 右耳位置 |

## 🔗 面部连接线

系统会自动绘制以下面部连接线：
- 鼻子 ↔ 左眼
- 鼻子 ↔ 右眼  
- 左眼 ↔ 左耳
- 右眼 ↔ 右耳
- 左眼 ↔ 右眼

## 📊 参数说明

| 参数 | 说明 | 默认值 | 可选值 |
|------|------|--------|--------|
| `--mode` | 检测模式 | camera | image, video, camera |
| `--input` | 输入文件路径 | - | 图像或视频文件路径 |
| `--output` | 输出文件路径 | - | 保存结果的文件路径 |
| `--model` | 模型文件路径 | yolov8n-pose.pt | 自定义模型路径 |
| `--camera` | 摄像头索引 | 0 | 0, 1, 2, ... |
| `--confidence` | 置信度阈值 | 0.5 | 0.0-1.0 |

## 💻 代码示例

### 基本使用

```python
from face_keypoint_detector import FaceKeypointDetector

# 创建检测器
detector = FaceKeypointDetector()

# 检测图像
detector.detect_face_keypoints_image('input.jpg', 'output.jpg')

# 检测视频
detector.detect_face_keypoints_video('input.mp4', 'output.mp4')

# 实时检测
detector.detect_face_keypoints_camera()
```

### 自定义参数

```python
# 使用自定义模型和摄像头
detector = FaceKeypointDetector(
    model_path='custom_model.pt',
    camera_index=1
)

# 检测图像并调整置信度
detector.detect_face_keypoints_image(
    'image.jpg', 
    'result.jpg', 
    confidence_threshold=0.7
)
```

### 提取面部关键点数据

```python
from face_keypoint_detector import FaceKeypointDetector
import cv2

detector = FaceKeypointDetector()

# 读取图像
frame = cv2.imread('image.jpg')

# 进行检测
results = detector.model(frame, verbose=False)

if results[0].keypoints is not None:
    keypoints = results[0].keypoints.data.cpu().numpy()
    
    # 提取面部关键点
    face_keypoints = detector.extract_face_keypoints(keypoints)
    
    # 处理关键点数据
    for person_face_keypoints in face_keypoints:
        for i, keypoint in enumerate(person_face_keypoints):
            if keypoint is not None:
                x, y, confidence = keypoint[0], keypoint[1], keypoint[2]
                print(f"关键点 {i}: ({x:.1f}, {y:.1f}), 置信度: {confidence:.3f}")
```

## 🔧 技术特点

- **模型**: YOLOv8-pose (轻量级姿态估计模型)
- **关键点**: 5个面部关键点（鼻子、双眼、双耳）
- **连接**: 自动绘制面部连接线
- **可视化**: 彩色关键点和连接线
- **性能**: 实时检测，支持多种输入源
- **鲁棒性**: 自动过滤低置信度关键点

## 📈 性能优化建议

1. **GPU加速**: 使用支持CUDA的GPU可显著提升检测速度
2. **分辨率调整**: 降低输入分辨率可提高FPS
3. **置信度调节**: 根据场景调整置信度阈值平衡精度和召回率
4. **模型选择**: 使用更小的模型（如yolov8n-pose）可获得更高FPS

## ⚠️ 注意事项

1. **首次运行**: 会自动下载YOLOv8n-pose模型（约6MB）
2. **摄像头权限**: 确保摄像头权限已开启
3. **光照条件**: 建议在光线充足的环境下使用
4. **面部朝向**: 检测效果取决于面部朝向和清晰度
5. **多人检测**: 支持同时检测多个人脸

## 🐛 故障排除

### 常见问题

1. **摄像头无法打开**
   ```
   解决方案:
   - 检查摄像头是否被其他程序占用
   - 尝试不同的摄像头索引（--camera 1, 2, 3...）
   - 检查摄像头驱动是否正常
   ```

2. **模型下载失败**
   ```
   解决方案:
   - 检查网络连接
   - 手动下载模型文件到项目目录
   - 使用代理或镜像源
   ```

3. **检测效果不佳**
   ```
   解决方案:
   - 调整置信度阈值（--confidence 0.3-0.7）
   - 确保面部在画面中清晰可见
   - 避免遮挡和模糊
   - 调整光照条件
   ```

4. **FPS过低**
   ```
   解决方案:
   - 使用GPU加速
   - 降低摄像头分辨率
   - 使用更小的模型
   - 关闭其他占用资源的程序
   ```

## 📝 更新日志

### v1.0.0 (当前版本)
- ✅ 初始版本发布
- ✅ 支持实时摄像头面部关键点检测
- ✅ 支持图像和视频文件检测
- ✅ 彩色关键点和连接线可视化
- ✅ 交互式演示脚本
- ✅ 命令行界面和参数配置
- ✅ 实时置信度调节功能

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

---

**享受面部关键点检测的乐趣！** 🎉
