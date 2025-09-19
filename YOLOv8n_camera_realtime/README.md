# YOLOv8 实时摄像头目标检测

这是一个基于YOLOv8的实时摄像头目标检测项目，可以实时检测摄像头画面中的各种目标对象。

## 功能特点

- 🎥 **实时检测**: 使用摄像头进行实时目标检测
- 🚀 **高性能**: 基于YOLOv8n轻量级模型，检测速度快
- 📊 **可视化**: 实时显示检测框、类别标签和置信度
- 💾 **截图保存**: 按's'键保存当前检测画面
- 📈 **FPS显示**: 实时显示检测帧率
- 🔧 **自动下载**: 如果模型文件不存在会自动下载

## 环境要求

- Python 3.8+
- 摄像头设备
- 支持CUDA的GPU（可选，用于加速）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本使用

```bash
cd YOLOv8n_camera_realtime
python camera_realtime_detect.py
```

### 操作说明

- **退出程序**: 按 `q` 键
- **保存截图**: 按 `s` 键保存当前检测画面
- **关闭窗口**: 点击窗口右上角的关闭按钮

## 项目结构

```
YOLOv8n_camera_realtime/
├── camera_realtime_detect.py  # 主程序文件
├── requirements.txt           # 依赖包列表
└── README.md                 # 说明文档
```

## 模型文件

项目使用 `models/yolov8n.pt` 模型文件。如果该文件不存在，程序会自动下载。

## 支持的检测类别

YOLOv8n模型可以检测80种常见的目标类别，包括：

- 人物 (person)
- 车辆 (car, truck, bus, motorcycle, bicycle)
- 动物 (cat, dog, horse, cow, etc.)
- 日常物品 (bottle, cup, book, laptop, etc.)
- 食物 (apple, banana, pizza, etc.)
- 等等...

## 性能优化

### GPU加速

如果您的系统有NVIDIA GPU并安装了CUDA，程序会自动使用GPU加速：

```bash
# 检查CUDA是否可用
python -c "import torch; print(torch.cuda.is_available())"
```

### 调整检测参数

您可以在代码中调整以下参数来优化性能：

- **置信度阈值**: 修改 `confidence > 0.5` 来调整检测敏感度
- **摄像头分辨率**: 修改 `set(cv2.CAP_PROP_FRAME_WIDTH/HEIGHT)` 来调整输入分辨率
- **检测间隔**: 可以添加帧跳过逻辑来减少计算量

## 故障排除

### 常见问题

1. **摄像头无法打开**
   - 检查摄像头是否被其他程序占用
   - 尝试更改 `camera_index` 参数（0, 1, 2...）

2. **检测速度慢**
   - 降低摄像头分辨率
   - 确保安装了GPU版本的PyTorch
   - 增加置信度阈值减少误检

3. **模型下载失败**
   - 检查网络连接
   - 手动下载模型文件到根目录

### 调试模式

在代码中添加 `verbose=True` 参数可以看到更详细的检测信息：

```python
results = self.model(frame, verbose=True)
```

## 扩展功能

### 添加新的检测类别

可以通过训练自定义模型来检测特定类别：

```python
# 使用自定义模型
detector = CameraRealtimeDetector(model_path="path/to/your/model.pt")
```

### 保存检测结果

可以修改代码来保存检测结果到文件：

```python
# 保存检测结果
with open("detection_results.txt", "a") as f:
    f.write(f"{timestamp}: {detected_objects}\n")
```

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！
