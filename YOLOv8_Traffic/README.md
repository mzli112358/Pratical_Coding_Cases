# YOLO 交通分析系统

基于YOLOv8的智能交通监控和车辆分析系统，支持实时视频流分析和静态图像处理。

## 🚀 功能特性

- **实时车辆检测**：使用YOLOv8模型检测车辆、公交车、卡车、行人等
- **车辆追踪**：基于轨迹的车辆追踪算法，支持多目标追踪
- **交通流量统计**：自动统计通过指定计数线的车辆数量
- **轨迹分析**：绘制车辆运动轨迹和回归线
- **实时GUI显示**：支持实时视频窗口显示和键盘交互
- **图像处理**：支持静态图像的目标检测和车辆裁剪

## 📁 项目结构

```
yolo/
├── video_traffic_live.py      # 实时视频流分析（GUI版本）
├── video_traffic_analysis.py  # 视频分析（保存帧版本）
├── image_object_detect.py     # 静态图像目标检测
├── extract_frames.py          # 视频帧提取工具
├── download_yolov8n.py        # YOLO模型下载工具
├── traffic_cctv.mp4           # 示例交通视频
├── yolov8n.pt                 # YOLOv8n预训练模型
├── frames/                    # 提取的视频帧
├── detections/                # 检测结果图像
└── traffic_output/            # 交通分析输出结果
```

## 🛠️ 安装要求

### 系统要求
- Python 3.8+
- Windows/Linux/macOS

### 依赖包
```bash
pip install ultralytics opencv-python numpy scipy
```

**注意**：确保安装的是 `opencv-python` 而不是 `opencv-python-headless`，后者不支持GUI显示。

## 🚦 使用方法

### 1. 实时视频流分析（推荐）

```bash
python video_traffic_live.py
```

**功能**：
- 实时显示视频分析窗口
- 车辆检测和追踪
- 交通流量统计
- 轨迹可视化

**控制键**：
- `q`：退出程序
- `r`：重置车辆计数

### 2. 视频分析（保存帧模式）

```bash
python video_traffic_analysis.py
```

**功能**：
- 分析视频并保存关键帧
- 适用于无GUI环境
- 自动保存到 `traffic_output/` 目录

### 3. 静态图像检测

```bash
python image_object_detect.py
```

**功能**：
- 检测图像中的车辆
- 自动裁剪检测到的车辆
- 保存到 `detections/` 目录

### 4. 视频帧提取

```bash
python extract_frames.py
```

**功能**：
- 从视频中提取关键帧
- 每分钟提取一帧
- 保存到 `frames/` 目录

## 🔧 配置说明

### 车辆追踪参数

在 `VehicleTracker` 类中可以调整以下参数：

```python
self.max_disappeared = 30      # 车辆消失多少帧后移除追踪
self.trajectory_length = 50    # 轨迹点最大保存数量
self.counting_line_y = None    # 计数线Y坐标（自动设置为视频中心）
```

### 检测阈值

```python
confidence > 0.5  # 检测置信度阈值
```

## 📊 输出说明

### 实时显示信息
- **Vehicle Count**：通过计数线的车辆总数
- **Active Tracks**：当前活跃的追踪数量
- **检测框**：绿色框显示检测到的车辆
- **轨迹线**：黄色线显示车辆运动轨迹
- **回归线**：紫色线显示车辆运动趋势
- **计数线**：红色线显示车辆计数位置

### 保存文件
- `traffic_output/traffic_frame_XXXXXX.png`：分析结果帧
- `detections/car_XXXXXX.png`：裁剪的车辆图像
- `frames/frame_XXXXXX.png`：提取的视频帧

## 🎯 核心算法

### 车辆追踪算法
1. **检测匹配**：基于欧几里得距离的最近邻匹配
2. **轨迹更新**：维护车辆位置历史
3. **轨迹拟合**：使用线性回归拟合车辆运动趋势
4. **计数检测**：检测车辆是否穿越计数线

### 目标检测
- 使用YOLOv8n预训练模型
- 支持COCO数据集的80个类别
- 主要检测：car, bus, truck, person等

## 🔍 故障排除

### 常见问题

1. **GUI不显示窗口**
   ```bash
   # 卸载headless版本
   pip uninstall opencv-python-headless
   # 重新安装完整版本
   pip install opencv-python
   ```

2. **模型下载失败**
   ```bash
   # 手动下载模型
   python download_yolov8n.py
   ```

3. **视频文件无法打开**
   - 确保视频文件存在且格式支持
   - 检查文件路径是否正确

## 📈 性能优化

- **GPU加速**：安装CUDA版本的PyTorch可显著提升检测速度
- **分辨率调整**：降低视频分辨率可提高处理速度
- **追踪参数**：调整 `max_disappeared` 参数平衡准确性和性能

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 📄 许可证

本项目基于MIT许可证开源。

## 🙏 致谢

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)

---

**注意**：本项目仅用于学习和研究目的，请遵守相关法律法规和隐私政策。
