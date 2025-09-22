# TensorFlow单目深度估计

基于TensorFlow的深度学习单目深度估计项目，使用编码器-解码器架构从单张RGB图像预测深度信息。

## 功能特性

- 🎯 **高精度深度估计**: 基于深度学习的端到端深度预测
- ⚡ **实时性能**: 支持实时摄像头深度估计
- 📊 **多模态支持**: 支持图像、视频和实时摄像头输入
- 🎨 **可视化深度图**: 彩色深度图显示，直观展示深度信息
- 🔧 **易于使用**: 命令行界面，支持多种参数配置
- 📚 **完整训练流程**: 包含模型训练和评估功能

## 项目结构

```
tf_monocular_depth/
├── requirements.txt              # 依赖包列表
├── depth_estimation_model.py     # 深度估计模型定义
├── depth_demo.py                 # 深度估计演示脚本
├── train_depth_model.py          # 模型训练脚本
└── README.md                     # 项目说明文档
```

## 安装依赖

```bash
pip install -r requirements.txt
```

### 依赖包说明

- `tensorflow>=2.10.0`: 深度学习框架
- `opencv-python>=4.5.0`: 图像处理和摄像头操作
- `numpy>=1.21.0`: 数值计算
- `matplotlib>=3.5.0`: 数据可视化
- `Pillow>=8.0.0`: 图像处理
- `scikit-image>=0.19.0`: 图像处理算法

## 使用方法

### 0. 设置预训练模型（推荐）

首先设置预训练模型以获得更好的效果：

```bash
# 自动设置预训练模型
python setup_pretrained.py

# 或者手动运行预训练模型测试
python pretrained_models.py
```

### 1. 模型训练

首先需要训练深度估计模型：

```bash
# 使用默认参数训练
python train_depth_model.py

# 自定义训练参数
python train_depth_model.py --epochs 100 --batch_size 16 --samples 2000 --output my_depth_model
```

**训练参数说明:**
- `--epochs`: 训练轮数 (默认: 50)
- `--batch_size`: 批次大小 (默认: 8)
- `--samples`: 训练样本数量 (默认: 1000)
- `--output`: 模型保存路径 (默认: trained_depth_model)

### 2. 图像深度估计

```bash
# 使用预训练模型处理图像（推荐）
python depth_demo.py --mode image --input path/to/image.jpg --pretrained

# 处理图像并保存结果
python depth_demo.py --mode image --input path/to/image.jpg --output result.jpg --pretrained

# 使用自定义训练的模型
python depth_demo.py --mode image --input path/to/image.jpg --model trained_depth_model

# 使用自定义模型（未训练）
python depth_demo.py --mode image --input path/to/image.jpg --no-pretrained
```

### 3. 视频深度估计

```bash
# 使用预训练模型处理视频（推荐）
python depth_demo.py --mode video --input path/to/video.mp4 --pretrained

# 处理视频并保存结果
python depth_demo.py --mode video --input path/to/video.mp4 --output result.mp4 --pretrained

# 使用自定义训练的模型
python depth_demo.py --mode video --input path/to/video.mp4 --model trained_depth_model

# 使用自定义模型（未训练）
python depth_demo.py --mode video --input path/to/video.mp4 --no-pretrained
```

### 4. 实时摄像头深度估计

```bash
# 使用预训练模型（推荐）
python depth_demo.py --mode camera --pretrained

# 指定摄像头ID
python depth_demo.py --mode camera --camera 1 --pretrained

# 使用自定义训练的模型
python depth_demo.py --mode camera --model trained_depth_model

# 使用自定义模型（未训练）
python depth_demo.py --mode camera --no-pretrained
```

### 5. 快速启动（推荐）

使用交互式启动脚本：

```bash
python run_demo.py
```

这将显示一个菜单，让您选择运行模式和模型类型。

## 模型架构

本项目使用编码器-解码器架构进行深度估计：

### 编码器部分
- **输入层**: RGB图像 (H×W×3)
- **特征提取**: 4层卷积网络，逐步提取高级特征
- **下采样**: 通过最大池化减少空间维度
- **瓶颈层**: 最深层特征表示

### 解码器部分
- **上采样**: 4层转置卷积，恢复空间维度
- **特征融合**: 跳跃连接和特征组合
- **输出层**: 单通道深度图 (H×W×1)

### 损失函数
结合L1损失和SSIM损失：
- **L1损失**: 像素级深度值误差
- **SSIM损失**: 结构相似性损失，保持深度图结构

## 深度图可视化

深度图使用颜色映射进行可视化：
- 🔴 **红色/暖色**: 近距离物体
- 🔵 **蓝色/冷色**: 远距离物体
- ⚪ **白色**: 中等距离

## 性能优化建议

### 1. 模型训练
- 使用GPU加速训练过程
- 增加训练样本数量提高泛化能力
- 调整学习率和批次大小
- 使用数据增强技术

### 2. 推理优化
- 使用TensorRT或OpenVINO进行模型优化
- 降低输入图像分辨率提高速度
- 使用量化技术减少模型大小

### 3. 硬件要求
- **推荐**: NVIDIA GPU with CUDA support
- **最低**: CPU (速度较慢)
- **内存**: 至少8GB RAM
- **存储**: 至少2GB可用空间

## 常见问题

### Q: 模型预测的深度图不准确怎么办？
A: 这是正常现象，因为使用的是合成数据训练的模型。建议：
- 增加训练样本数量
- 调整训练参数
- 使用真实深度数据集训练

### Q: 实时处理速度慢怎么办？
A: 可以尝试以下优化：
- 降低输入图像分辨率
- 使用更小的模型
- 启用GPU加速
- 使用模型量化

### Q: 如何提高深度估计精度？
A: 建议：
- 使用更大的预训练模型
- 增加训练数据多样性
- 使用更复杂的损失函数
- 添加数据增强

## 扩展功能

### 1. 添加新的损失函数
在 `depth_estimation_model.py` 中修改 `depth_loss` 函数

### 2. 支持更多输入格式
在 `depth_demo.py` 中添加新的处理模式

### 3. 集成其他深度估计模型
可以集成MiDaS、DPT等预训练模型

## 参考资料

- [TensorFlow官方文档](https://www.tensorflow.org/)
- [深度估计论文](https://paperswithcode.com/task/monocular-depth-estimation)
- [OpenCV文档](https://docs.opencv.org/)

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

---

**注意**: 本项目主要用于学习和研究目的。在生产环境中使用前，请确保模型经过充分的训练和验证。
