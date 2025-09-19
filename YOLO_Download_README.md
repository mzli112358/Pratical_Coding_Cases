# YOLO模型下载器

一个简单易用的YOLO模型下载脚本，支持YOLO v5、v8、v11系列的所有版本。

## 🚀 功能特性

- ✅ 支持15个YOLO模型（v5、v8、v11系列，每个系列5个版本：n, s, m, l, x）
- ✅ 序号选择下载（1-15）
- ✅ 支持批量下载（空格分隔序号）
- ✅ 支持范围下载（如：1-5）
- ✅ 支持下载全部模型（输入：all）
- ✅ 进度条显示
- ✅ 断点续传
- ✅ 重复下载检查

## 📋 模型列表

| 序号 | 模型名称 | 文件名 | 系列 |
|------|----------|--------|------|
| 1 | YOLOv5n | yolov5n.pt | YOLOv5 |
| 2 | YOLOv5s | yolov5s.pt | YOLOv5 |
| 3 | YOLOv5m | yolov5m.pt | YOLOv5 |
| 4 | YOLOv5l | yolov5l.pt | YOLOv5 |
| 5 | YOLOv5x | yolov5x.pt | YOLOv5 |
| 6 | YOLOv8n | yolov8n.pt | YOLOv8 |
| 7 | YOLOv8s | yolov8s.pt | YOLOv8 |
| 8 | YOLOv8m | yolov8m.pt | YOLOv8 |
| 9 | YOLOv8l | yolov8l.pt | YOLOv8 |
| 10 | YOLOv8x | yolov8x.pt | YOLOv8 |
| 11 | YOLOv11n | yolo11n.pt | YOLOv11 |
| 12 | YOLOv11s | yolo11s.pt | YOLOv11 |
| 13 | YOLOv11m | yolo11m.pt | YOLOv11 |
| 14 | YOLOv11l | yolo11l.pt | YOLOv11 |
| 15 | YOLOv11x | yolo11x.pt | YOLOv11 |

## 🛠️ 安装依赖

```bash
pip install requests tqdm
```

## 📖 使用方法

### 1. 交互式界面（推荐）

```bash
python code_for_download_yolo.py
```

然后按照提示操作：
1. 选择 "1. 查看模型列表" - 查看所有15个模型
2. 选择 "2. 下载模型" - 输入序号下载

### 2. 命令行直接下载

#### 查看模型列表
```bash
python code_for_download_yolo.py --list
```

#### 下载单个模型
```bash
python code_for_download_yolo.py 1
```

#### 下载多个模型（空格分隔）
```bash
python code_for_download_yolo.py 1 3 5
```

#### 下载范围模型
```bash
python code_for_download_yolo.py 1-5
```

#### 下载所有模型
```bash
python code_for_download_yolo.py --all
```

## 💡 使用示例

### 示例1：下载YOLOv8系列
```bash
python code_for_download_yolo.py 6-10
```

### 示例2：下载所有nano版本
```bash
python code_for_download_yolo.py 1 6 11
```

### 示例3：下载YOLOv5和YOLOv11的x版本
```bash
python code_for_download_yolo.py 5 15
```

## 📁 文件结构

```
models/
├── code_for_download_yolo.py    # 主脚本
├── README.md                    # 说明文档
├── yolov5n.pt                   # 下载的模型文件
├── yolov5s.pt
├── ...
└── yolo11x.pt
```

## ⚠️ 注意事项

1. **网络要求**：需要稳定的网络连接
2. **存储空间**：所有模型约需要1.5GB存储空间
3. **下载时间**：根据网络速度，完整下载可能需要10-30分钟
4. **断点续传**：如果下载中断，重新运行会检查已下载文件
5. **重复下载**：已存在的文件会询问是否重新下载

## 🔧 故障排除

### 下载失败
- 检查网络连接
- 确认防火墙设置
- 尝试使用VPN

### 文件损坏
- 删除损坏的文件重新下载
- 检查磁盘空间

### 权限问题
- 确保对下载目录有写入权限
- 在Windows上可能需要管理员权限

## 📊 模型大小对比

| 模型版本 | 参数量 | 模型大小 | 速度 | 精度 |
|----------|--------|----------|------|------|
| n (nano) | 最小   | ~6MB     | 最快 | 最低 |
| s (small) | 小     | ~22MB    | 快   | 低   |
| m (medium) | 中等   | ~50MB    | 中等 | 中等 |
| l (large) | 大     | ~87MB    | 慢   | 高   |
| x (xlarge) | 最大   | ~136MB   | 最慢 | 最高 |
