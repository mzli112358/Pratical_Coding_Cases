import numpy as np
from PIL import Image, ImageDraw

# 设置图像尺寸
width, height = 800, 600

# 创建黑色图像
image = Image.new('L', (width, height), 0)
draw = ImageDraw.Draw(image)

# 你的区域数据
regions = (
    ((0, 0), (40, 0), (0, 70)),
    ((55, 0), (100, 0), (100, 100), (85, 100))
)

# 绘制所有区域
for region in regions:
    # 转换百分比坐标为像素坐标
    pixel_points = []
    for w, h in region:
        x = int(w * width / 100)
        y = int(h * height / 100)
        pixel_points.append((x, y))
    
    # 填充多边形
    draw.polygon(pixel_points, fill=255)

# 保存mask
image.save('pure_python_mask.png')
print("Mask创建完成并保存为 pure_python_mask.png")