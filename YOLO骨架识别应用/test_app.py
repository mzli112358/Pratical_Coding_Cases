#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO骨架识别应用测试脚本
用于验证各个组件是否正常工作
"""

import os
import sys
import cv2
import numpy as np
from cartoon_character import CartoonCharacter

def test_cartoon_character():
    """测试卡通形象类"""
    print("测试卡通形象类...")
    
    try:
        # 创建卡通形象实例
        character = CartoonCharacter()
        
        # 测试获取角色类型
        character_types = character.get_character_types()
        print(f"可用角色类型: {character_types}")
        
        # 创建模拟关键点数据
        mock_keypoints = np.array([
            [320, 100, 0.9],  # 鼻子
            [310, 90, 0.8],   # 左眼
            [330, 90, 0.8],   # 右眼
            [300, 100, 0.7],  # 左耳
            [340, 100, 0.7],  # 右耳
            [280, 150, 0.9],  # 左肩
            [360, 150, 0.9],  # 右肩
            [250, 200, 0.8],  # 左肘
            [390, 200, 0.8],  # 右肘
            [220, 250, 0.7],  # 左腕
            [420, 250, 0.7],  # 右腕
            [300, 250, 0.9],  # 左臀
            [340, 250, 0.9],  # 右臀
            [280, 350, 0.8],  # 左膝
            [360, 350, 0.8],  # 右膝
            [260, 450, 0.7],  # 左踝
            [380, 450, 0.7]   # 右踝
        ])
        
        # 更新姿态
        character.update_pose(mock_keypoints)
        
        # 创建测试图像
        test_image = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # 测试不同角色类型
        for char_type in character_types:
            character.set_character_type(char_type)
            test_image_copy = test_image.copy()
            result = character.draw(test_image_copy)
            print(f"角色类型 '{char_type}' 测试通过")
        
        print("卡通形象类测试通过！")
        return True
        
    except Exception as e:
        print(f"卡通形象类测试失败: {e}")
        return False

def test_opencv():
    """测试OpenCV功能"""
    print("测试OpenCV功能...")
    
    try:
        # 测试摄像头
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("摄像头可用")
            cap.release()
        else:
            print("摄像头不可用，但OpenCV正常")
        
        # 测试图像操作
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(test_image, (50, 50), 20, (0, 255, 0), -1)
        
        print("OpenCV功能测试通过！")
        return True
        
    except Exception as e:
        print(f"OpenCV功能测试失败: {e}")
        return False

def test_yolo_model():
    """测试YOLO模型加载"""
    print("测试YOLO模型加载...")
    
    try:
        from ultralytics import YOLO
        
        # 尝试加载模型
        model = YOLO('yolov8n-pose.pt')
        print("YOLO模型加载成功！")
        return True
        
    except Exception as e:
        print(f"YOLO模型加载失败: {e}")
        print("请确保已安装ultralytics包")
        return False

def test_dependencies():
    """测试依赖包"""
    print("测试依赖包...")
    
    required_packages = [
        'cv2',
        'numpy',
        'ultralytics',
        'torch',
        'torchvision',
        'PIL',
        'matplotlib'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"缺少依赖包: {missing_packages}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("所有依赖包已安装！")
        return True

def main():
    """主测试函数"""
    print("=" * 60)
    print("           YOLO骨架识别应用测试")
    print("=" * 60)
    
    tests = [
        ("依赖包检查", test_dependencies),
        ("OpenCV功能", test_opencv),
        ("YOLO模型", test_yolo_model),
        ("卡通形象类", test_cartoon_character)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        if test_func():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！应用可以正常运行")
        print("\n运行应用:")
        print("  python run_app.py")
    else:
        print("❌ 部分测试失败，请检查错误信息")
        print("\n建议:")
        print("1. 安装缺少的依赖包")
        print("2. 检查摄像头设备")
        print("3. 确保网络连接正常（用于下载模型）")

if __name__ == "__main__":
    main()
