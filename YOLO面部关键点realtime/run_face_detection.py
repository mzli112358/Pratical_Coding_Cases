#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO面部关键点检测快速启动脚本
一键启动面部关键点实时检测
"""

import sys
import os

def main():
    """快速启动面部关键点检测"""
    print("🎯 YOLO面部关键点实时检测系统")
    print("=" * 50)
    
    # 检查依赖
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
        print("✅ 依赖检查通过")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r requirements.txt")
        return
    
    # 检查模型文件
    model_path = "yolov8n-pose.pt"
    if not os.path.exists(model_path):
        print(f"⚠️  模型文件不存在: {model_path}")
        print("首次运行将自动下载模型文件...")
    
    print("\n🚀 正在启动实时面部关键点检测...")
    print("📋 操作说明:")
    print("   - 按 'q' 键退出程序")
    print("   - 按 's' 键保存当前帧")
    print("   - 按 'c' 键切换置信度阈值")
    print("   - 确保摄像头权限已开启")
    
    try:
        from face_keypoint_detector import FaceKeypointDetector
        detector = FaceKeypointDetector()
        detector.detect_face_keypoints_camera()
    except KeyboardInterrupt:
        print("\n👋 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        print("请检查摄像头是否正常工作")

if __name__ == "__main__":
    main()
