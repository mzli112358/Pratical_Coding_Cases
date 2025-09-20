#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO面部关键点检测演示脚本
快速启动面部关键点检测的演示程序
"""

import cv2
import os
import sys
from face_keypoint_detector import FaceKeypointDetector

def main():
    """主演示函数"""
    print("=" * 60)
    print("🎯 YOLO面部关键点实时检测演示")
    print("=" * 60)
    
    print("\n请选择检测模式:")
    print("1. 实时摄像头检测")
    print("2. 图像文件检测")
    print("3. 视频文件检测")
    
    try:
        choice = input("\n请输入选择 (1-3): ").strip()
        
        if choice == "1":
            # 实时摄像头检测
            print("\n🚀 启动实时摄像头面部关键点检测...")
            print("📋 操作说明:")
            print("   - 按 'q' 键退出程序")
            print("   - 按 's' 键保存当前帧")
            print("   - 按 'c' 键切换置信度阈值")
            
            detector = FaceKeypointDetector()
            detector.detect_face_keypoints_camera()
            
        elif choice == "2":
            # 图像文件检测
            image_path = input("\n请输入图像文件路径: ").strip()
            
            if not os.path.exists(image_path):
                print(f"❌ 图像文件不存在: {image_path}")
                return
            
            print(f"\n🖼️  正在检测图像: {image_path}")
            
            # 询问是否保存结果
            save_choice = input("是否保存检测结果? (y/n): ").strip().lower()
            output_path = None
            
            if save_choice == 'y':
                # 生成输出文件名
                name, ext = os.path.splitext(image_path)
                output_path = f"{name}_face_keypoints{ext}"
                print(f"📁 结果将保存到: {output_path}")
            
            detector = FaceKeypointDetector()
            detector.detect_face_keypoints_image(image_path, output_path)
            
        elif choice == "3":
            # 视频文件检测
            video_path = input("\n请输入视频文件路径: ").strip()
            
            if not os.path.exists(video_path):
                print(f"❌ 视频文件不存在: {video_path}")
                return
            
            print(f"\n🎬 正在检测视频: {video_path}")
            
            # 询问是否保存结果
            save_choice = input("是否保存检测结果? (y/n): ").strip().lower()
            output_path = None
            
            if save_choice == 'y':
                # 生成输出文件名
                name, ext = os.path.splitext(video_path)
                output_path = f"{name}_face_keypoints{ext}"
                print(f"📁 结果将保存到: {output_path}")
            
            print("📋 操作说明:")
            print("   - 按 'q' 键退出视频播放")
            
            detector = FaceKeypointDetector()
            detector.detect_face_keypoints_video(video_path, output_path)
            
        else:
            print("❌ 无效选择，请输入 1-3")
            return
            
    except KeyboardInterrupt:
        print("\n\n👋 程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
    
    print("\n✅ 程序结束")

if __name__ == "__main__":
    main()
