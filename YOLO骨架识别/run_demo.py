#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO骨架识别演示启动器
提供简单的菜单界面来运行不同的演示程序
"""

import os
import sys
import subprocess

def print_menu():
    """打印菜单"""
    print("\n" + "="*60)
    print("           YOLO骨架识别演示系统")
    print("="*60)
    print("1. 实时摄像头骨架检测")
    print("2. 图像骨架检测")
    print("3. 视频骨架检测")
    print("4. 图像骨架检测（详细分析）")
    print("5. 视频骨架检测（详细分析）")
    print("6. 安装依赖包")
    print("7. 退出")
    print("="*60)

def run_camera_detection():
    """运行实时摄像头检测"""
    print("\n启动实时摄像头骨架检测...")
    print("按 'q' 键退出，按 's' 键保存当前帧")
    try:
        subprocess.run([sys.executable, "skeleton_detection.py", "--mode", "camera"])
    except KeyboardInterrupt:
        print("\n程序被用户中断")

def run_image_detection():
    """运行图像检测"""
    image_path = input("\n请输入图像文件路径: ").strip()
    if not os.path.exists(image_path):
        print(f"文件 {image_path} 不存在")
        return
    
    output_path = input("请输入输出文件路径（留空则只显示）: ").strip()
    if not output_path:
        output_path = None
    
    print(f"\n正在检测图像: {image_path}")
    try:
        cmd = [sys.executable, "skeleton_detection.py", "--mode", "image", "--input", image_path]
        if output_path:
            cmd.extend(["--output", output_path])
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n程序被用户中断")

def run_video_detection():
    """运行视频检测"""
    video_path = input("\n请输入视频文件路径: ").strip()
    if not os.path.exists(video_path):
        print(f"文件 {video_path} 不存在")
        return
    
    output_path = input("请输入输出文件路径（留空则只显示）: ").strip()
    if not output_path:
        output_path = None
    
    print(f"\n正在检测视频: {video_path}")
    try:
        cmd = [sys.executable, "skeleton_detection.py", "--mode", "video", "--input", video_path]
        if output_path:
            cmd.extend(["--output", output_path])
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n程序被用户中断")

def run_image_demo():
    """运行图像详细分析"""
    image_path = input("\n请输入图像文件路径: ").strip()
    if not os.path.exists(image_path):
        print(f"文件 {image_path} 不存在")
        return
    
    output_path = input("请输入输出文件路径（留空则只显示）: ").strip()
    if not output_path:
        output_path = None
    
    print(f"\n正在分析图像: {image_path}")
    try:
        cmd = [sys.executable, "image_skeleton_demo.py", "--input", image_path]
        if output_path:
            cmd.extend(["--output", output_path])
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n程序被用户中断")

def run_video_demo():
    """运行视频详细分析"""
    video_path = input("\n请输入视频文件路径: ").strip()
    if not os.path.exists(video_path):
        print(f"文件 {video_path} 不存在")
        return
    
    output_path = input("请输入输出文件路径（留空则只显示）: ").strip()
    if not output_path:
        output_path = None
    
    print(f"\n正在分析视频: {video_path}")
    try:
        cmd = [sys.executable, "video_skeleton_demo.py", "--input", video_path]
        if output_path:
            cmd.extend(["--output", output_path])
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n程序被用户中断")

def install_dependencies():
    """安装依赖包"""
    print("\n正在安装依赖包...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("依赖包安装完成！")
    except Exception as e:
        print(f"安装依赖包失败: {e}")

def main():
    """主函数"""
    print("欢迎使用YOLO骨架识别演示系统！")
    
    while True:
        print_menu()
        choice = input("\n请选择功能 (1-7): ").strip()
        
        if choice == "1":
            run_camera_detection()
        elif choice == "2":
            run_image_detection()
        elif choice == "3":
            run_video_detection()
        elif choice == "4":
            run_image_demo()
        elif choice == "5":
            run_video_demo()
        elif choice == "6":
            install_dependencies()
        elif choice == "7":
            print("\n感谢使用！再见！")
            break
        else:
            print("\n无效选择，请重新输入")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    main()
