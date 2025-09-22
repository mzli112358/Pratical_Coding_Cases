#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速启动脚本 - 单目深度估计演示
"""

import sys
import os

# 切换到脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

def main():
    """主函数"""
    print("=== TensorFlow单目深度估计演示 ===")
    print("请选择运行模式:")
    print("1. 实时摄像头深度估计 (预训练模型)")
    print("2. 图像深度估计 (预训练模型)")
    print("3. 视频深度估计 (预训练模型)")
    print("4. 训练新模型")
    print("5. 实时摄像头深度估计 (自定义模型)")
    print("6. 图像深度估计 (自定义模型)")
    print("7. 视频深度估计 (自定义模型)")
    print("8. 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (1-8): ").strip()
            
            if choice == '1':
                print("\n启动实时摄像头深度估计 (预训练模型)...")
                os.system("python depth_demo.py --mode camera --pretrained")
                break
                
            elif choice == '2':
                image_path = input("请输入图像路径: ").strip()
                if os.path.exists(image_path):
                    print(f"\n处理图像: {image_path} (预训练模型)")
                    os.system(f"python depth_demo.py --mode image --input {image_path} --pretrained")
                else:
                    print("错误: 图像文件不存在")
                break
                
            elif choice == '3':
                video_path = input("请输入视频路径: ").strip()
                if os.path.exists(video_path):
                    print(f"\n处理视频: {video_path} (预训练模型)")
                    os.system(f"python depth_demo.py --mode video --input {video_path} --pretrained")
                else:
                    print("错误: 视频文件不存在")
                break
                
            elif choice == '4':
                print("\n开始训练新模型...")
                epochs = input("训练轮数 (默认50): ").strip() or "50"
                batch_size = input("批次大小 (默认8): ").strip() or "8"
                samples = input("训练样本数 (默认1000): ").strip() or "1000"
                
                cmd = f"python train_depth_model.py --epochs {epochs} --batch_size {batch_size} --samples {samples}"
                print(f"执行命令: {cmd}")
                os.system(cmd)
                break
                
            elif choice == '5':
                print("\n启动实时摄像头深度估计 (自定义模型)...")
                os.system("python depth_demo.py --mode camera --no-pretrained")
                break
                
            elif choice == '6':
                image_path = input("请输入图像路径: ").strip()
                if os.path.exists(image_path):
                    print(f"\n处理图像: {image_path} (自定义模型)")
                    os.system(f"python depth_demo.py --mode image --input {image_path} --no-pretrained")
                else:
                    print("错误: 图像文件不存在")
                break
                
            elif choice == '7':
                video_path = input("请输入视频路径: ").strip()
                if os.path.exists(video_path):
                    print(f"\n处理视频: {video_path} (自定义模型)")
                    os.system(f"python depth_demo.py --mode video --input {video_path} --no-pretrained")
                else:
                    print("错误: 视频文件不存在")
                break
                
            elif choice == '8':
                print("退出程序")
                break
                
            else:
                print("无效选择，请输入1-8")
                
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            break

if __name__ == "__main__":
    main()
