#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO骨架识别应用启动器
提供简单的菜单界面来运行不同的应用版本
"""

import os
import sys
import subprocess

def print_menu():
    """打印菜单"""
    print("\n" + "="*60)
    print("           YOLO骨架识别应用系统")
    print("="*60)
    print("1. 基础版骨架识别应用")
    print("2. 增强版骨架识别应用（推荐）")
    print("3. 查看录制数据")
    print("4. 安装依赖包")
    print("5. 退出")
    print("="*60)

def run_basic_app():
    """运行基础版应用"""
    print("\n启动基础版骨架识别应用...")
    print("功能: 实时骨架识别、录制、卡通形象绑定")
    try:
        subprocess.run([sys.executable, "skeleton_recording_app.py"])
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"运行失败: {e}")

def run_enhanced_app():
    """运行增强版应用"""
    print("\n启动增强版骨架识别应用...")
    print("功能: 实时骨架识别、录制、卡通形象绑定、完整UI控制")
    try:
        subprocess.run([sys.executable, "enhanced_skeleton_app.py"])
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"运行失败: {e}")

def view_recordings():
    """查看录制数据"""
    recordings_dir = "recordings"
    if not os.path.exists(recordings_dir):
        print("\n录制数据目录不存在")
        return
    
    files = [f for f in os.listdir(recordings_dir) if f.endswith('.json')]
    if not files:
        print("\n没有找到录制数据文件")
        return
    
    print(f"\n找到 {len(files)} 个录制数据文件:")
    for i, file in enumerate(files, 1):
        filepath = os.path.join(recordings_dir, file)
        size = os.path.getsize(filepath)
        print(f"{i}. {file} ({size} bytes)")
    
    choice = input("\n请输入要查看的文件编号 (0返回): ").strip()
    try:
        file_index = int(choice) - 1
        if file_index == -1:
            return
        if 0 <= file_index < len(files):
            filepath = os.path.join(recordings_dir, files[file_index])
            with open(filepath, 'r', encoding='utf-8') as f:
                import json
                data = json.load(f)
                print(f"\n文件: {files[file_index]}")
                print(f"录制帧数: {len(data)}")
                if data:
                    print(f"开始时间: {data[0]['timestamp']}")
                    print(f"结束时间: {data[-1]['timestamp']}")
                    print(f"时长: {data[-1]['timestamp'] - data[0]['timestamp']:.2f} 秒")
        else:
            print("无效的文件编号")
    except ValueError:
        print("请输入有效的数字")
    except Exception as e:
        print(f"读取文件失败: {e}")

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
    print("欢迎使用YOLO骨架识别应用系统！")
    print("本应用支持实时摄像头骨架识别、录制和卡通形象绑定")
    
    while True:
        print_menu()
        choice = input("\n请选择功能 (1-5): ").strip()
        
        if choice == "1":
            run_basic_app()
        elif choice == "2":
            run_enhanced_app()
        elif choice == "3":
            view_recordings()
        elif choice == "4":
            install_dependencies()
        elif choice == "5":
            print("\n感谢使用！再见！")
            break
        else:
            print("\n无效选择，请重新输入")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    main()
