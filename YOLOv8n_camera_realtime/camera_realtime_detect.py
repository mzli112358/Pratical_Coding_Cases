#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv8 实时摄像头目标检测
支持实时显示检测结果，包括边界框、类别标签和置信度
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import sys

class CameraRealtimeDetector:
    def __init__(self, model_path=None, camera_index=0):
        """
        初始化实时检测器
        
        Args:
            model_path (str): YOLOv8模型文件路径，如果为None则自动查找
            camera_index (int): 摄像头索引，默认为0
        """
        # 如果没有提供模型路径，则自动查找
        if model_path is None:
            # 获取脚本所在目录
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # 模型文件在父目录
            model_path = os.path.join(os.path.dirname(script_dir), 'models', 'yolov8n.pt')
        
        self.model_path = model_path
        self.camera_index = camera_index
        self.model = None
        self.cap = None
        
        print(f"脚本目录: {os.path.dirname(os.path.abspath(__file__))}")
        print(f"模型路径: {model_path}")
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"模型文件 {model_path} 不存在，正在下载...")
            self.download_model()
        
        # 加载模型
        self.load_model()
        
        # 初始化摄像头
        self.init_camera()
    
    def download_model(self):
        """下载YOLOv8模型"""
        try:
            from ultralytics import YOLO
            print("正在下载YOLOv8n模型...")
            # 下载模型到指定的路径
            model = YOLO(self.model_path)
            print(f"模型下载完成！保存到: {self.model_path}")
        except Exception as e:
            print(f"下载模型失败: {e}")
            sys.exit(1)
    
    def load_model(self):
        """加载YOLOv8模型"""
        try:
            print(f"正在加载模型: {self.model_path}")
            self.model = YOLO(self.model_path)
            print("模型加载成功！")
        except Exception as e:
            print(f"加载模型失败: {e}")
            sys.exit(1)
    
    def init_camera(self):
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise Exception(f"无法打开摄像头 {self.camera_index}")
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print(f"摄像头 {self.camera_index} 初始化成功！")
        except Exception as e:
            print(f"初始化摄像头失败: {e}")
            sys.exit(1)
    
    def draw_detections(self, frame, results):
        """
        在帧上绘制检测结果
        
        Args:
            frame: 输入帧
            results: YOLO检测结果
            
        Returns:
            绘制了检测结果的帧
        """
        # 获取检测框、置信度和类别
        boxes = results[0].boxes
        if boxes is not None:
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # 获取置信度和类别
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.model.names[class_id]
                
                # 只显示置信度大于0.5的检测结果
                if confidence > 0.5:
                    # 绘制边界框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 准备标签文本
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # 计算文本大小
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    
                    # 绘制标签背景
                    cv2.rectangle(
                        frame, 
                        (x1, y1 - text_height - 10), 
                        (x1 + text_width, y1), 
                        (0, 255, 0), 
                        -1
                    )
                    
                    # 绘制标签文本
                    cv2.putText(
                        frame, 
                        label, 
                        (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (0, 0, 0), 
                        2
                    )
        
        return frame
    
    def run_detection(self):
        """运行实时检测"""
        print("开始实时检测...")
        print("按 'q' 键退出程序")
        print("按 's' 键保存当前帧")
        
        frame_count = 0
        fps_start_time = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法读取摄像头画面")
                    break
                
                # 进行目标检测
                results = self.model(frame, verbose=False)
                
                # 绘制检测结果
                frame = self.draw_detections(frame, results)
                
                # 计算并显示FPS
                frame_count += 1
                if frame_count % 30 == 0:  # 每30帧计算一次FPS
                    fps_end_time = time.time()
                    fps = 30 / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                    cv2.putText(
                        frame, 
                        f"FPS: {fps:.1f}", 
                        (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (255, 255, 255), 
                        2
                    )
                
                # 显示帧
                cv2.imshow('YOLOv8 实时检测', frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户退出程序")
                    break
                elif key == ord('s'):
                    # 保存当前帧到脚本所在目录
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(script_dir, f"detection_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"已保存图片: {filename}")
                
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"检测过程中出现错误: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("资源清理完成")

def main():
    """主函数"""
    print("=" * 50)
    print("YOLOv8 实时摄像头目标检测")
    print("=" * 50)
    
    # 检查摄像头是否可用
    cap_test = cv2.VideoCapture(0)
    if not cap_test.isOpened():
        print("错误: 无法访问摄像头，请检查摄像头连接")
        return
    cap_test.release()
    
    # 创建检测器实例
    detector = CameraRealtimeDetector()
    
    # 运行检测
    detector.run_detection()

if __name__ == "__main__":
    main()
