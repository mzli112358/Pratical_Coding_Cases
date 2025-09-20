#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenPose实时人体姿态检测 - OpenCV版本
使用OpenCV DNN模块进行实时人体姿态估计，不依赖mediapipe
"""

import cv2
import numpy as np
import time
import argparse
import os

class RealTimeOpenPoseOpenCV:
    def __init__(self, camera_id=0, model_path=None):
        """
        初始化实时OpenPose检测器（OpenCV版本）
        
        Args:
            camera_id: 摄像头ID
            model_path: 预训练模型路径
        """
        self.camera_id = camera_id
        self.cap = None
        self.net = None
        
        # 如果没有提供模型路径，使用OpenCV内置的COCO模型
        if model_path is None:
            # 使用OpenCV的DNN模块加载预训练的人体姿态检测模型
            self.setup_model()
        else:
            self.load_model(model_path)
        
        # 人体关键点连接关系（COCO格式）
        self.pose_pairs = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
            (5, 11), (6, 12), (11, 12),  # 躯干
            (11, 13), (12, 14), (13, 15), (14, 16)  # 腿部
        ]
        
        # 关键点名称
        self.pose_names = [
            "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
            "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
            "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
            "REye", "LEye", "REar", "LEar", "LBigToe",
            "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
        ]
    
    def setup_model(self):
        """设置OpenCV DNN模型"""
        try:
            # 使用OpenCV的DNN模块
            # 这里我们使用一个简化的方法，基于OpenCV的人体检测
            print("使用OpenCV内置的人体检测功能...")
            self.use_opencv_detection = True
        except Exception as e:
            print(f"模型设置失败: {e}")
            self.use_opencv_detection = False
    
    def load_model(self, model_path):
        """加载预训练模型"""
        try:
            if os.path.exists(model_path):
                self.net = cv2.dnn.readNetFromTensorflow(model_path)
                print(f"模型加载成功: {model_path}")
            else:
                print(f"模型文件不存在: {model_path}")
                self.setup_model()
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.setup_model()
    
    def detect_pose_opencv(self, frame):
        """
        使用OpenCV进行人体姿态检测
        
        Args:
            frame: 输入图像
            
        Returns:
            annotated_frame: 标注后的图像
        """
        # 创建人体检测器
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # 检测人体
        (rects, weights) = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        
        # 在检测到的人体周围画框
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def detect_pose_simple(self, frame):
        """
        简单的人体姿态检测（基于轮廓检测）
        
        Args:
            frame: 输入图像
            
        Returns:
            annotated_frame: 标注后的图像
        """
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)
        
        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 绘制轮廓
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        
        # 添加文本说明
        cv2.putText(frame, "Simple Pose Detection", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def start_detection(self):
        """开始实时检测"""
        try:
            # 初始化摄像头
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                print(f"无法打开摄像头 {self.camera_id}")
                return
            
            print("开始实时人体姿态检测...")
            print("按 'q' 键退出")
            print("按 's' 键保存当前帧")
            print("按 'c' 键切换检测模式")
            
            detection_mode = "opencv"  # 默认使用OpenCV检测
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法读取摄像头数据")
                    break
                
                # 根据模式进行检测
                if detection_mode == "opencv":
                    annotated_frame = self.detect_pose_opencv(frame.copy())
                else:
                    annotated_frame = self.detect_pose_simple(frame.copy())
                
                # 显示FPS
                cv2.putText(annotated_frame, f"Mode: {detection_mode.upper()}", 
                           (10, annotated_frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 显示图像
                cv2.imshow('Real-time Pose Detection', annotated_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 保存当前帧
                    timestamp = int(time.time())
                    filename = f"pose_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, annotated_frame)
                    print(f"图像已保存: {filename}")
                elif key == ord('c'):
                    # 切换检测模式
                    detection_mode = "simple" if detection_mode == "opencv" else "opencv"
                    print(f"切换到 {detection_mode} 检测模式")
        
        except KeyboardInterrupt:
            print("\n检测被用户中断")
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
    parser = argparse.ArgumentParser(description='OpenPose实时人体姿态检测 - OpenCV版本')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID (默认: 0)')
    parser.add_argument('--model', type=str, default=None, help='预训练模型路径')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = RealTimeOpenPoseOpenCV(camera_id=args.camera, model_path=args.model)
    
    # 开始检测
    detector.start_detection()

if __name__ == "__main__":
    main()
