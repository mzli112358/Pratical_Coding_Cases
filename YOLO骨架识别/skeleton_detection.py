#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO骨架识别系统
支持实时摄像头和视频文件的人体姿态估计
使用YOLOv8-pose模型进行关键点检测
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import sys
import argparse

class SkeletonDetector:
    def __init__(self, model_path=None, camera_index=0):
        """
        初始化骨架检测器
        
        Args:
            model_path (str): YOLOv8-pose模型文件路径，如果为None则自动下载
            camera_index (int): 摄像头索引，默认为0
        """
        self.model_path = model_path
        self.camera_index = camera_index
        self.model = None
        self.cap = None
        
        # 人体关键点连接关系 (COCO格式)
        self.skeleton_connections = [
            # 头部连接
            (0, 1), (0, 2), (1, 3), (2, 4),  # 鼻子-眼睛-耳朵
            # 躯干连接
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11),  # 肩膀-手臂
            (5, 11), (6, 12), (11, 12),  # 躯干
            # 腿部连接
            (11, 13), (12, 14), (13, 15), (14, 16)  # 臀部-膝盖-脚踝
        ]
        
        # 关键点颜色 (BGR格式)
        self.keypoint_colors = [
            (0, 255, 255),    # 鼻子 - 黄色
            (255, 0, 0),      # 左眼 - 蓝色
            (255, 0, 0),      # 右眼 - 蓝色
            (0, 255, 0),      # 左耳 - 绿色
            (0, 255, 0),      # 右耳 - 绿色
            (255, 255, 0),    # 左肩 - 青色
            (255, 255, 0),    # 右肩 - 青色
            (255, 0, 255),    # 左肘 - 洋红
            (255, 0, 255),    # 右肘 - 洋红
            (0, 255, 255),    # 左腕 - 黄色
            (0, 255, 255),    # 右腕 - 黄色
            (128, 0, 128),    # 左臀 - 紫色
            (128, 0, 128),    # 右臀 - 紫色
            (0, 128, 255),    # 左膝 - 橙色
            (0, 128, 255),    # 右膝 - 橙色
            (255, 192, 203),  # 左踝 - 粉色
            (255, 192, 203)   # 右踝 - 粉色
        ]
        
        # 骨架连接颜色
        self.skeleton_colors = [
            (0, 255, 0),      # 绿色
            (255, 0, 0),      # 蓝色
            (0, 0, 255),      # 红色
            (255, 255, 0),    # 青色
            (255, 0, 255),    # 洋红
            (0, 255, 255),    # 黄色
            (128, 0, 128),    # 紫色
            (0, 128, 255),    # 橙色
            (255, 192, 203),  # 粉色
            (0, 128, 128)     # 深青色
        ]
        
        print("正在初始化骨架检测器...")
        
        # 如果没有提供模型路径，则使用YOLOv8-pose
        if model_path is None:
            print("使用YOLOv8-pose模型...")
            self.model = YOLO('yolov8n-pose.pt')
        else:
            if not os.path.exists(model_path):
                print(f"模型文件 {model_path} 不存在，使用默认模型...")
                self.model = YOLO('yolov8n-pose.pt')
            else:
                self.model = YOLO(model_path)
        
        print("模型加载成功！")
    
    def draw_skeleton(self, frame, keypoints, confidence_threshold=0.5):
        """
        在帧上绘制骨架
        
        Args:
            frame: 输入帧
            keypoints: 关键点坐标和置信度
            confidence_threshold: 置信度阈值
            
        Returns:
            绘制了骨架的帧
        """
        if keypoints is None or len(keypoints) == 0:
            return frame
        
        # 绘制关键点
        for person_keypoints in keypoints:
            # 绘制关键点
            for i, (x, y, conf) in enumerate(person_keypoints):
                if conf > confidence_threshold:
                    # 绘制关键点圆圈
                    cv2.circle(frame, (int(x), int(y)), 5, self.keypoint_colors[i % len(self.keypoint_colors)], -1)
                    # 绘制关键点编号
                    cv2.putText(frame, str(i), (int(x) + 5, int(y) - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 绘制骨架连接
            for i, (start_idx, end_idx) in enumerate(self.skeleton_connections):
                if (start_idx < len(person_keypoints) and end_idx < len(person_keypoints)):
                    start_point = person_keypoints[start_idx]
                    end_point = person_keypoints[end_idx]
                    
                    # 检查两个关键点的置信度
                    if (start_point[2] > confidence_threshold and end_point[2] > confidence_threshold):
                        start_pos = (int(start_point[0]), int(start_point[1]))
                        end_pos = (int(end_point[0]), int(end_point[1]))
                        
                        # 绘制连接线
                        color = self.skeleton_colors[i % len(self.skeleton_colors)]
                        cv2.line(frame, start_pos, end_pos, color, 2)
        
        return frame
    
    def detect_skeleton_image(self, image_path, output_path=None, confidence_threshold=0.5):
        """
        检测图像中的骨架
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径，如果为None则显示图像
            confidence_threshold: 置信度阈值
        """
        if not os.path.exists(image_path):
            print(f"图像文件 {image_path} 不存在")
            return
        
        # 读取图像
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"无法读取图像 {image_path}")
            return
        
        print(f"正在检测图像: {image_path}")
        
        # 进行姿态估计
        results = self.model(frame, verbose=False)
        
        # 获取关键点
        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.data.cpu().numpy()
            
            # 绘制骨架
            frame = self.draw_skeleton(frame, keypoints, confidence_threshold)
            
            # 显示检测到的关键点数量
            num_persons = len(keypoints)
            cv2.putText(frame, f"检测到 {num_persons} 个人", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, frame)
            print(f"结果已保存到: {output_path}")
        else:
            # 显示图像
            cv2.imshow('骨架检测结果', frame)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def detect_skeleton_video(self, video_path, output_path=None, confidence_threshold=0.5):
        """
        检测视频中的骨架
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径，如果为None则显示视频
            confidence_threshold: 置信度阈值
        """
        if not os.path.exists(video_path):
            print(f"视频文件 {video_path} 不存在")
            return
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频 {video_path}")
            return
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"视频信息: {width}x{height}, {fps} FPS")
        
        # 设置输出视频
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 进行姿态估计
                results = self.model(frame, verbose=False)
                
                # 获取关键点
                if results[0].keypoints is not None:
                    keypoints = results[0].keypoints.data.cpu().numpy()
                    
                    # 绘制骨架
                    frame = self.draw_skeleton(frame, keypoints, confidence_threshold)
                    
                    # 显示检测到的关键点数量
                    num_persons = len(keypoints)
                    cv2.putText(frame, f"检测到 {num_persons} 个人", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 计算并显示FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed_time = time.time() - start_time
                    current_fps = frame_count / elapsed_time
                    cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                if output_path:
                    out.write(frame)
                else:
                    # 显示视频
                    cv2.imshow('骨架检测视频', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        except KeyboardInterrupt:
            print("\n处理被用户中断")
        finally:
            cap.release()
            if output_path:
                out.release()
                print(f"结果已保存到: {output_path}")
            cv2.destroyAllWindows()
    
    def detect_skeleton_camera(self, confidence_threshold=0.5):
        """
        实时检测摄像头中的骨架
        
        Args:
            confidence_threshold: 置信度阈值
        """
        # 初始化摄像头
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"无法打开摄像头 {self.camera_index}")
            return
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("开始实时骨架检测...")
        print("按 'q' 键退出程序")
        print("按 's' 键保存当前帧")
        
        frame_count = 0
        fps_start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头画面")
                    break
                
                # 进行姿态估计
                results = self.model(frame, verbose=False)
                
                # 获取关键点
                if results[0].keypoints is not None:
                    keypoints = results[0].keypoints.data.cpu().numpy()
                    
                    # 绘制骨架
                    frame = self.draw_skeleton(frame, keypoints, confidence_threshold)
                    
                    # 显示检测到的关键点数量
                    num_persons = len(keypoints)
                    cv2.putText(frame, f"检测到 {num_persons} 个人", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # 计算并显示FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    fps_end_time = time.time()
                    fps = 30 / (fps_end_time - fps_start_time)
                    fps_start_time = fps_end_time
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # 显示帧
                cv2.imshow('实时骨架检测', frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户退出程序")
                    break
                elif key == ord('s'):
                    # 保存当前帧
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"skeleton_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"已保存图片: {filename}")
        
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"检测过程中出现错误: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("资源清理完成")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='YOLO骨架识别系统')
    parser.add_argument('--mode', type=str, choices=['image', 'video', 'camera'], 
                       default='camera', help='检测模式: image/video/camera')
    parser.add_argument('--input', type=str, help='输入文件路径')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--model', type=str, help='模型文件路径')
    parser.add_argument('--camera', type=int, default=0, help='摄像头索引')
    parser.add_argument('--confidence', type=float, default=0.5, help='置信度阈值')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("YOLO骨架识别系统")
    print("=" * 50)
    
    # 创建检测器
    detector = SkeletonDetector(model_path=args.model, camera_index=args.camera)
    
    if args.mode == 'image':
        if not args.input:
            print("图像模式需要指定输入文件路径 (--input)")
            return
        detector.detect_skeleton_image(args.input, args.output, args.confidence)
    
    elif args.mode == 'video':
        if not args.input:
            print("视频模式需要指定输入文件路径 (--input)")
            return
        detector.detect_skeleton_video(args.input, args.output, args.confidence)
    
    elif args.mode == 'camera':
        detector.detect_skeleton_camera(args.confidence)

if __name__ == "__main__":
    main()
