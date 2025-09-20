#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版YOLO骨架识别应用
包含完整的UI界面、录制功能和卡通形象绑定
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import threading
import json
from datetime import datetime
from cartoon_character import CartoonCharacter

class EnhancedSkeletonApp:
    def __init__(self, model_path=None, camera_index=0):
        """
        初始化增强版骨架应用
        
        Args:
            model_path (str): YOLOv8-pose模型文件路径
            camera_index (int): 摄像头索引
        """
        self.model_path = model_path
        self.camera_index = camera_index
        self.model = None
        self.cap = None
        
        # 录制状态
        self.is_recording = False
        self.recording_start_time = None
        self.recording_delay = 5.0  # 5秒延迟开始录制
        self.recording_data = []
        
        # 卡通形象
        self.cartoon_character = CartoonCharacter()
        self.show_cartoon = False
        self.current_character_type = 0
        self.character_types = self.cartoon_character.get_character_types()
        
        # 显示设置
        self.show_skeleton = True
        self.show_keypoints = True
        self.confidence_threshold = 0.5
        
        # 统计信息
        self.frame_count = 0
        self.fps = 0
        self.fps_start_time = time.time()
        
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
        
        print("正在初始化增强版骨架识别应用...")
        self._initialize_model()
        self._initialize_camera()
        
    def _initialize_model(self):
        """初始化YOLO模型"""
        try:
            if self.model_path is None:
                print("使用YOLOv8-pose模型...")
                self.model = YOLO('yolov8n-pose.pt')
            else:
                if not os.path.exists(self.model_path):
                    print(f"模型文件 {self.model_path} 不存在，使用默认模型...")
                    self.model = YOLO('yolov8n-pose.pt')
                else:
                    self.model = YOLO(self.model_path)
            print("模型加载成功！")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def _initialize_camera(self):
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise Exception(f"无法打开摄像头 {self.camera_index}")
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            print("摄像头初始化成功！")
        except Exception as e:
            print(f"摄像头初始化失败: {e}")
            raise
    
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
            if self.show_keypoints:
                # 绘制关键点
                for i, (x, y, conf) in enumerate(person_keypoints):
                    if conf > confidence_threshold:
                        # 绘制关键点圆圈
                        cv2.circle(frame, (int(x), int(y)), 5, self.keypoint_colors[i % len(self.keypoint_colors)], -1)
                        # 绘制关键点编号
                        cv2.putText(frame, str(i), (int(x) + 5, int(y) - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            if self.show_skeleton:
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
    
    def draw_ui(self, frame):
        """
        绘制用户界面
        
        Args:
            frame: 输入帧
            
        Returns:
            绘制了UI的帧
        """
        height, width = frame.shape[:2]
        
        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (width - 10, 160), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 绘制状态信息
        status_text = "录制中" if self.is_recording else "待机中"
        status_color = (0, 0, 255) if self.is_recording else (0, 255, 0)
        cv2.putText(frame, f"状态: {status_text}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # 绘制录制倒计时
        if self.is_recording and self.recording_start_time:
            elapsed = time.time() - self.recording_start_time
            if elapsed < self.recording_delay:
                countdown = int(self.recording_delay - elapsed)
                cv2.putText(frame, f"录制倒计时: {countdown}秒", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "正在录制...", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 绘制卡通形象状态
        cartoon_text = f"卡通形象: {self.character_types[self.current_character_type]}" if self.show_cartoon else "卡通形象: 关闭"
        cv2.putText(frame, cartoon_text, (20, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # 绘制控制说明
        cv2.putText(frame, "控制: r-录制, c-卡通, s-骨架, k-关键点, t-类型, q-退出", (20, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def start_recording(self):
        """开始录制"""
        if not self.is_recording:
            self.is_recording = True
            self.recording_start_time = time.time()
            self.recording_data = []
            print("开始录制骨架数据...")
    
    def stop_recording(self):
        """停止录制"""
        if self.is_recording:
            self.is_recording = False
            self.recording_start_time = None
            self.save_recording_data()
            print("停止录制，数据已保存")
    
    def save_recording_data(self):
        """保存录制数据"""
        if not self.recording_data:
            print("没有录制数据可保存")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.json"
        filepath = os.path.join("recordings", filename)
        
        # 创建recordings目录
        os.makedirs("recordings", exist_ok=True)
        
        # 保存数据
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.recording_data, f, ensure_ascii=False, indent=2)
        
        print(f"录制数据已保存到: {filepath}")
    
    def process_frame(self, frame, confidence_threshold=0.5):
        """
        处理单帧图像
        
        Args:
            frame: 输入帧
            confidence_threshold: 置信度阈值
            
        Returns:
            处理后的帧
        """
        # 进行姿态估计
        results = self.model(frame, verbose=False)
        
        # 获取关键点
        keypoints = None
        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.data.cpu().numpy()
            
            # 绘制骨架
            frame = self.draw_skeleton(frame, keypoints, confidence_threshold)
            
            # 显示检测到的关键点数量
            num_persons = len(keypoints)
            cv2.putText(frame, f"检测到 {num_persons} 个人", (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 录制数据
            if self.is_recording and self.recording_start_time:
                elapsed = time.time() - self.recording_start_time
                if elapsed >= self.recording_delay:
                    # 记录关键点数据
                    frame_data = {
                        'timestamp': time.time(),
                        'keypoints': keypoints.tolist() if keypoints is not None else []
                    }
                    self.recording_data.append(frame_data)
            
            # 绘制卡通形象
            if self.show_cartoon and keypoints is not None and len(keypoints) > 0:
                # 使用第一个检测到的人的关键点
                person_keypoints = keypoints[0]
                self.cartoon_character.update_pose(person_keypoints)
                frame = self.cartoon_character.draw(frame)
        
        # 绘制UI
        frame = self.draw_ui(frame)
        
        # 计算FPS
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            fps_end_time = time.time()
            self.fps = 30 / (fps_end_time - self.fps_start_time)
            self.fps_start_time = fps_end_time
        
        # 显示FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 220), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame
    
    def run(self):
        """运行主程序"""
        print("开始运行增强版骨架识别应用...")
        print("控制说明:")
        print("  - 按 'r' 键开始/停止录制")
        print("  - 按 'c' 键切换卡通形象显示")
        print("  - 按 's' 键切换骨架显示")
        print("  - 按 'k' 键切换关键点显示")
        print("  - 按 't' 键切换卡通形象类型")
        print("  - 按 'q' 键退出程序")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("无法读取摄像头画面")
                    break
                
                # 处理帧
                frame = self.process_frame(frame, self.confidence_threshold)
                
                # 显示帧
                cv2.imshow('增强版YOLO骨架识别应用', frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("用户退出程序")
                    break
                elif key == ord('r'):
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording()
                elif key == ord('c'):
                    self.show_cartoon = not self.show_cartoon
                    print(f"卡通形象显示: {'开启' if self.show_cartoon else '关闭'}")
                elif key == ord('s'):
                    self.show_skeleton = not self.show_skeleton
                    print(f"骨架显示: {'开启' if self.show_skeleton else '关闭'}")
                elif key == ord('k'):
                    self.show_keypoints = not self.show_keypoints
                    print(f"关键点显示: {'开启' if self.show_keypoints else '关闭'}")
                elif key == ord('t'):
                    self.current_character_type = (self.current_character_type + 1) % len(self.character_types)
                    self.cartoon_character.set_character_type(self.character_types[self.current_character_type])
                    print(f"卡通形象类型: {self.character_types[self.current_character_type]}")
        
        except KeyboardInterrupt:
            print("\n程序被用户中断")
        except Exception as e:
            print(f"运行过程中出现错误: {e}")
        finally:
            if self.is_recording:
                self.stop_recording()
            self.cap.release()
            cv2.destroyAllWindows()
            print("资源清理完成")

def main():
    """主函数"""
    print("=" * 60)
    print("           增强版YOLO骨架识别应用")
    print("=" * 60)
    
    try:
        app = EnhancedSkeletonApp()
        app.run()
    except Exception as e:
        print(f"应用启动失败: {e}")

if __name__ == "__main__":
    main()
