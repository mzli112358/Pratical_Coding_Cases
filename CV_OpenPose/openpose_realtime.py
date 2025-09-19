#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenPose实时人体姿态检测
使用摄像头进行实时人体姿态估计
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import argparse

class RealTimeOpenPose:
    def __init__(self, camera_id=0, model_complexity=2):
        """
        初始化实时OpenPose检测器
        
        Args:
            camera_id: 摄像头ID
            model_complexity: 模型复杂度 (0, 1, 2)
        """
        self.camera_id = camera_id
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 初始化姿态检测模型
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # 性能统计
        self.fps_counter = 0
        self.start_time = time.time()
        self.fps = 0
        
    def calculate_fps(self):
        """计算FPS"""
        self.fps_counter += 1
        current_time = time.time()
        if current_time - self.start_time >= 1.0:
            self.fps = self.fps_counter
            self.fps_counter = 0
            self.start_time = current_time
    
    def draw_fps(self, image, fps):
        """在图像上绘制FPS"""
        cv2.putText(image, f'FPS: {fps}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    def draw_landmark_info(self, image, results):
        """绘制关键点信息"""
        if results.pose_landmarks:
            height, width = image.shape[:2]
            
            # 关键点名称
            landmark_names = [
                "鼻子", "左眼内", "左眼", "左眼外", "右眼内", "右眼", "右眼外",
                "左耳", "右耳", "嘴左", "嘴右", "左肩", "右肩", "左肘", "右肘",
                "左腕", "右腕", "左小指", "右小指", "左食指", "右食指", "左拇指", "右拇指",
                "左髋", "右髋", "左膝", "右膝", "左踝", "右踝", "左跟", "右跟", "左脚趾", "右脚趾"
            ]
            
            # 显示关键点数量
            num_landmarks = len(results.pose_landmarks.landmark)
            cv2.putText(image, f'关键点数量: {num_landmarks}', (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示置信度信息
            if hasattr(results.pose_landmarks, 'landmark'):
                confidences = []
                for landmark in results.pose_landmarks.landmark:
                    if hasattr(landmark, 'visibility'):
                        confidences.append(landmark.visibility)
                
                if confidences:
                    avg_confidence = np.mean(confidences)
                    cv2.putText(image, f'平均置信度: {avg_confidence:.2f}', (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def process_frame(self, frame):
        """
        处理单帧图像
        
        Args:
            frame: 输入帧
            
        Returns:
            processed_frame: 处理后的帧
            results: 检测结果
        """
        # 转换BGR到RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 进行姿态检测
        results = self.pose.process(rgb_frame)
        
        # 绘制姿态关键点
        annotated_frame = rgb_frame.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # 转换回BGR
        processed_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        
        return processed_frame, results
    
    def run(self, show_info=True, record_video=False, output_path="openpose_output.mp4"):
        """
        运行实时检测
        
        Args:
            show_info: 是否显示详细信息
            record_video: 是否录制视频
            output_path: 视频输出路径
        """
        # 打开摄像头
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"无法打开摄像头 {self.camera_id}")
            return
        
        # 设置摄像头参数
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # 视频录制设置
        video_writer = None
        if record_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"开始录制视频到: {output_path}")
        
        print("OpenPose实时检测启动")
        print("按 'q' 键退出")
        print("按 's' 键保存当前帧")
        print("按 'r' 键开始/停止录制")
        
        frame_count = 0
        recording = False
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取摄像头数据")
                    break
                
                # 处理帧
                processed_frame, results = self.process_frame(frame)
                
                # 计算FPS
                self.calculate_fps()
                
                # 绘制信息
                if show_info:
                    self.draw_fps(processed_frame, self.fps)
                    self.draw_landmark_info(processed_frame, results)
                
                # 录制视频
                if record_video and video_writer is not None:
                    video_writer.write(processed_frame)
                
                # 显示结果
                cv2.imshow('OpenPose实时检测', processed_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # 保存当前帧
                    filename = f"openpose_frame_{frame_count:06d}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"已保存帧: {filename}")
                elif key == ord('r'):
                    # 切换录制状态
                    recording = not recording
                    if recording:
                        print("开始录制...")
                    else:
                        print("停止录制")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n检测被用户中断")
        
        finally:
            # 清理资源
            cap.release()
            if video_writer is not None:
                video_writer.release()
            cv2.destroyAllWindows()
            print("检测结束")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='OpenPose实时人体姿态检测')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID (默认: 0)')
    parser.add_argument('--model', type=int, default=2, choices=[0, 1, 2], 
                       help='模型复杂度 (0=轻量, 1=平衡, 2=高精度, 默认: 2)')
    parser.add_argument('--record', action='store_true', help='录制视频')
    parser.add_argument('--output', type=str, default='openpose_output.mp4', 
                       help='输出视频路径 (默认: openpose_output.mp4)')
    parser.add_argument('--no-info', action='store_true', help='不显示详细信息')
    
    args = parser.parse_args()
    
    # 创建检测器
    detector = RealTimeOpenPose(
        camera_id=args.camera,
        model_complexity=args.model
    )
    
    # 运行检测
    detector.run(
        show_info=not args.no_info,
        record_video=args.record,
        output_path=args.output
    )

if __name__ == "__main__":
    main()
