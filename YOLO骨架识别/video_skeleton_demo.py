#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频骨架识别演示程序
专门用于视频文件的骨架检测和分析
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
import time
from skeleton_detection import SkeletonDetector

class VideoSkeletonDemo:
    def __init__(self, model_path=None):
        """
        初始化视频骨架演示
        
        Args:
            model_path (str): 模型文件路径
        """
        self.detector = SkeletonDetector(model_path=model_path)
        self.frame_count = 0
        self.total_persons = 0
        self.pose_history = []
        print("视频骨架识别演示程序已初始化")
    
    def detect_and_analyze(self, video_path, output_path=None, analyze_pose=True):
        """
        检测视频中的骨架并进行分析
        
        Args:
            video_path (str): 输入视频路径
            output_path (str): 输出视频路径
            analyze_pose (bool): 是否进行姿态分析
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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height}, {fps} FPS, {total_frames} 帧")
        
        # 设置输出视频
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        self.frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 进行姿态估计
                results = self.detector.model(frame, verbose=False)
                
                # 获取关键点
                if results[0].keypoints is not None:
                    keypoints = results[0].keypoints.data.cpu().numpy()
                    
                    # 绘制骨架
                    frame = self.detector.draw_skeleton(frame, keypoints, confidence_threshold=0.5)
                    
                    # 显示检测到的关键点数量
                    num_persons = len(keypoints)
                    self.total_persons = max(self.total_persons, num_persons)
                    
                    # 添加信息显示
                    self.add_info_overlay(frame, num_persons, fps)
                    
                    # 姿态分析
                    if analyze_pose and len(keypoints) > 0:
                        self.analyze_frame_pose(keypoints, frame)
                
                # 计算处理进度
                progress = (self.frame_count / total_frames) * 100
                elapsed_time = time.time() - start_time
                estimated_total = elapsed_time / (self.frame_count / total_frames) if self.frame_count > 0 else 0
                remaining_time = max(0, estimated_total - elapsed_time)
                
                # 显示进度信息
                cv2.putText(frame, f"进度: {progress:.1f}%", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"剩余时间: {remaining_time:.1f}s", (10, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if output_path:
                    out.write(frame)
                else:
                    # 显示视频
                    cv2.imshow('视频骨架检测', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                self.frame_count += 1
        
        except KeyboardInterrupt:
            print("\n处理被用户中断")
        finally:
            cap.release()
            if output_path:
                out.release()
                print(f"结果已保存到: {output_path}")
            cv2.destroyAllWindows()
            
            # 显示分析结果
            if analyze_pose:
                self.show_analysis_summary()
    
    def add_info_overlay(self, frame, num_persons, fps):
        """
        添加信息覆盖层
        
        Args:
            frame: 视频帧
            num_persons: 检测到的人数
            fps: 视频帧率
        """
        # 显示帧数
        cv2.putText(frame, f"帧数: {self.frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示检测到的人数
        cv2.putText(frame, f"检测到 {num_persons} 个人", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 显示最大人数
        cv2.putText(frame, f"最大人数: {self.total_persons}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    def analyze_frame_pose(self, keypoints, frame):
        """
        分析单帧的姿态
        
        Args:
            keypoints: 关键点数据
            frame: 视频帧
        """
        for person_idx, person_keypoints in enumerate(keypoints):
            # 分析每个人的姿态
            pose_info = self.get_pose_info(person_keypoints)
            self.pose_history.append(pose_info)
            
            # 在帧上显示姿态信息
            y_offset = 170 + person_idx * 60
            cv2.putText(frame, f"人 {person_idx + 1}:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if pose_info['standing']:
                cv2.putText(frame, "站立", (120, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "其他姿态", (120, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if pose_info['arms_raised']:
                cv2.putText(frame, "举手", (200, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def get_pose_info(self, keypoints):
        """
        获取姿态信息
        
        Args:
            keypoints: 单个人的关键点数据
            
        Returns:
            姿态信息字典
        """
        pose_info = {
            'standing': False,
            'arms_raised': False,
            'confidence': 0.0
        }
        
        try:
            # 检查关键点置信度
            valid_keypoints = sum(1 for _, _, conf in keypoints if conf > 0.5)
            pose_info['confidence'] = valid_keypoints / len(keypoints)
            
            # 判断是否站立（基于脚踝和头部的相对位置）
            if (keypoints[15][2] > 0.5 and keypoints[16][2] > 0.5 and  # 脚踝可见
                keypoints[0][2] > 0.5):  # 鼻子可见
                head_y = keypoints[0][1]
                left_ankle_y = keypoints[15][1]
                right_ankle_y = keypoints[16][1]
                avg_ankle_y = (left_ankle_y + right_ankle_y) / 2
                
                # 如果头部在脚踝上方，认为是站立
                if head_y < avg_ankle_y:
                    pose_info['standing'] = True
            
            # 判断是否举手（基于手腕和肩膀的相对位置）
            if (keypoints[5][2] > 0.5 and keypoints[6][2] > 0.5 and  # 肩膀可见
                keypoints[9][2] > 0.5 and keypoints[10][2] > 0.5):  # 手腕可见
                
                left_shoulder_y = keypoints[5][1]
                right_shoulder_y = keypoints[6][1]
                left_wrist_y = keypoints[9][1]
                right_wrist_y = keypoints[10][1]
                
                # 如果手腕在肩膀上方，认为是举手
                if (left_wrist_y < left_shoulder_y or right_wrist_y < right_shoulder_y):
                    pose_info['arms_raised'] = True
                    
        except Exception as e:
            print(f"姿态分析错误: {e}")
        
        return pose_info
    
    def show_analysis_summary(self):
        """
        显示分析摘要
        """
        if not self.pose_history:
            print("没有姿态数据可供分析")
            return
        
        print("\n=== 视频姿态分析摘要 ===")
        print(f"总帧数: {self.frame_count}")
        print(f"最大同时检测人数: {self.total_persons}")
        
        # 统计姿态信息
        standing_frames = sum(1 for pose in self.pose_history if pose['standing'])
        arms_raised_frames = sum(1 for pose in self.pose_history if pose['arms_raised'])
        avg_confidence = sum(pose['confidence'] for pose in self.pose_history) / len(self.pose_history)
        
        print(f"站立帧数: {standing_frames} ({standing_frames/len(self.pose_history)*100:.1f}%)")
        print(f"举手帧数: {arms_raised_frames} ({arms_raised_frames/len(self.pose_history)*100:.1f}%)")
        print(f"平均检测置信度: {avg_confidence:.3f}")
        
        # 分析姿态变化
        self.analyze_pose_changes()
    
    def analyze_pose_changes(self):
        """
        分析姿态变化
        """
        if len(self.pose_history) < 2:
            return
        
        print("\n=== 姿态变化分析 ===")
        
        # 检测姿态变化点
        changes = 0
        for i in range(1, len(self.pose_history)):
            prev_pose = self.pose_history[i-1]
            curr_pose = self.pose_history[i]
            
            # 检查站立状态变化
            if prev_pose['standing'] != curr_pose['standing']:
                changes += 1
                print(f"帧 {i}: 站立状态变化")
            
            # 检查举手状态变化
            if prev_pose['arms_raised'] != curr_pose['arms_raised']:
                changes += 1
                print(f"帧 {i}: 举手状态变化")
        
        print(f"总姿态变化次数: {changes}")
        print(f"平均每帧变化率: {changes/len(self.pose_history)*100:.2f}%")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='视频骨架识别演示')
    parser.add_argument('--input', type=str, required=True, help='输入视频路径')
    parser.add_argument('--output', type=str, help='输出视频路径')
    parser.add_argument('--model', type=str, help='模型文件路径')
    parser.add_argument('--no-analysis', action='store_true', help='不进行姿态分析')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("视频骨架识别演示程序")
    print("=" * 50)
    
    # 创建演示程序
    demo = VideoSkeletonDemo(model_path=args.model)
    
    # 运行检测
    demo.detect_and_analyze(
        args.input, 
        args.output, 
        analyze_pose=not args.no_analysis
    )

if __name__ == "__main__":
    main()
