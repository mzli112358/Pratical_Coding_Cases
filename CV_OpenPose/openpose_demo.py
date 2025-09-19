#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenPose人体姿态估计演示脚本
使用MediaPipe实现轻量级的OpenPose功能
"""

import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os

class OpenPoseDetector:
    def __init__(self):
        """初始化OpenPose检测器"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 初始化姿态检测模型
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
    
    def detect_pose(self, image):
        """
        检测图像中的人体姿态
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            results: MediaPipe姿态检测结果
            annotated_image: 标注后的图像
        """
        # 转换BGR到RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 进行姿态检测
        results = self.pose.process(rgb_image)
        
        # 绘制姿态关键点
        annotated_image = rgb_image.copy()
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return results, annotated_image
    
    def get_landmark_coordinates(self, results, image_shape):
        """
        获取关键点坐标
        
        Args:
            results: MediaPipe检测结果
            image_shape: 图像尺寸 (height, width)
            
        Returns:
            landmarks: 关键点坐标列表
        """
        landmarks = []
        if results.pose_landmarks:
            height, width = image_shape[:2]
            for landmark in results.pose_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                landmarks.append((x, y))
        return landmarks
    
    def visualize_pose_2d(self, landmarks, image_shape, save_path=None):
        """
        2D可视化人体姿态
        
        Args:
            landmarks: 关键点坐标列表
            image_shape: 图像尺寸
            save_path: 保存路径（可选）
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 绘制关键点
        if landmarks:
            x_coords = [landmark[0] for landmark in landmarks]
            y_coords = [landmark[1] for landmark in landmarks]
            
            # 绘制关键点
            ax.scatter(x_coords, y_coords, c='red', s=50, alpha=0.7)
            
            # 绘制骨架连接
            connections = [
                (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
                (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
                (11, 23), (12, 24), (23, 24), (23, 25), (24, 26),
                (25, 27), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32),
                (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
                (5, 6), (6, 8), (9, 10)
            ]
            
            for connection in connections:
                if (connection[0] < len(landmarks) and 
                    connection[1] < len(landmarks)):
                    x1, y1 = landmarks[connection[0]]
                    x2, y2 = landmarks[connection[1]]
                    ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2)
        
        ax.set_xlim(0, image_shape[1])
        ax.set_ylim(image_shape[0], 0)  # 翻转Y轴
        ax.set_aspect('equal')
        ax.set_title('OpenPose人体姿态检测结果', fontsize=14)
        ax.set_xlabel('X坐标')
        ax.set_ylabel('Y坐标')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"可视化结果已保存到: {save_path}")
        
        plt.show()

def main():
    """主函数"""
    print("OpenPose人体姿态估计演示")
    print("=" * 50)
    
    # 初始化检测器
    detector = OpenPoseDetector()
    
    # 检查是否有测试图像
    test_image_path = "test_image.jpg"
    if not os.path.exists(test_image_path):
        print("未找到测试图像，请将图像文件命名为 'test_image.jpg' 并放在当前目录")
        print("或者使用摄像头进行实时检测...")
        
        # 实时摄像头检测
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return
        
        print("按 'q' 键退出实时检测")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 检测姿态
            results, annotated_image = detector.detect_pose(frame)
            
            # 显示结果
            cv2.imshow('OpenPose实时检测', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # 处理静态图像
    print(f"处理图像: {test_image_path}")
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"无法读取图像: {test_image_path}")
        return
    
    # 检测姿态
    results, annotated_image = detector.detect_pose(image)
    
    # 获取关键点坐标
    landmarks = detector.get_landmark_coordinates(results, image.shape)
    
    if landmarks:
        print(f"检测到 {len(landmarks)} 个关键点")
        
        # 保存标注后的图像
        output_path = "pose_detection_result.jpg"
        cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        print(f"检测结果已保存到: {output_path}")
        
        # 2D可视化
        detector.visualize_pose_2d(landmarks, image.shape, "pose_visualization.png")
        
        # 打印关键点信息
        print("\n关键点坐标:")
        pose_landmarks = [
            "鼻子", "左眼内角", "左眼", "左眼外角", "右眼内角", "右眼", "右眼外角",
            "左耳", "右耳", "嘴左", "嘴右", "左肩", "右肩", "左肘", "右肘",
            "左腕", "右腕", "左小指", "右小指", "左食指", "右食指", "左拇指", "右拇指",
            "左髋", "右髋", "左膝", "右膝", "左踝", "右踝", "左跟", "右跟", "左脚趾", "右脚趾"
        ]
        
        for i, (landmark, name) in enumerate(zip(landmarks, pose_landmarks)):
            print(f"{i:2d}. {name:8s}: ({landmark[0]:4d}, {landmark[1]:4d})")
    else:
        print("未检测到人体姿态")

if __name__ == "__main__":
    main()
