#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像骨架识别演示程序
专门用于单张图像的骨架检测和可视化
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
import matplotlib.pyplot as plt
from skeleton_detection import SkeletonDetector

class ImageSkeletonDemo:
    def __init__(self, model_path=None):
        """
        初始化图像骨架演示
        
        Args:
            model_path (str): 模型文件路径
        """
        self.detector = SkeletonDetector(model_path=model_path)
        print("图像骨架识别演示程序已初始化")
    
    def detect_and_visualize(self, image_path, output_path=None, show_plot=True):
        """
        检测图像中的骨架并进行可视化
        
        Args:
            image_path (str): 输入图像路径
            output_path (str): 输出图像路径
            show_plot (bool): 是否显示matplotlib图表
        """
        if not os.path.exists(image_path):
            print(f"图像文件 {image_path} 不存在")
            return
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像 {image_path}")
            return
        
        print(f"正在检测图像: {image_path}")
        
        # 进行姿态估计
        results = self.detector.model(image, verbose=False)
        
        # 创建可视化图像
        vis_image = image.copy()
        
        # 获取关键点
        if results[0].keypoints is not None:
            keypoints = results[0].keypoints.data.cpu().numpy()
            
            # 绘制骨架
            vis_image = self.detector.draw_skeleton(vis_image, keypoints, confidence_threshold=0.5)
            
            # 显示检测到的关键点数量
            num_persons = len(keypoints)
            cv2.putText(vis_image, f"检测到 {num_persons} 个人", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 分析关键点数据
            self.analyze_keypoints(keypoints)
            
            # 使用matplotlib进行更详细的可视化
            if show_plot:
                self.matplotlib_visualization(image, keypoints)
        
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, vis_image)
            print(f"结果已保存到: {output_path}")
        
        # 显示OpenCV结果
        cv2.imshow('骨架检测结果', vis_image)
        print("按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def analyze_keypoints(self, keypoints):
        """
        分析关键点数据
        
        Args:
            keypoints: 关键点数据
        """
        print("\n=== 关键点分析 ===")
        
        for i, person_keypoints in enumerate(keypoints):
            print(f"\n第 {i+1} 个人:")
            
            # 关键点名称
            keypoint_names = [
                "鼻子", "左眼", "右眼", "左耳", "右耳",
                "左肩", "右肩", "左肘", "右肘", "左腕", "右腕",
                "左臀", "右臀", "左膝", "右膝", "左踝", "右踝"
            ]
            
            # 分析每个关键点
            for j, (x, y, conf) in enumerate(person_keypoints):
                if conf > 0.5:  # 只显示高置信度的关键点
                    print(f"  {keypoint_names[j]}: 位置({x:.1f}, {y:.1f}), 置信度: {conf:.3f}")
            
            # 计算身体姿态角度
            self.calculate_pose_angles(person_keypoints)
    
    def calculate_pose_angles(self, keypoints):
        """
        计算身体姿态角度
        
        Args:
            keypoints: 单个人的关键点数据
        """
        try:
            # 左臂角度 (肩膀-肘-腕)
            if (keypoints[5][2] > 0.5 and keypoints[7][2] > 0.5 and keypoints[9][2] > 0.5):
                left_arm_angle = self.calculate_angle(
                    keypoints[5][:2], keypoints[7][:2], keypoints[9][:2]
                )
                print(f"  左臂角度: {left_arm_angle:.1f}°")
            
            # 右臂角度
            if (keypoints[6][2] > 0.5 and keypoints[8][2] > 0.5 and keypoints[10][2] > 0.5):
                right_arm_angle = self.calculate_angle(
                    keypoints[6][:2], keypoints[8][:2], keypoints[10][:2]
                )
                print(f"  右臂角度: {right_arm_angle:.1f}°")
            
            # 左腿角度 (臀部-膝-踝)
            if (keypoints[11][2] > 0.5 and keypoints[13][2] > 0.5 and keypoints[15][2] > 0.5):
                left_leg_angle = self.calculate_angle(
                    keypoints[11][:2], keypoints[13][:2], keypoints[15][:2]
                )
                print(f"  左腿角度: {left_leg_angle:.1f}°")
            
            # 右腿角度
            if (keypoints[12][2] > 0.5 and keypoints[14][2] > 0.5 and keypoints[16][2] > 0.5):
                right_leg_angle = self.calculate_angle(
                    keypoints[12][:2], keypoints[14][:2], keypoints[16][:2]
                )
                print(f"  右腿角度: {right_leg_angle:.1f}°")
                
        except Exception as e:
            print(f"  角度计算错误: {e}")
    
    def calculate_angle(self, p1, p2, p3):
        """
        计算三点之间的角度
        
        Args:
            p1, p2, p3: 三个点的坐标
            
        Returns:
            角度（度）
        """
        # 计算向量
        v1 = np.array(p1) - np.array(p2)
        v2 = np.array(p3) - np.array(p2)
        
        # 计算角度
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免数值误差
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def matplotlib_visualization(self, image, keypoints):
        """
        使用matplotlib进行详细可视化
        
        Args:
            image: 原始图像
            keypoints: 关键点数据
        """
        # 转换BGR到RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 显示原始图像
        ax1.imshow(image_rgb)
        ax1.set_title('原始图像')
        ax1.axis('off')
        
        # 显示骨架检测结果
        ax2.imshow(image_rgb)
        
        # 关键点名称
        keypoint_names = [
            "鼻子", "左眼", "右眼", "左耳", "右耳",
            "左肩", "右肩", "左肘", "右肘", "左腕", "右腕",
            "左臀", "右臀", "左膝", "右膝", "左踝", "右踝"
        ]
        
        # 绘制关键点和连接
        colors = plt.cm.tab20(np.linspace(0, 1, len(keypoint_names)))
        
        for person_idx, person_keypoints in enumerate(keypoints):
            # 绘制关键点
            for i, (x, y, conf) in enumerate(person_keypoints):
                if conf > 0.5:
                    ax2.scatter(x, y, c=[colors[i]], s=100, alpha=0.8)
                    ax2.annotate(f"{i}", (x, y), xytext=(5, 5), 
                               textcoords='offset points', fontsize=8)
            
            # 绘制骨架连接
            skeleton_connections = [
                (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
                (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (9, 11),  # 上身
                (5, 11), (6, 12), (11, 12),  # 躯干
                (11, 13), (12, 14), (13, 15), (14, 16)  # 下身
            ]
            
            for start_idx, end_idx in skeleton_connections:
                if (start_idx < len(person_keypoints) and end_idx < len(person_keypoints)):
                    start_point = person_keypoints[start_idx]
                    end_point = person_keypoints[end_idx]
                    
                    if (start_point[2] > 0.5 and end_point[2] > 0.5):
                        ax2.plot([start_point[0], end_point[0]], 
                                [start_point[1], end_point[1]], 
                                'r-', linewidth=2, alpha=0.8)
        
        ax2.set_title('骨架检测结果')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='图像骨架识别演示')
    parser.add_argument('--input', type=str, required=True, help='输入图像路径')
    parser.add_argument('--output', type=str, help='输出图像路径')
    parser.add_argument('--model', type=str, help='模型文件路径')
    parser.add_argument('--no-plot', action='store_true', help='不显示matplotlib图表')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("图像骨架识别演示程序")
    print("=" * 50)
    
    # 创建演示程序
    demo = ImageSkeletonDemo(model_path=args.model)
    
    # 运行检测
    demo.detect_and_visualize(
        args.input, 
        args.output, 
        show_plot=not args.no_plot
    )

if __name__ == "__main__":
    main()
