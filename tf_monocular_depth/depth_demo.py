#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单目深度估计演示脚本
支持图像、视频和实时摄像头深度估计
"""

import cv2
import numpy as np
import argparse
import os
import time
from depth_estimation_model import MonocularDepthEstimator


class DepthEstimationDemo:
    """深度估计演示类"""
    
    def __init__(self, model_path=None, camera_id=0, use_pretrained=True):
        """
        初始化演示
        
        Args:
            model_path: 预训练模型路径（可选）
            camera_id: 摄像头ID
            use_pretrained: 是否使用预训练模型
        """
        self.camera_id = camera_id
        
        # 创建深度估计器
        self.depth_estimator = MonocularDepthEstimator(use_pretrained=use_pretrained)
        
        # 如果有预训练模型，加载它
        if model_path and os.path.exists(model_path):
            self.depth_estimator.load_model(model_path)
        elif not use_pretrained:
            # 编译新模型（仅在自定义模式下）
            self.depth_estimator.compile_model()
            print("警告: 使用未训练的模型，预测结果可能不准确")
        
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
    
    def process_image(self, image_path, output_path=None):
        """
        处理单张图像
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径（可选）
        """
        if not os.path.exists(image_path):
            print(f"错误: 图像文件 {image_path} 不存在")
            return
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法读取图像 {image_path}")
            return
        
        print(f"正在处理图像: {image_path}")
        
        # 预测深度
        depth_map = self.depth_estimator.predict_depth(image)
        
        # 后处理深度图
        depth_processed = self.depth_estimator.postprocess_depth(depth_map)
        
        # 生成彩色深度图
        depth_colored = self.depth_estimator.visualize_depth(depth_processed)
        
        # 调整深度图尺寸以匹配原图
        depth_resized = cv2.resize(depth_colored, (image.shape[1], image.shape[0]))
        
        # 创建对比显示
        comparison = np.hstack([image, depth_resized])
        
        # 显示结果
        cv2.imshow("原图 | 深度图", comparison)
        print("按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, comparison)
            print(f"结果已保存到: {output_path}")
    
    def process_video(self, video_path, output_path=None):
        """
        处理视频文件
        
        Args:
            video_path: 输入视频路径
            output_path: 输出视频路径（可选）
        """
        if not os.path.exists(video_path):
            print(f"错误: 视频文件 {video_path} 不存在")
            return
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"错误: 无法打开视频 {video_path}")
            return
        
        # 获取视频属性
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"视频信息: {width}x{height}, {fps}fps")
        
        # 设置输出视频写入器
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            print(f"处理进度: {frame_count}/{total_frames}", end='\r')
            
            # 预测深度
            depth_map = self.depth_estimator.predict_depth(frame)
            depth_processed = self.depth_estimator.postprocess_depth(depth_map)
            depth_colored = self.depth_estimator.visualize_depth(depth_processed)
            
            # 调整深度图尺寸
            depth_resized = cv2.resize(depth_colored, (width, height))
            
            # 创建对比显示
            comparison = np.hstack([frame, depth_resized])
            
            # 显示结果
            cv2.imshow("视频深度估计", comparison)
            
            # 保存视频
            if writer:
                writer.write(comparison)
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"\n视频处理完成")
        if output_path:
            print(f"输出视频已保存到: {output_path}")
    
    def process_camera(self):
        """处理实时摄像头"""
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"错误: 无法打开摄像头 {self.camera_id}")
            return
        
        print("实时深度估计开始，按'q'退出")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("错误: 无法读取摄像头数据")
                break
            
            # 预测深度
            depth_map = self.depth_estimator.predict_depth(frame)
            depth_processed = self.depth_estimator.postprocess_depth(depth_map)
            depth_colored = self.depth_estimator.visualize_depth(depth_processed)
            
            # 调整深度图尺寸
            depth_resized = cv2.resize(depth_colored, (frame.shape[1], frame.shape[0]))
            
            # 计算FPS
            self.calculate_fps()
            
            # 添加FPS信息
            cv2.putText(frame, f'FPS: {self.fps}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(depth_resized, f'Depth Map', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # 创建对比显示
            comparison = np.hstack([frame, depth_resized])
            
            # 显示结果
            cv2.imshow("实时深度估计", comparison)
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("实时深度估计结束")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='单目深度估计演示')
    parser.add_argument('--mode', choices=['image', 'video', 'camera'], 
                       default='camera', help='运行模式')
    parser.add_argument('--input', type=str, help='输入文件路径')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--model', type=str, help='预训练模型路径')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID')
    parser.add_argument('--pretrained', action='store_true', help='使用预训练模型')
    parser.add_argument('--no-pretrained', action='store_true', help='不使用预训练模型')
    
    args = parser.parse_args()
    
    # 确定是否使用预训练模型
    use_pretrained = True  # 默认使用预训练模型
    if args.no_pretrained:
        use_pretrained = False
    elif args.pretrained:
        use_pretrained = True
    
    # 创建演示实例
    demo = DepthEstimationDemo(model_path=args.model, camera_id=args.camera, use_pretrained=use_pretrained)
    
    if args.mode == 'image':
        if not args.input:
            print("错误: 图像模式需要指定输入文件")
            return
        demo.process_image(args.input, args.output)
        
    elif args.mode == 'video':
        if not args.input:
            print("错误: 视频模式需要指定输入文件")
            return
        demo.process_video(args.input, args.output)
        
    elif args.mode == 'camera':
        demo.process_camera()


if __name__ == "__main__":
    main()
