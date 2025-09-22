#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度估计模型训练脚本
使用合成数据训练单目深度估计模型
"""

import tensorflow as tf
import numpy as np
import cv2
import os
import argparse
from depth_estimation_model import MonocularDepthEstimator


class DepthModelTrainer:
    """深度模型训练器"""
    
    def __init__(self, input_height=480, input_width=640):
        """
        初始化训练器
        
        Args:
            input_height: 输入图像高度
            input_width: 输入图像宽度
        """
        self.input_height = input_height
        self.input_width = input_width
        self.depth_estimator = MonocularDepthEstimator(input_height, input_width)
        self.depth_estimator.compile_model()
        
    def generate_synthetic_data(self, num_samples=1000):
        """
        生成合成训练数据
        
        Args:
            num_samples: 生成样本数量
            
        Returns:
            (images, depth_maps): 图像和对应的深度图
        """
        print(f"生成 {num_samples} 个合成训练样本...")
        
        images = []
        depth_maps = []
        
        for i in range(num_samples):
            # 生成随机几何形状
            image = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
            depth_map = np.zeros((self.input_height, self.input_width), dtype=np.float32)
            
            # 随机背景颜色
            bg_color = np.random.randint(50, 200, 3)
            image[:] = bg_color
            
            # 添加多个几何形状
            num_shapes = np.random.randint(3, 8)
            for _ in range(num_shapes):
                # 随机深度值
                depth = np.random.uniform(0.1, 10.0)
                
                # 随机颜色
                color = np.random.randint(0, 255, 3)
                
                # 随机位置和大小
                center_x = np.random.randint(50, self.input_width - 50)
                center_y = np.random.randint(50, self.input_height - 50)
                radius = np.random.randint(20, 80)
                
                # 绘制圆形
                cv2.circle(image, (center_x, center_y), radius, color.tolist(), -1)
                cv2.circle(depth_map, (center_x, center_y), radius, depth, -1)
                
                # 添加一些矩形
                if np.random.random() > 0.5:
                    x1 = center_x - radius//2
                    y1 = center_y - radius//2
                    x2 = center_x + radius//2
                    y2 = center_y + radius//2
                    
                    # 确保矩形在图像范围内
                    x1 = max(0, min(x1, self.input_width))
                    y1 = max(0, min(y1, self.input_height))
                    x2 = max(0, min(x2, self.input_width))
                    y2 = max(0, min(y2, self.input_height))
                    
                    if x2 > x1 and y2 > y1:
                        rect_color = np.random.randint(0, 255, 3)
                        cv2.rectangle(image, (x1, y1), (x2, y2), rect_color.tolist(), -1)
                        cv2.rectangle(depth_map, (x1, y1), (x2, y2), depth * 0.8, -1)
            
            # 归一化深度图到[0,1]
            depth_map = depth_map / 10.0
            depth_map = np.clip(depth_map, 0, 1)
            
            # 添加噪声
            noise = np.random.normal(0, 0.02, image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            
            images.append(image)
            depth_maps.append(depth_map)
            
            if (i + 1) % 100 == 0:
                print(f"已生成 {i + 1}/{num_samples} 个样本")
        
        return np.array(images), np.array(depth_maps)
    
    def create_data_generator(self, images, depth_maps, batch_size=8):
        """
        创建数据生成器
        
        Args:
            images: 图像数组
            depth_maps: 深度图数组
            batch_size: 批次大小
            
        Returns:
            数据生成器
        """
        def generator():
            while True:
                # 随机选择批次
                indices = np.random.choice(len(images), batch_size, replace=False)
                
                batch_images = []
                batch_depths = []
                
                for idx in indices:
                    # 归一化图像到[0,1]
                    img = images[idx].astype(np.float32) / 255.0
                    depth = depth_maps[idx]
                    
                    batch_images.append(img)
                    batch_depths.append(np.expand_dims(depth, axis=-1))
                
                yield np.array(batch_images), np.array(batch_depths)
        
        return generator
    
    def train(self, num_epochs=50, batch_size=8, num_samples=1000):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            batch_size: 批次大小
            num_samples: 训练样本数量
        """
        print("开始训练深度估计模型...")
        
        # 生成训练数据
        images, depth_maps = self.generate_synthetic_data(num_samples)
        
        # 分割训练和验证数据
        split_idx = int(0.8 * len(images))
        train_images = images[:split_idx]
        train_depths = depth_maps[:split_idx]
        val_images = images[split_idx:]
        val_depths = depth_maps[split_idx:]
        
        print(f"训练样本: {len(train_images)}, 验证样本: {len(val_images)}")
        
        # 创建数据生成器
        train_gen = self.create_data_generator(train_images, train_depths, batch_size)
        val_gen = self.create_data_generator(val_images, val_depths, batch_size)
        
        # 计算每轮的步数
        steps_per_epoch = len(train_images) // batch_size
        validation_steps = len(val_images) // batch_size
        
        # 回调函数
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
        ]
        
        # 开始训练
        history = self.depth_estimator.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=num_epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        print("训练完成!")
        return history
    
    def save_model(self, filepath):
        """保存训练好的模型"""
        self.depth_estimator.save_model(filepath)
    
    def evaluate_model(self, test_images, test_depths):
        """
        评估模型性能
        
        Args:
            test_images: 测试图像
            test_depths: 测试深度图
            
        Returns:
            评估结果
        """
        print("评估模型性能...")
        
        # 预处理测试数据
        test_images_norm = test_images.astype(np.float32) / 255.0
        test_depths_expanded = np.expand_dims(test_depths, axis=-1)
        
        # 评估
        results = self.depth_estimator.model.evaluate(
            test_images_norm, test_depths_expanded, verbose=1
        )
        
        print(f"测试损失: {results[0]:.4f}")
        print(f"测试MAE: {results[1]:.4f}")
        print(f"测试MSE: {results[2]:.4f}")
        
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='深度估计模型训练')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--samples', type=int, default=1000, help='训练样本数量')
    parser.add_argument('--output', type=str, default='trained_depth_model', 
                       help='模型保存路径')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = DepthModelTrainer()
    
    # 开始训练
    history = trainer.train(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.samples
    )
    
    # 保存模型
    trainer.save_model(args.output)
    
    print(f"训练完成，模型已保存到: {args.output}")


if __name__ == "__main__":
    main()
