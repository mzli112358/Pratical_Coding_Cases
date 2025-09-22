#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预训练深度估计模型支持
包括MiDaS、DPT等知名模型的下载和使用
"""

import os
import requests
import zipfile
import tensorflow as tf
import numpy as np
import cv2
from urllib.parse import urlparse


class PretrainedDepthModels:
    """预训练深度估计模型管理器"""
    
    def __init__(self, models_dir="pretrained_models"):
        """
        初始化预训练模型管理器
        
        Args:
            models_dir: 模型保存目录
        """
        self.models_dir = models_dir
        self.create_models_dir()
        
        # 预训练模型配置
        self.model_configs = {
            "midas_small": {
                "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_small_512.pt",
                "description": "MiDaS DPT-BEiT Small (512x512)",
                "input_size": (512, 512),
                "framework": "pytorch"
            },
            "midas_large": {
                "url": "https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_large_384.pt",
                "description": "MiDaS DPT Large (384x384)",
                "input_size": (384, 384),
                "framework": "pytorch"
            }
        }
    
    def create_models_dir(self):
        """创建模型目录"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            print(f"创建模型目录: {self.models_dir}")
    
    def download_file(self, url, filename):
        """
        下载文件
        
        Args:
            url: 下载链接
            filename: 保存文件名
            
        Returns:
            下载是否成功
        """
        try:
            print(f"正在下载: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            filepath = os.path.join(self.models_dir, filename)
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"下载完成: {filepath}")
            return True
            
        except Exception as e:
            print(f"下载失败: {e}")
            return False
    
    def download_model(self, model_name):
        """
        下载指定的预训练模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            下载是否成功
        """
        if model_name not in self.model_configs:
            print(f"未知模型: {model_name}")
            print(f"可用模型: {list(self.model_configs.keys())}")
            return False
        
        config = self.model_configs[model_name]
        url = config["url"]
        
        # 从URL提取文件名
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        # 检查文件是否已存在
        filepath = os.path.join(self.models_dir, filename)
        if os.path.exists(filepath):
            print(f"模型已存在: {filepath}")
            return True
        
        return self.download_file(url, filename)
    
    def list_available_models(self):
        """列出可用的预训练模型"""
        print("可用的预训练深度估计模型:")
        print("-" * 50)
        for name, config in self.model_configs.items():
            print(f"名称: {name}")
            print(f"描述: {config['description']}")
            print(f"输入尺寸: {config['input_size']}")
            print(f"框架: {config['framework']}")
            print(f"URL: {config['url']}")
            print("-" * 50)
    
    def create_simple_depth_model(self):
        """
        创建一个简单的预训练风格深度模型
        使用预训练的特征提取器
        """
        print("创建简单深度估计模型...")
        
        # 使用预训练的VGG16作为编码器
        base_model = tf.keras.applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # 冻结预训练层
        for layer in base_model.layers:
            layer.trainable = False
        
        # 添加自定义解码器
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(224 * 224, activation='sigmoid')(x)
        x = layers.Reshape((224, 224, 1))(x)
        
        model = tf.keras.Model(base_model.input, x)
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        # 保存模型
        model_path = os.path.join(self.models_dir, "simple_vgg_depth.h5")
        model.save(model_path)
        print(f"简单深度模型已保存到: {model_path}")
        
        return model


class SimpleDepthEstimator:
    """简化的深度估计器，使用预训练特征"""
    
    def __init__(self, model_path=None):
        """
        初始化深度估计器
        
        Args:
            model_path: 预训练模型路径
        """
        self.input_size = (224, 224)
        
        if model_path and os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"已加载预训练模型: {model_path}")
        else:
            self.model = self._create_model()
            print("使用默认模型")
    
    def _create_model(self):
        """创建深度估计模型"""
        # 使用预训练的MobileNetV2作为特征提取器
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.input_size, 3)
        )
        
        # 冻结预训练层
        for layer in base_model.layers:
            layer.trainable = False
        
        # 添加解码器
        x = base_model.output
        
        # 上采样到原始尺寸
        x = tf.keras.layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2DTranspose(16, 3, strides=2, padding='same', activation='relu')(x)
        
        # 输出层
        outputs = tf.keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
        
        model = tf.keras.Model(base_model.input, outputs)
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def preprocess_image(self, image):
        """预处理图像"""
        # 转换为RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # 调整尺寸
        image_resized = cv2.resize(image_rgb, self.input_size)
        
        # 归一化
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # 添加批次维度
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch
    
    def predict_depth(self, image):
        """预测深度"""
        # 预处理
        processed_image = self.preprocess_image(image)
        
        # 预测
        depth_pred = self.model.predict(processed_image, verbose=0)
        
        # 后处理
        depth_map = depth_pred[0, :, :, 0]
        
        # 调整到原始图像尺寸
        original_height, original_width = image.shape[:2]
        depth_resized = cv2.resize(depth_map, (original_width, original_height))
        
        return depth_resized
    
    def visualize_depth(self, depth_map, colormap=cv2.COLORMAP_JET):
        """可视化深度图"""
        # 归一化到[0,255]
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_uint8 = depth_normalized.astype(np.uint8)
        
        # 应用颜色映射
        depth_colored = cv2.applyColorMap(depth_uint8, colormap)
        
        return depth_colored


def main():
    """主函数 - 演示预训练模型的使用"""
    print("=== 预训练深度估计模型管理器 ===")
    
    # 创建模型管理器
    model_manager = PretrainedDepthModels()
    
    # 列出可用模型
    model_manager.list_available_models()
    
    # 创建简单深度估计器
    print("\n创建简单深度估计器...")
    estimator = SimpleDepthEstimator()
    
    # 测试图像
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # 预测深度
    depth_map = estimator.predict_depth(test_image)
    depth_colored = estimator.visualize_depth(depth_map)
    
    print(f"深度图形状: {depth_map.shape}")
    print("深度估计完成!")


if __name__ == "__main__":
    main()
