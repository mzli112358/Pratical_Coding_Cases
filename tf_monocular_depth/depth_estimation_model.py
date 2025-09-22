#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于TensorFlow的单目深度估计模型
使用编码器-解码器架构进行深度预测
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import cv2
import os
from pretrained_models import SimpleDepthEstimator


class MonocularDepthEstimator:
    """单目深度估计器"""
    
    def __init__(self, input_height=480, input_width=640, use_pretrained=True):
        """
        初始化深度估计器
        
        Args:
            input_height: 输入图像高度
            input_width: 输入图像宽度
            use_pretrained: 是否使用预训练模型
        """
        self.input_height = input_height
        self.input_width = input_width
        self.use_pretrained = use_pretrained
        
        if use_pretrained:
            # 使用预训练模型
            pretrained_model_path = os.path.join("pretrained_models", "simple_vgg_depth.h5")
            self.pretrained_estimator = SimpleDepthEstimator(pretrained_model_path)
            print("使用预训练深度估计模型")
        else:
            # 使用自定义模型
            self.model = self._build_model()
            print("使用自定义深度估计模型")
        
    def _build_model(self):
        """构建编码器-解码器深度估计模型"""
        
        # 输入层
        inputs = layers.Input(shape=(self.input_height, self.input_width, 3))
        
        # 编码器部分 - 特征提取
        # 第一层
        x = layers.Conv2D(64, 7, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        # 第二层
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        # 第三层
        x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        # 第四层
        x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        # 瓶颈层
        x = layers.Conv2D(512, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # 解码器部分 - 上采样和特征融合
        # 第一层上采样
        x = layers.Conv2DTranspose(512, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # 第二层上采样
        x = layers.Conv2DTranspose(256, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # 第三层上采样
        x = layers.Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # 第四层上采样
        x = layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # 输出层 - 单通道深度图
        outputs = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)
        
        # 创建模型
        model = Model(inputs, outputs, name='MonocularDepthEstimator')
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """编译模型"""
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # 自定义损失函数 - 结合L1和SSIM损失
        def depth_loss(y_true, y_pred):
            # L1损失
            l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
            
            # SSIM损失
            ssim_loss = 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)
            
            return l1_loss + 0.1 * ssim_loss
        
        self.model.compile(
            optimizer=optimizer,
            loss=depth_loss,
            metrics=['mae', 'mse']
        )
        
        print("模型编译完成")
        
    def preprocess_image(self, image):
        """
        预处理输入图像
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            预处理后的图像
        """
        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整尺寸
        image_resized = cv2.resize(image_rgb, (self.input_width, self.input_height))
        
        # 归一化到[0,1]
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # 添加批次维度
        image_batch = np.expand_dims(image_normalized, axis=0)
        
        return image_batch
    
    def predict_depth(self, image):
        """
        预测深度图
        
        Args:
            image: 输入图像
            
        Returns:
            深度图 (numpy array)
        """
        if self.use_pretrained:
            # 使用预训练模型预测
            return self.pretrained_estimator.predict_depth(image)
        else:
            # 使用自定义模型预测
            # 预处理图像
            processed_image = self.preprocess_image(image)
            
            # 预测深度
            depth_pred = self.model.predict(processed_image, verbose=0)
            
            # 移除批次维度并调整尺寸
            depth_map = depth_pred[0, :, :, 0]
            
            return depth_map
    
    def postprocess_depth(self, depth_map, min_depth=0.1, max_depth=10.0):
        """
        后处理深度图
        
        Args:
            depth_map: 原始深度图
            min_depth: 最小深度值
            max_depth: 最大深度值
            
        Returns:
            后处理的深度图
        """
        # 将深度值映射到实际深度范围
        depth_scaled = depth_map * (max_depth - min_depth) + min_depth
        
        # 应用高斯滤波平滑
        depth_smoothed = cv2.GaussianBlur(depth_scaled, (5, 5), 0)
        
        return depth_smoothed
    
    def visualize_depth(self, depth_map, colormap=cv2.COLORMAP_JET):
        """
        将深度图转换为彩色可视化
        
        Args:
            depth_map: 深度图
            colormap: OpenCV颜色映射
            
        Returns:
            彩色深度图
        """
        if self.use_pretrained:
            # 使用预训练模型的可视化方法
            return self.pretrained_estimator.visualize_depth(depth_map, colormap)
        else:
            # 使用自定义模型的可视化方法
            # 归一化深度图到[0,255]
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_uint8 = depth_normalized.astype(np.uint8)
            
            # 应用颜色映射
            depth_colored = cv2.applyColorMap(depth_uint8, colormap)
            
            return depth_colored
    
    def save_model(self, filepath):
        """保存模型"""
        self.model.save(filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """加载预训练模型"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"模型已从 {filepath} 加载")
    
    def summary(self):
        """显示模型结构"""
        self.model.summary()
