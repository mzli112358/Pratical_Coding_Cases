#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设置预训练深度估计模型
自动创建和配置预训练模型
"""

import os
import sys
from pretrained_models import PretrainedDepthModels, SimpleDepthEstimator


def setup_pretrained_models():
    """设置预训练模型"""
    print("=== 设置预训练深度估计模型 ===")
    
    # 创建模型管理器
    model_manager = PretrainedDepthModels()
    
    # 创建简单深度估计器（这会自动创建预训练模型）
    print("\n正在创建预训练深度估计模型...")
    try:
        estimator = SimpleDepthEstimator()
        print("✅ 预训练模型创建成功!")
        
        # 测试模型
        print("\n测试预训练模型...")
        import numpy as np
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth_map = estimator.predict_depth(test_image)
        print(f"✅ 模型测试成功! 深度图形状: {depth_map.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ 预训练模型创建失败: {e}")
        return False


def main():
    """主函数"""
    print("正在设置预训练深度估计模型...")
    
    success = setup_pretrained_models()
    
    if success:
        print("\n🎉 预训练模型设置完成!")
        print("\n现在您可以使用以下命令运行深度估计:")
        print("python run_demo.py")
        print("\n或者直接使用预训练模型:")
        print("python depth_demo.py --mode camera --pretrained")
    else:
        print("\n❌ 预训练模型设置失败")
        print("您仍可以使用自定义模型:")
        print("python depth_demo.py --mode camera --no-pretrained")


if __name__ == "__main__":
    main()
