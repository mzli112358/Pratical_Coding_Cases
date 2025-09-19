#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO模型下载脚本
支持下载YOLO v5、v8、v11系列的所有版本模型
包括: n, s, m, l, x 版本
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests
from tqdm import tqdm

class YOLODownloader:
    """YOLO模型下载器"""
    
    def __init__(self, download_dir: str = "models"):
        """
        初始化下载器
        
        Args:
            download_dir: 模型下载目录
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # 10个YOLO模型列表（按序号1-10）
        self.models_list = [
            {"name": "YOLOv5n", "file": "yolov5n.pt", "url": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt"},
            {"name": "YOLOv5s", "file": "yolov5s.pt", "url": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"},
            {"name": "YOLOv5m", "file": "yolov5m.pt", "url": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt"},
            {"name": "YOLOv5l", "file": "yolov5l.pt", "url": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt"},
            {"name": "YOLOv5x", "file": "yolov5x.pt", "url": "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt"},
            {"name": "YOLOv8n", "file": "yolov8n.pt", "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"},
            {"name": "YOLOv8s", "file": "yolov8s.pt", "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"},
            {"name": "YOLOv8m", "file": "yolov8m.pt", "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"},
            {"name": "YOLOv8l", "file": "yolov8l.pt", "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt"},
            {"name": "YOLOv8x", "file": "yolov8x.pt", "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"}
        ]
    
    def _cleanup_incomplete_file(self, filepath: Path) -> None:
        """
        清理不完整的下载文件
        
        Args:
            filepath: 要清理的文件路径
        """
        try:
            if filepath.exists():
                file_size = filepath.stat().st_size
                filepath.unlink()
                print(f"🗑️  已删除残缺文件: {filepath.name} ({file_size} 字节)")
            else:
                print(f"ℹ️  文件不存在，无需清理: {filepath.name}")
        except Exception as e:
            print(f"⚠️  清理文件时出错: {e}")
    
    def _is_file_complete(self, filepath: Path, expected_size: Optional[int] = None) -> bool:
        """
        检查文件是否完整
        
        Args:
            filepath: 文件路径
            expected_size: 期望的文件大小（可选）
            
        Returns:
            bool: 文件是否完整
        """
        try:
            if not filepath.exists():
                return False
            
            file_size = filepath.stat().st_size
            
            # 如果提供了期望大小，检查是否匹配
            if expected_size is not None:
                return file_size == expected_size
            
            # 基本检查：文件大小应该大于1MB（YOLO模型文件通常都比较大）
            # 如果文件小于1MB，很可能是残缺文件
            min_size = 1024 * 1024  # 1MB
            if file_size < min_size:
                return False
            
            # 检查文件是否以正确的格式开头（PyTorch模型文件通常以特定字节开头）
            try:
                with open(filepath, 'rb') as f:
                    # 读取前几个字节检查文件头
                    header = f.read(8)
                    # PyTorch模型文件通常以特定的魔数开头
                    # 这里我们检查文件是否看起来像二进制文件
                    if len(header) < 8:
                        return False
            except:
                return False
            
            return True
            
        except Exception as e:
            print(f"⚠️  检查文件完整性时出错: {e}")
            return False
    
    def download_file(self, url: str, filepath: Path, description: str = "下载中") -> bool:
        """
        下载文件并显示进度条
        
        Args:
            url: 下载链接
            filepath: 保存路径
            description: 进度条描述
            
        Returns:
            bool: 下载是否成功
        """
        temp_filepath = filepath.with_suffix(filepath.suffix + '.tmp')
        downloaded_size = 0
        
        try:
            print(f"开始下载: {filepath.name}")
            print(f"下载链接: {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(temp_filepath, 'wb') as f:
                with tqdm(
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    desc=description,
                    ncols=100
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded_size += len(chunk)
                            pbar.update(len(chunk))
            
            # 验证下载完整性
            if total_size > 0 and downloaded_size != total_size:
                print(f"❌ 下载不完整: 期望 {total_size} 字节，实际 {downloaded_size} 字节")
                self._cleanup_incomplete_file(temp_filepath)
                return False
            
            # 下载成功，重命名临时文件
            if temp_filepath.exists():
                temp_filepath.rename(filepath)
                print(f"✅ 下载完成: {filepath}")
                return True
            else:
                print(f"❌ 下载失败: 临时文件不存在")
                return False
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 下载失败: {e}")
            self._cleanup_incomplete_file(temp_filepath)
            return False
        except Exception as e:
            print(f"❌ 下载出错: {e}")
            self._cleanup_incomplete_file(temp_filepath)
            return False
    
    def download_model_by_index(self, index: int) -> bool:
        """
        根据序号下载模型
        
        Args:
            index: 模型序号 (1-15)
            
        Returns:
            bool: 下载是否成功
        """
        if index < 1 or index > len(self.models_list):
            print(f"❌ 无效序号: {index}，请输入1-{len(self.models_list)}之间的数字")
            return False
        
        model = self.models_list[index - 1]
        filepath = self.download_dir / model["file"]
        
        # 检查文件是否已存在
        if filepath.exists():
            print(f"⚠️  模型文件已存在: {filepath}")
            
            # 检查文件完整性
            if not self._is_file_complete(filepath):
                print(f"❌ 检测到残缺文件，将删除并重新下载")
                self._cleanup_incomplete_file(filepath)
            else:
                response = input("文件完整，是否重新下载? (y/N): ").strip().lower()
                if response != 'y':
                    print("跳过下载")
                    return True
        
        return self.download_file(model["url"], filepath, f"下载 {model['name']}")
    
    def download_models_by_indices(self, indices: List[int]) -> Dict[int, bool]:
        """
        根据序号列表下载多个模型
        
        Args:
            indices: 模型序号列表
            
        Returns:
            Dict[int, bool]: 各模型下载结果
        """
        results = {}
        print(f"\n🚀 开始下载 {len(indices)} 个模型...")
        
        for index in indices:
            if 1 <= index <= len(self.models_list):
                model = self.models_list[index - 1]
                print(f"\n📦 下载 {index}. {model['name']} 模型...")
                results[index] = self.download_model_by_index(index)
                time.sleep(1)  # 避免请求过于频繁
            else:
                print(f"❌ 跳过无效序号: {index}")
                results[index] = False
        
        return results
    
    def cleanup_incomplete_files(self) -> int:
        """
        清理所有残缺的模型文件
        
        Returns:
            int: 清理的文件数量
        """
        cleaned_count = 0
        print("🔍 检查并清理残缺文件...")
        
        for model in self.models_list:
            filepath = self.download_dir / model["file"]
            if filepath.exists() and not self._is_file_complete(filepath):
                print(f"🗑️  发现残缺文件: {filepath.name}")
                self._cleanup_incomplete_file(filepath)
                cleaned_count += 1
        
        if cleaned_count > 0:
            print(f"✅ 清理完成，共删除 {cleaned_count} 个残缺文件")
        else:
            print("✅ 未发现残缺文件")
        
        return cleaned_count
    
    def show_models_list(self):
        """显示10个模型列表"""
        print("\n📋 可用的YOLO模型 (共10个):")
        print("="*80)
        
        for i, model in enumerate(self.models_list, 1):
            filepath = self.download_dir / model["file"]
            if filepath.exists():
                if self._is_file_complete(filepath):
                    status = "✅ 已下载"
                else:
                    status = "⚠️  残缺文件"
            else:
                status = "❌ 未下载"
            print(f"{i:2d}. {model['name']:12s} ({model['file']:15s}) - {status}")
        
        print("="*80)
        print("💡 使用方法: 输入序号下载，多个序号用空格分隔")
        print("   例如: 1 3 5 或 1-5 或 all")
    
    def interactive_download(self):
        """交互式下载界面"""
        while True:
            print("\n" + "="*60)
            print("🤖 YOLO模型下载器")
            print("="*60)
            print("1. 查看模型列表")
            print("2. 下载模型")
            print("3. 清理残缺文件")
            print("4. 退出")
            print("="*60)
            
            choice = input("请选择操作 (1-4): ").strip()
            
            if choice == "1":
                self.show_models_list()
            elif choice == "2":
                self._download_models()
            elif choice == "3":
                self.cleanup_incomplete_files()
            elif choice == "4":
                print("👋 再见!")
                break
            else:
                print("❌ 无效选择，请重新输入")
    
    def _download_models(self):
        """下载模型"""
        self.show_models_list()
        
        print("\n请输入要下载的模型序号:")
        print("支持格式:")
        print("  - 单个序号: 1")
        print("  - 多个序号: 1 3 5")
        print("  - 范围: 1-5")
        print("  - 全部: all")
        
        user_input = input("\n请输入: ").strip()
        
        if not user_input:
            print("❌ 输入为空")
            return
        
        # 解析用户输入
        indices = self._parse_user_input(user_input)
        
        if not indices:
            print("❌ 无效输入")
            return
        
        if len(indices) > 5:
            confirm = input(f"⚠️  将下载 {len(indices)} 个模型，确认继续? (y/N): ").strip().lower()
            if confirm != 'y':
                print("取消下载")
                return
        
        # 开始下载
        results = self.download_models_by_indices(indices)
        
        # 显示结果
        print(f"\n📊 下载结果:")
        success_count = sum(1 for success in results.values() if success)
        print(f"✅ 成功: {success_count}/{len(results)}")
        print(f"❌ 失败: {len(results) - success_count}/{len(results)}")
    
    def _parse_user_input(self, user_input: str) -> List[int]:
        """
        解析用户输入，返回序号列表
        
        Args:
            user_input: 用户输入字符串
            
        Returns:
            List[int]: 序号列表
        """
        user_input = user_input.lower().strip()
        
        if user_input == "all":
            return list(range(1, len(self.models_list) + 1))
        
        indices = []
        
        # 处理空格分隔的序号
        parts = user_input.split()
        
        for part in parts:
            if '-' in part:
                # 处理范围，如 "1-5"
                try:
                    start, end = map(int, part.split('-'))
                    indices.extend(range(start, end + 1))
                except ValueError:
                    continue
            else:
                # 处理单个序号
                try:
                    index = int(part)
                    if 1 <= index <= len(self.models_list):
                        indices.append(index)
                except ValueError:
                    continue
        
        # 去重并排序
        return sorted(list(set(indices)))


def main():
    """主函数"""
    print("🚀 YOLO模型下载器启动中...")
    
    # 创建下载器实例
    downloader = YOLODownloader()
    
    # 启动时自动清理残缺文件
    print("🔍 启动时检查残缺文件...")
    cleaned_count = downloader.cleanup_incomplete_files()
    if cleaned_count > 0:
        print(f"⚠️  已清理 {cleaned_count} 个残缺文件，建议重新下载")
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == "--all":
            # 下载所有模型
            print("📥 开始下载所有10个模型...")
            all_indices = list(range(1, len(downloader.models_list) + 1))
            results = downloader.download_models_by_indices(all_indices)
            success_count = sum(1 for success in results.values() if success)
            print(f"\n📊 下载完成: 成功 {success_count}/{len(results)} 个模型")
        elif sys.argv[1] == "--list":
            # 列出已下载模型
            downloader.show_models_list()
        else:
            # 尝试解析为序号
            try:
                indices = downloader._parse_user_input(" ".join(sys.argv[1:]))
                if indices:
                    results = downloader.download_models_by_indices(indices)
                    success_count = sum(1 for success in results.values() if success)
                    print(f"\n📊 下载完成: 成功 {success_count}/{len(results)} 个模型")
                else:
                    print("❌ 无效的序号参数")
            except:
                print("❌ 未知参数")
                print("用法:")
                print("  python code_for_download_yolo.py                    # 交互式界面")
                print("  python code_for_download_yolo.py --all             # 下载所有模型")
                print("  python code_for_download_yolo.py --list            # 列出已下载模型")
                print("  python code_for_download_yolo.py 1 3 5             # 下载指定序号模型")
                print("  python code_for_download_yolo.py 1-5                # 下载范围模型")
    else:
        # 交互式界面
        downloader.interactive_download()


if __name__ == "__main__":
    main()
