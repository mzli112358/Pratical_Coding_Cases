#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卡通形象类
支持与人体骨架绑定并同步动作
"""

import cv2
import numpy as np
import math

class CartoonCharacter:
    def __init__(self, character_type="stick_figure"):
        """
        初始化卡通形象
        
        Args:
            character_type (str): 卡通形象类型 ("stick_figure", "simple_robot", "cute_character")
        """
        self.character_type = character_type
        self.current_pose = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        
        # 颜色定义
        self.colors = {
            'head': (255, 200, 200),      # 浅粉色
            'body': (200, 200, 255),      # 浅蓝色
            'arm': (255, 255, 200),       # 浅黄色
            'leg': (200, 255, 200),       # 浅绿色
            'joint': (100, 100, 100),     # 灰色
            'outline': (0, 0, 0)          # 黑色
        }
        
        # 身体部位尺寸
        self.body_parts = {
            'head_radius': 20,
            'body_width': 30,
            'body_height': 60,
            'arm_length': 40,
            'leg_length': 50,
            'joint_radius': 8
        }
    
    def update_pose(self, keypoints):
        """
        更新卡通形象的姿态
        
        Args:
            keypoints: 人体关键点数据 (17个关键点)
        """
        if keypoints is None or len(keypoints) < 17:
            return
        
        self.current_pose = keypoints
        self._calculate_scale_and_offset()
    
    def _calculate_scale_and_offset(self):
        """计算缩放因子和偏移量"""
        if self.current_pose is None:
            return
        
        # 找到有效的关键点
        valid_points = []
        for point in self.current_pose:
            if point[2] > 0.5:  # 置信度阈值
                valid_points.append([point[0], point[1]])
        
        if len(valid_points) < 2:
            return
        
        valid_points = np.array(valid_points)
        
        # 计算边界框
        min_x, min_y = np.min(valid_points, axis=0)
        max_x, max_y = np.max(valid_points, axis=0)
        
        # 计算缩放因子（基于身体高度）
        body_height = max_y - min_y
        if body_height > 0:
            self.scale_factor = 200 / body_height  # 目标高度200像素
        
        # 计算偏移量（居中显示）
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        self.offset_x = center_x
        self.offset_y = center_y
    
    def _get_keypoint(self, index):
        """获取指定关键点坐标"""
        if self.current_pose is None or index >= len(self.current_pose):
            return None
        
        point = self.current_pose[index]
        if point[2] < 0.5:  # 置信度太低
            return None
        
        # 应用缩放和偏移
        x = (point[0] - self.offset_x) * self.scale_factor + self.offset_x
        y = (point[1] - self.offset_y) * self.scale_factor + self.offset_y
        
        return (int(x), int(y))
    
    def _draw_circle(self, frame, center, radius, color, thickness=-1):
        """绘制圆形"""
        if center is not None:
            cv2.circle(frame, center, radius, color, thickness)
            cv2.circle(frame, center, radius, self.colors['outline'], 2)
    
    def _draw_line(self, frame, start, end, color, thickness=3):
        """绘制线条"""
        if start is not None and end is not None:
            cv2.line(frame, start, end, color, thickness)
            cv2.line(frame, start, end, self.colors['outline'], thickness + 2)
    
    def _draw_ellipse(self, frame, center, axes, angle, color, thickness=-1):
        """绘制椭圆"""
        if center is not None:
            cv2.ellipse(frame, center, axes, angle, 0, 360, color, thickness)
            cv2.ellipse(frame, center, axes, angle, 0, 360, self.colors['outline'], 2)
    
    def draw_stick_figure(self, frame):
        """绘制火柴人"""
        # 获取关键点
        nose = self._get_keypoint(0)
        left_eye = self._get_keypoint(1)
        right_eye = self._get_keypoint(2)
        left_ear = self._get_keypoint(3)
        right_ear = self._get_keypoint(4)
        left_shoulder = self._get_keypoint(5)
        right_shoulder = self._get_keypoint(6)
        left_elbow = self._get_keypoint(7)
        right_elbow = self._get_keypoint(8)
        left_wrist = self._get_keypoint(9)
        right_wrist = self._get_keypoint(10)
        left_hip = self._get_keypoint(11)
        right_hip = self._get_keypoint(12)
        left_knee = self._get_keypoint(13)
        right_knee = self._get_keypoint(14)
        left_ankle = self._get_keypoint(15)
        right_ankle = self._get_keypoint(16)
        
        # 绘制头部
        if nose is not None:
            head_radius = int(self.body_parts['head_radius'] * self.scale_factor)
            self._draw_circle(frame, nose, head_radius, self.colors['head'])
            
            # 绘制眼睛
            if left_eye is not None:
                eye_radius = int(3 * self.scale_factor)
                self._draw_circle(frame, left_eye, eye_radius, self.colors['outline'])
            if right_eye is not None:
                eye_radius = int(3 * self.scale_factor)
                self._draw_circle(frame, right_eye, eye_radius, self.colors['outline'])
        
        # 绘制身体
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) // 2,
                             (left_shoulder[1] + right_shoulder[1]) // 2)
        else:
            shoulder_center = left_shoulder or right_shoulder
        
        if left_hip is not None and right_hip is not None:
            hip_center = ((left_hip[0] + right_hip[0]) // 2,
                         (left_hip[1] + right_hip[1]) // 2)
        else:
            hip_center = left_hip or right_hip
        
        if shoulder_center is not None and hip_center is not None:
            # 绘制躯干
            self._draw_line(frame, shoulder_center, hip_center, self.colors['body'])
        
        # 绘制手臂
        if left_shoulder is not None and left_elbow is not None:
            self._draw_line(frame, left_shoulder, left_elbow, self.colors['arm'])
        if left_elbow is not None and left_wrist is not None:
            self._draw_line(frame, left_elbow, left_wrist, self.colors['arm'])
        
        if right_shoulder is not None and right_elbow is not None:
            self._draw_line(frame, right_shoulder, right_elbow, self.colors['arm'])
        if right_elbow is not None and right_wrist is not None:
            self._draw_line(frame, right_elbow, right_wrist, self.colors['arm'])
        
        # 绘制腿部
        if left_hip is not None and left_knee is not None:
            self._draw_line(frame, left_hip, left_knee, self.colors['leg'])
        if left_knee is not None and left_ankle is not None:
            self._draw_line(frame, left_knee, left_ankle, self.colors['leg'])
        
        if right_hip is not None and right_knee is not None:
            self._draw_line(frame, right_hip, right_knee, self.colors['leg'])
        if right_knee is not None and right_ankle is not None:
            self._draw_line(frame, right_knee, right_ankle, self.colors['leg'])
        
        # 绘制关节
        joints = [left_shoulder, right_shoulder, left_elbow, right_elbow,
                 left_wrist, right_wrist, left_hip, right_hip,
                 left_knee, right_knee, left_ankle, right_ankle]
        
        for joint in joints:
            if joint is not None:
                joint_radius = int(self.body_parts['joint_radius'] * self.scale_factor)
                self._draw_circle(frame, joint, joint_radius, self.colors['joint'])
    
    def draw_simple_robot(self, frame):
        """绘制简单机器人"""
        # 获取关键点
        nose = self._get_keypoint(0)
        left_shoulder = self._get_keypoint(5)
        right_shoulder = self._get_keypoint(6)
        left_elbow = self._get_keypoint(7)
        right_elbow = self._get_keypoint(8)
        left_wrist = self._get_keypoint(9)
        right_wrist = self._get_keypoint(10)
        left_hip = self._get_keypoint(11)
        right_hip = self._get_keypoint(12)
        left_knee = self._get_keypoint(13)
        right_knee = self._get_keypoint(14)
        left_ankle = self._get_keypoint(15)
        right_ankle = self._get_keypoint(16)
        
        # 绘制头部（方形）
        if nose is not None:
            head_size = int(self.body_parts['head_radius'] * self.scale_factor)
            top_left = (nose[0] - head_size, nose[1] - head_size)
            bottom_right = (nose[0] + head_size, nose[1] + head_size)
            cv2.rectangle(frame, top_left, bottom_right, self.colors['head'], -1)
            cv2.rectangle(frame, top_left, bottom_right, self.colors['outline'], 2)
            
            # 绘制眼睛
            eye_size = int(5 * self.scale_factor)
            left_eye_pos = (nose[0] - head_size//2, nose[1] - head_size//2)
            right_eye_pos = (nose[0] + head_size//2, nose[1] - head_size//2)
            cv2.circle(frame, left_eye_pos, eye_size, self.colors['outline'], -1)
            cv2.circle(frame, right_eye_pos, eye_size, self.colors['outline'], -1)
        
        # 绘制身体（矩形）
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) // 2,
                             (left_shoulder[1] + right_shoulder[1]) // 2)
        else:
            shoulder_center = left_shoulder or right_shoulder
        
        if left_hip is not None and right_hip is not None:
            hip_center = ((left_hip[0] + right_hip[0]) // 2,
                         (left_hip[1] + right_hip[1]) // 2)
        else:
            hip_center = left_hip or right_hip
        
        if shoulder_center is not None and hip_center is not None:
            body_width = int(self.body_parts['body_width'] * self.scale_factor)
            top_left = (shoulder_center[0] - body_width//2, shoulder_center[1])
            bottom_right = (hip_center[0] + body_width//2, hip_center[1])
            cv2.rectangle(frame, top_left, bottom_right, self.colors['body'], -1)
            cv2.rectangle(frame, top_left, bottom_right, self.colors['outline'], 2)
        
        # 绘制手臂（粗线条）
        if left_shoulder is not None and left_elbow is not None:
            self._draw_line(frame, left_shoulder, left_elbow, self.colors['arm'], 8)
        if left_elbow is not None and left_wrist is not None:
            self._draw_line(frame, left_elbow, left_wrist, self.colors['arm'], 6)
        
        if right_shoulder is not None and right_elbow is not None:
            self._draw_line(frame, right_shoulder, right_elbow, self.colors['arm'], 8)
        if right_elbow is not None and right_wrist is not None:
            self._draw_line(frame, right_elbow, right_wrist, self.colors['arm'], 6)
        
        # 绘制腿部（粗线条）
        if left_hip is not None and left_knee is not None:
            self._draw_line(frame, left_hip, left_knee, self.colors['leg'], 8)
        if left_knee is not None and left_ankle is not None:
            self._draw_line(frame, left_knee, left_ankle, self.colors['leg'], 6)
        
        if right_hip is not None and right_knee is not None:
            self._draw_line(frame, right_hip, right_knee, self.colors['leg'], 8)
        if right_knee is not None and right_ankle is not None:
            self._draw_line(frame, right_knee, right_ankle, self.colors['leg'], 6)
    
    def draw_cute_character(self, frame):
        """绘制可爱角色"""
        # 获取关键点
        nose = self._get_keypoint(0)
        left_eye = self._get_keypoint(1)
        right_eye = self._get_keypoint(2)
        left_shoulder = self._get_keypoint(5)
        right_shoulder = self._get_keypoint(6)
        left_elbow = self._get_keypoint(7)
        right_elbow = self._get_keypoint(8)
        left_wrist = self._get_keypoint(9)
        right_wrist = self._get_keypoint(10)
        left_hip = self._get_keypoint(11)
        right_hip = self._get_keypoint(12)
        left_knee = self._get_keypoint(13)
        right_knee = self._get_keypoint(14)
        left_ankle = self._get_keypoint(15)
        right_ankle = self._get_keypoint(16)
        
        # 绘制头部（圆形，更大）
        if nose is not None:
            head_radius = int(self.body_parts['head_radius'] * self.scale_factor * 1.5)
            self._draw_circle(frame, nose, head_radius, self.colors['head'])
            
            # 绘制眼睛（更大更可爱）
            if left_eye is not None:
                eye_radius = int(6 * self.scale_factor)
                self._draw_circle(frame, left_eye, eye_radius, self.colors['outline'])
                # 绘制眼珠
                pupil_radius = int(3 * self.scale_factor)
                self._draw_circle(frame, left_eye, pupil_radius, (0, 0, 0))
            
            if right_eye is not None:
                eye_radius = int(6 * self.scale_factor)
                self._draw_circle(frame, right_eye, eye_radius, self.colors['outline'])
                # 绘制眼珠
                pupil_radius = int(3 * self.scale_factor)
                self._draw_circle(frame, right_eye, pupil_radius, (0, 0, 0))
            
            # 绘制嘴巴
            mouth_center = (nose[0], nose[1] + head_radius//2)
            mouth_radius = int(8 * self.scale_factor)
            cv2.ellipse(frame, mouth_center, (mouth_radius, mouth_radius//2), 0, 0, 180, self.colors['outline'], 2)
        
        # 绘制身体（椭圆形）
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_center = ((left_shoulder[0] + right_shoulder[0]) // 2,
                             (left_shoulder[1] + right_shoulder[1]) // 2)
        else:
            shoulder_center = left_shoulder or right_shoulder
        
        if left_hip is not None and right_hip is not None:
            hip_center = ((left_hip[0] + right_hip[0]) // 2,
                         (left_hip[1] + right_hip[1]) // 2)
        else:
            hip_center = left_hip or right_hip
        
        if shoulder_center is not None and hip_center is not None:
            body_center = ((shoulder_center[0] + hip_center[0]) // 2,
                          (shoulder_center[1] + hip_center[1]) // 2)
            body_width = int(self.body_parts['body_width'] * self.scale_factor)
            body_height = int(abs(hip_center[1] - shoulder_center[1]))
            self._draw_ellipse(frame, body_center, (body_width//2, body_height//2), 0, self.colors['body'])
        
        # 绘制手臂（带关节的线条）
        if left_shoulder is not None and left_elbow is not None:
            self._draw_line(frame, left_shoulder, left_elbow, self.colors['arm'], 6)
        if left_elbow is not None and left_wrist is not None:
            self._draw_line(frame, left_elbow, left_wrist, self.colors['arm'], 4)
        
        if right_shoulder is not None and right_elbow is not None:
            self._draw_line(frame, right_shoulder, right_elbow, self.colors['arm'], 6)
        if right_elbow is not None and right_wrist is not None:
            self._draw_line(frame, right_elbow, right_wrist, self.colors['arm'], 4)
        
        # 绘制腿部（带关节的线条）
        if left_hip is not None and left_knee is not None:
            self._draw_line(frame, left_hip, left_knee, self.colors['leg'], 6)
        if left_knee is not None and left_ankle is not None:
            self._draw_line(frame, left_knee, left_ankle, self.colors['leg'], 4)
        
        if right_hip is not None and right_knee is not None:
            self._draw_line(frame, right_hip, right_knee, self.colors['leg'], 6)
        if right_knee is not None and right_ankle is not None:
            self._draw_line(frame, right_knee, right_ankle, self.colors['leg'], 4)
        
        # 绘制手和脚（小圆圈）
        if left_wrist is not None:
            hand_radius = int(5 * self.scale_factor)
            self._draw_circle(frame, left_wrist, hand_radius, self.colors['joint'])
        if right_wrist is not None:
            hand_radius = int(5 * self.scale_factor)
            self._draw_circle(frame, right_wrist, hand_radius, self.colors['joint'])
        
        if left_ankle is not None:
            foot_radius = int(6 * self.scale_factor)
            self._draw_circle(frame, left_ankle, foot_radius, self.colors['joint'])
        if right_ankle is not None:
            foot_radius = int(6 * self.scale_factor)
            self._draw_circle(frame, right_ankle, foot_radius, self.colors['joint'])
    
    def draw(self, frame):
        """
        绘制卡通形象
        
        Args:
            frame: 输入帧
            
        Returns:
            绘制了卡通形象的帧
        """
        if self.current_pose is None:
            return frame
        
        if self.character_type == "stick_figure":
            self.draw_stick_figure(frame)
        elif self.character_type == "simple_robot":
            self.draw_simple_robot(frame)
        elif self.character_type == "cute_character":
            self.draw_cute_character(frame)
        
        return frame
    
    def set_character_type(self, character_type):
        """设置卡通形象类型"""
        self.character_type = character_type
    
    def get_character_types(self):
        """获取可用的卡通形象类型"""
        return ["stick_figure", "simple_robot", "cute_character"]
