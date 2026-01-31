# -*- coding: utf-8 -*-
# 最终版本: 包含PRB利用率预测、训练/测试模式切换、模型保存/加载、性能记录

# 多分枝决策

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import copy
import math
import os
import time


class SatelliteOrbitModel:
    """LEO卫星轨道模型 (此类无需修改)"""

    def __init__(self, altitude=600, earth_radius=6371):
        self.h = altitude
        self.Ra = earth_radius
        self.lambda_param = (earth_radius + altitude) / earth_radius
        self.velocity = 7.5
        self.psi = 0

    def update_position(self, dt=0.02):
        delta_psi = self.velocity * dt / (self.Ra + self.h)
        self.psi += delta_psi
        return self.psi

    def get_distance_to_ground(self):
        phi = self.get_elevation_angle()
        # 修正公式
        val_inside_sqrt = self.Ra ** 2 * math.sin(phi) ** 2 + self.h ** 2 + 2 * self.Ra * self.h
        distance = math.sqrt(val_inside_sqrt) - self.Ra * math.sin(phi)
        return distance

    def get_elevation_angle(self):
        cos_psi = math.cos(self.psi)
        denominator = math.sqrt(max(1e-10, 1 - cos_psi ** 2))
        if abs(denominator) < 1e-10:
            phi = math.pi / 2 if cos_psi > 0 else 0
        else:
            numerator = cos_psi - 1 / self.lambda_param
            phi = math.atan(max(-1e10, min(1e10, numerator / denominator)))
        phi = max(0, min(math.pi / 2, abs(phi)))
        return phi


class ChannelModel:
    """信道模型 (此类无需修改)"""

    def __init__(self, frequency=17.5e9):
        self.frequency = frequency

    def free_space_loss(self, distance_km):
        safe_distance = np.maximum(distance_km, 1e-6)
        return 32.45 + 20 * np.log10(self.frequency / 1e9) + 20 * np.log10(safe_distance)

    def shadowed_rician_fading(self, size, ms=8, bs=0.129, omega_s=0.372):
        return np.random.exponential(1 / bs, size=size) * np.random.gamma(ms, omega_s / ms, size=size)

    def nakagami_fading(self, size, m=3, omega=1):
        return np.random.gamma(m, omega / m, size=size)

    def get_sat_to_ground_channel(self, distance_km, size, is_ntn=True):
        path_loss_db = self.free_space_loss(distance_km)
        path_loss_linear = 10 ** (-path_loss_db / 10)
        if is_ntn:
            fading = self.shadowed_rician_fading(size, 8, 0.129, 0.372)
        else:
            fading = self.shadowed_rician_fading(size, 5, 0.251, 0.279)
        return path_loss_linear * fading

    def get_terrestrial_channel(self, distance_km, path_loss_exp=3.5):
        safe_distance = np.maximum(distance_km, 1e-6)
        path_loss_linear = (1 / safe_distance) ** path_loss_exp
        fading = self.nakagami_fading(size=distance_km.shape)
        return path_loss_linear * fading

    # 确保这个方法存在于ChannelModel类中或已动态绑定
    def get_full_terrestrial_gains(self, env):
        num_gnbs, num_users, num_channels = env.num_gnbs, env.users_per_gnb, env.num_channels
        gains = np.zeros((num_gnbs, num_users, num_channels))
        for n in range(num_channels):
            gains[:, :, n] = env.channel_model.get_terrestrial_channel(env.distances_gnb_to_user)
        return None, None, gains, None

