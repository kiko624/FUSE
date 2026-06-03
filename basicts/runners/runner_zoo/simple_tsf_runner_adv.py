from typing import Dict, Optional
import torch
import torch.nn as nn
import logging
import os
import numpy as np
from datetime import datetime
from ..base_tsf_runner_adv import BaseTimeSeriesForecastingRunner_adv
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import math
class EMA(nn.Module):
    """
    指数移动平均模块，用于提取时间序列的趋势成分
    """
    def __init__(self, alpha):
        super(EMA, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        batch_size, t, num_nodes, channels = x.shape
        # 生成EMA权重
        powers = torch.flip(torch.arange(t, dtype=torch.float32, device=x.device), dims=(0,))
        weights = torch.pow((1 - self.alpha), powers)
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        
        # 重塑权重以匹配输入维度
        weights = weights.reshape(1, t, 1, 1)
        divisor = divisor.reshape(1, t, 1, 1)
        
        # 计算加权累积和
        x_weighted = x * weights
        x_cumsum = torch.cumsum(x_weighted, dim=1)
        
        # 归一化
        trend = x_cumsum / divisor
        return trend


class DECOMP(nn.Module):
    """
    序列分解模块，将时间序列分解为季节性和趋势成分
    """
    def __init__(self, alpha):
        super(DECOMP, self).__init__()
        self.ma = EMA(alpha)

    def forward(self, x):
        moving_average = self.ma(x)  # 趋势成分
        seasonal = x - moving_average  # 季节性成分
        return seasonal, moving_average


class SemanticReversalTool:
    def __init__(self, alpha=0.1, save_dir="ema_reversal_targets"):
        self.decomp = DECOMP(alpha=alpha)
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def get_reversal_target(self, history_data, future_data, intensity=1.0):
        """
        语义级趋势调节工具
        Args:
            intensity: 调节强度
                0.0: 原封不动
                1.0: 趋势抹平 (Flatten)
                2.0: 趋势完全反转 (Full Reversal)
                推荐值: 0.8 ~ 1.2 (减缓趋势或微弱反转)
        """
        B, L_f, N, C = future_data.shape
        device = future_data.device
        
        _, trend_f = self.decomp(future_data)
        start_trend = trend_f[:, 0:1]
        end_trend = trend_f[:, -1:]
        net_growth = end_trend - start_trend
        
        t = torch.linspace(0, 1, L_f, device=device).view(1, L_f, 1, 1)
        
        # 关键改动：系数变为 (1 - intensity)
        adjustment = (1 - intensity) * t * net_growth
        
        target_adjusted = future_data - adjustment
        return target_adjusted
    def save_comparison_plot(self, history, future, target, iters_num=0, idx=0):
        """保存对比图，不显示"""
        h = history[idx, :, 0, 0].detach().cpu().numpy()
        f = future[idx, :, 0, 0].detach().cpu().numpy()
        t = target[idx, :, 0, 0].detach().cpu().numpy()

        plt.figure(figsize=(12, 6))
        # 历史
        plt.plot(np.arange(len(h)), h, label="History", color="gray", alpha=0.5)
        # 原始未来 (上升趋势)
        plt.plot(np.arange(len(h), len(h)+len(f)), f, label="Original Future (Up)", color="blue", alpha=0.7)
        # 攻击目标 (反转趋势)
        plt.plot(np.arange(len(h), len(h)+len(t)), t, label="Target Target (Down)", color="red", linestyle="--")
        
        plt.axvline(x=len(h)-1, color='black', linestyle=':', alpha=0.5)
        plt.title(f"EMA-based Semantic Trend Reversal (Iter {iters_num})")
        plt.legend()
        
        save_path = os.path.join(self.save_dir, f"target_iter_{iters_num}.png")
        plt.savefig(save_path)
        plt.close()


class SeasonalAugmentor(nn.Module):
    """
    基于分解的季节性增强模块
    """
    def __init__(self, alpha=0.3):
        super(SeasonalAugmentor, self).__init__()
        self.decomp = DECOMP(alpha)
    
    def adjust_seasonal_strength(self, data, strength_factor):
        """
        调整季节性强度
        strength_factor: 季节性强度乘数
        """
        seasonal, trend = self.decomp(data)
        adjusted_seasonal = seasonal * strength_factor
        return trend + adjusted_seasonal

def masked_mae_per_sample(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Calculate the Masked Mean Absolute Error (MAE) for each sample between the predicted and target values,
    while ignoring the entries in the target tensor that match the specified null value.

    This function returns the MAE for each sample individually, rather than the mean across all samples.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor. Shape: [B, ...]
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor. 
            Default is `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A tensor of shape [B] representing the masked mean absolute error for each sample.

    """

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.0)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to avoid bias in the loss due to the number of valid entries
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    loss = torch.abs(prediction - target)
    loss = loss * mask  # Apply the mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    # 计算每个样本的损失，而不是整个batch的平均值
    batch_size = prediction.shape[0]
    sample_dims = list(range(1, loss.dim()))  # 除了batch维度外的所有维度
    loss_per_sample = loss.mean(dim=sample_dims)  # 对每个样本的所有维度取平均
    
    return loss_per_sample  # 形状: [B]
def masked_mse_per_sample(prediction: torch.Tensor, target: torch.Tensor, null_val: float = np.nan) -> torch.Tensor:
    """
    Calculate the Masked Mean Absolute Error (MAE) for each sample between the predicted and target values,
    while ignoring the entries in the target tensor that match the specified null value.

    This function returns the MAE for each sample individually, rather than the mean across all samples.

    Args:
        prediction (torch.Tensor): The predicted values as a tensor. Shape: [B, ...]
        target (torch.Tensor): The ground truth values as a tensor with the same shape as `prediction`.
        null_val (float, optional): The value considered as null or missing in the `target` tensor. 
            Default is `np.nan`. The function will mask all `NaN` values in the target.

    Returns:
        torch.Tensor: A tensor of shape [B] representing the masked mean absolute error for each sample.

    """

    if np.isnan(null_val):
        mask = ~torch.isnan(target)
    else:
        eps = 5e-5
        mask = ~torch.isclose(target, torch.tensor(null_val).expand_as(target).to(target.device), atol=eps, rtol=0.0)

    mask = mask.float()
    mask /= torch.mean(mask)  # Normalize mask to avoid bias in the loss due to the number of valid entries
    mask = torch.nan_to_num(mask)  # Replace any NaNs in the mask with zero

    loss = (prediction - target) ** 2
    loss = loss * mask  # Apply the mask to the loss
    loss = torch.nan_to_num(loss)  # Replace any NaNs in the loss with zero

    # 计算每个样本的损失，而不是整个batch的平均值
    batch_size = prediction.shape[0]
    sample_dims = list(range(1, loss.dim()))  # 除了batch维度外的所有维度
    loss_per_sample = loss.mean(dim=sample_dims)  # 对每个样本的所有维度取平均
    
    return loss_per_sample  # 形状: [B]
class SimpleTimeSeriesForecastingRunner_adv(BaseTimeSeriesForecastingRunner_adv):
    """
    A Simple Runner for Time Series Forecasting: 
    Selects forward and target features. This runner is designed to handle most cases.

    Args:
        cfg (Dict): Configuration dictionary.
    """

    def __init__(self, cfg: Dict):

        super().__init__(cfg)
        self.forward_features = cfg['MODEL'].get('FORWARD_FEATURES', None)
        self.target_features = cfg['MODEL'].get('TARGET_FEATURES', None)
        self.target_time_series = cfg['MODEL'].get('TARGET_TIME_SERIES', None)
    
    def preprocessing(self, input_data: Dict) -> Dict:
        """Preprocess data.

        Args:
            input_data (Dict): Dictionary containing data to be processed.

        Returns:
            Dict: Processed data.
        """

        if self.scaler is not None:
            input_data['target'] = self.scaler.transform(input_data['target'])
            input_data['inputs'] = self.scaler.transform(input_data['inputs'])
        # TODO: add more preprocessing steps as needed.
        return input_data

    def postprocessing(self, input_data: Dict) -> Dict:
        """Postprocess data.

        Args:
            input_data (Dict): Dictionary containing data to be processed.

        Returns:
            Dict: Processed data.
        """

        # rescale data
        if self.scaler is not None and self.scaler.rescale:
            input_data['prediction'] = self.scaler.inverse_transform(input_data['prediction'])
            input_data['target'] = self.scaler.inverse_transform(input_data['target'])
            input_data['inputs'] = self.scaler.inverse_transform(input_data['inputs'])
            
        # subset forecasting
        if self.target_time_series is not None:
            input_data['target'] = input_data['target'][:, :, self.target_time_series, :]
            input_data['prediction'] = input_data['prediction'][:, :, self.target_time_series, :]

        # TODO: add more postprocessing steps as needed.
        return input_data
    # ---------- 频域工具 ----------
    
    def _rfft_time(self, x: torch.Tensor, time_dim: int = 1):
        return torch.fft.rfft(x, dim=time_dim, norm='ortho')

    def _irfft_time(self, X: torch.Tensor, n_time: int, time_dim: int = 1):
        return torch.fft.irfft(X, n=n_time, dim=time_dim, norm='ortho')

    def _band_mask(self, L: int, band: str, low_ratio: float = 0.10, mid_ratio=(0.10, 0.40)) -> torch.Tensor:
        """基于 rfft 频点划分低/中/高频，返回 [K] bool 掩码"""
        K = L // 2 + 1
        nyq = K - 1
        m = torch.ones(K, dtype=torch.bool)  # 初始化全True，表示所有频带都可以被扰动
        
        if band == 'low':
            # 低频部分保留扰动
            hi = int((low_ratio * nyq))
            m[:hi]=True
            m[hi + 1:] = False  # 屏蔽掉高频和中频部分
        elif band == 'mid':
            # 中频部分保留扰动
            lo = max(int(mid_ratio[0] * nyq) + 1, 1)
            hi = min(int(mid_ratio[1] * nyq), nyq)
            m[:lo] = False  # 屏蔽掉低频
            m[hi + 1:] = False  # 屏蔽掉高频
        elif band == 'high':
            # 高频部分保留扰动
            cut = int(mid_ratio[1] * nyq)
            m[:cut + 1] = False  # 屏蔽掉低频和中频部分
        elif band == 'all':
            return m
        else:
            raise ValueError(f"Unknown band {band}")
        
        # Debug print: Check masked frequencies
        print(f"Mask for {band} band: {m}")
        return m

    def basic_cw_attack(
        self,
        time_eps: float = 0.25,     # 扰动约束
        alpha:float=0,
        lr: float = 0.01,       # 论文中Adam优化器的学习率
        iters: int = 100,       # 论文中的最大迭代次数
        c: float = 1,         # 损失平衡参数
        data: Dict = None
    ) -> Dict:
        """
        论文中最基础的C&W无目标攻击（回归版本）
        
        按照论文第3.2节描述，无目标攻击是最大化预测误差：
        argmax_η ||Y - f_θ(X+η)||₂², s.t. ||η||∞ ≤ ε
        
        论文中实际使用的是C&W框架：min ||r||₂ + MSE(f(x+r), y)
        但对于无目标攻击，他们使用真实序列作为目标序列
        """
        
        epsilon=time_eps
        # 1. 获取原始数据
        ori_history_data = self.to_running_device(data['inputs'])
        future_data = self.to_running_device(data['target'])
        
        # 归一化数据
        history_data_normalized = self.scaler.transform(ori_history_data)
        future_data_normalized = self.scaler.transform(future_data)
        future_data_normalized2=future_data_normalized.clone()
        future_data_normalized = self._generate_attack_target(history_data_normalized, future_data_normalized)
        attack_mode = self.attack_cfg.get("MODE", "untargeted")
        loss_sign = -1 if attack_mode == "targeted" else 1
        # 初始化对抗样本
        X_adv = ori_history_data.clone().detach().requires_grad_(True)
        temp_data1 = {'inputs': X_adv, 'target': future_data_normalized2}
        
        # 调用模型前向传播
        outputs1 = self._forward_without_preprocessing(data=temp_data1, epoch=None, iter_num=None, train=False)
        loss_per_sample_b = masked_mae_per_sample(outputs1['prediction'], outputs1['target'], null_val=np.nan).detach()
        
        # 3. 初始化扰动（从零开始）
        delta = torch.zeros_like(ori_history_data, requires_grad=True)
    
        # 4. 使用Adam优化器（论文中的设置）
        optimizer = torch.optim.Adam([delta], lr=lr)
        
        # 5. 迭代优化
        for i in range(iters):
            # 生成对抗样本
            adv_inputs = ori_history_data + delta
            
            # 前向传播
            temp_data_adv = {'inputs': adv_inputs, 'target': future_data_normalized}
            outputs = self._forward_without_preprocessing(data=temp_data_adv, train=False)
            current_pred = outputs['prediction']

            # 对于无目标攻击，我们希望最大化这个误差
            mae_error = self.loss(outputs['prediction'], outputs['target'])
            l2_norm = torch.mean(delta ** 2)  # L2范数的平方
            
            # 总损失 = L2正则化 + c * 预测误差
            # 我们最小化这个损失，但预测误差大时损失会增加
            total_loss = l2_norm-c*mae_error*loss_sign
            # 优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 6. 投影裁剪（论文中参考PGD的约束策略）
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -time_eps, time_eps)
            
            
        # 7. 最终评估
        final_adv_inputs = ori_history_data + delta.detach()
        
        with torch.no_grad():
            temp_data_final = {'inputs': final_adv_inputs, 'target': future_data_normalized}
            outputs_final = self._forward_without_preprocessing(data=temp_data_final, train=False)
            loss_per_sample_a = masked_mae_per_sample(outputs_final['prediction'], outputs_final['target'], null_val=np.nan)
            loss_reduction_per_sample=loss_per_sample_b-loss_per_sample_a
            
            # 计算扰动大小
        final_inputs_perturbation = delta.detach()
        final_inputs_perturbation = final_inputs_perturbation[:, :, :, self.forward_features]    
        reduced_samples_count = (loss_reduction_per_sample > 0).sum().item()
        # 返回结果
        return {
            'inputs': outputs_final['inputs'],
            'original_inputs': ori_history_data,
            'prediction': outputs_final['prediction'].detach(),
            'target': outputs_final['target'].detach(),
            'perturbation': final_inputs_perturbation,
            'count':reduced_samples_count
        }
        
        
    def fre_cw_attack(
        self,
        time_eps: float = 0.25,     # 扰动约束
        alpha:float=0,
        lr: float = 0.01,          # Adam优化器的学习率
        iters: int = 100,          # 最大迭代次数
        c: float = 1.0,            # 时域损失平衡参数
        freq_alpha: float = 0.6,   # 频域损失权重，论文中最佳区间0.5-0.7
        data: Dict = None
    ) -> Dict:
        """
        Fre-CW无目标攻击版本
        
        基于论文第3.2节和Algorithm 1，结合时域和频域损失
        总损失: L = α·L_freq + (1-α)·L_time
        """
        
        # 1. 获取原始数据
        ori_history_data = self.to_running_device(data['inputs'])
        future_data = self.to_running_device(data['target'])
        
        # 归一化数据
        history_data_normalized = self.scaler.transform(ori_history_data)
        future_data_normalized = self.scaler.transform(future_data)
        future_data_normalized2=future_data_normalized.clone()
        future_data_normalized = self._generate_attack_target(history_data_normalized, future_data_normalized)

        attack_mode = self.attack_cfg.get("MODE", "untargeted")
        loss_sign = -1 if attack_mode == "targeted" else 1
        # 初始化对抗样本
        X_adv = ori_history_data.clone().detach().requires_grad_(True)
        temp_data1 = {'inputs': X_adv, 'target': future_data_normalized2}
        
        # 调用模型前向传播
        outputs1 = self._forward_without_preprocessing(data=temp_data1, epoch=None, iter_num=None, train=False)
        loss_per_sample_b = masked_mae_per_sample(outputs1['prediction'], outputs1['target'], null_val=np.nan).detach()
        
        
        # 3. 初始化扰动（从零开始）
        delta = torch.zeros_like(ori_history_data, requires_grad=True)
        
        # 4. 使用Adam优化器（论文中的设置）
        optimizer = torch.optim.Adam([delta], lr=lr)
        
        # 5. FFT辅助函数（用于频域变换）
        def compute_fft(x: torch.Tensor) -> torch.Tensor:
            """计算输入张量的FFT，自动识别特征维度"""
            batch_size, seq_len = x.shape[0], x.shape[1]
            # 无论后面是 (7, 5) 还是 (7, 1)，全部展平到最后一维
            x_flat = x.reshape(batch_size, seq_len, -1)
            x_fft = torch.fft.rfft(x_flat, dim=1, norm='ortho')
            magnitude = torch.abs(x_fft)
            # 归一化
            magnitude_mean = magnitude.mean(dim=1, keepdim=True).detach()
            return magnitude / (magnitude_mean + 1e-8)
        
        # 6. 计算频域损失函数
        def compute_frequency_loss(x_orig: torch.Tensor, x_adv: torch.Tensor,
                                 pred_target: torch.Tensor, pred_adv: torch.Tensor, 
                                 mode: str = "untargeted") -> torch.Tensor:
            """
            修正后的频域损失函数：自动适配不同通道数
            """
            # 1. 计算输入端的频域表示 (隐蔽性)
            F_x_orig = compute_fft(x_orig) # 这里自动处理 35 维 (7*5)
            F_x_adv = compute_fft(x_adv)
            loss1 = F.l1_loss(F_x_adv, F_x_orig)
    
            # 2. 计算输出端的频域表示 (引导/破坏)
            # 注意：这里直接传入张量，不需要再手动 reshape
            # compute_fft 内部会自动处理 x.reshape(batch, seq_len, -1)
            
            # 确保 pred_target 至少是 4 维，如果不是，补一维通道
            if pred_target.dim() == 3:
                p_target = pred_target.unsqueeze(-1)
                p_adv = pred_adv.unsqueeze(-1)
            else:
                p_target = pred_target
                p_adv = pred_adv
    
            F_pred_target = compute_fft(p_target) # 这里自动处理 7 维 (7*1)
            F_pred_adv = compute_fft(p_adv)
            
            dist2 = F.l1_loss(F_pred_adv, F_pred_target)
            
            # 模式切换逻辑
            if mode == "targeted":
                loss2 = dist2   # 有目标：拉近距离 (正号)
            else:
                loss2 = -dist2  # 无目标：推开距离 (负号)
            
            return loss1 + loss2
        
        # 7. 迭代优化
        for i in range(iters):
            # 生成对抗样本
            adv_inputs = ori_history_data + delta
            
            # 前向传播
            temp_data_adv = {'inputs': adv_inputs, 'target': future_data_normalized}
            outputs = self._forward_without_preprocessing(data=temp_data_adv, train=False)
            current_pred = outputs['prediction']
            
            # 计算时域误差（MAE）
            mae_error = self.loss(current_pred, future_data_normalized)
            
            # 计算L2范数（扰动的平方和）
            l2_norm = torch.mean(delta ** 2)
            
            # 计算时域损失：L_time = ||r||₂ - c * MAE(f(x+r), y)
            # 负号是因为我们要最大化MAE（使预测误差更大）
            time_loss = l2_norm - c * mae_error*loss_sign
            future_target_only = future_data_normalized[:, :, :, self.target_features] 
            freq_loss = compute_frequency_loss(
                ori_history_data, 
                adv_inputs, 
                future_target_only, # 这里的参数名取决于你传入的是伪目标还是真值
                current_pred,
                mode=attack_mode
            )
            
            
            # 总损失：L = α·L_freq + (1-α)·L_time （论文公式8）
            total_loss = freq_alpha * freq_loss + (1 - freq_alpha) * time_loss
            
            # 优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 8. 投影裁剪（论文中参考PGD的约束策略）
            with torch.no_grad():
                delta.data = torch.clamp(delta.data, -time_eps, time_eps)
        
        # 9. 最终评估
        final_adv_inputs = ori_history_data + delta.detach()
        
        with torch.no_grad():
            temp_data_final = {'inputs': final_adv_inputs, 'target': future_data_normalized}
            outputs_final = self._forward_without_preprocessing(data=temp_data_final, train=False)
            loss_per_sample_a = masked_mae_per_sample(outputs_final['prediction'], outputs_final['target'], null_val=np.nan)
            loss_reduction_per_sample=loss_per_sample_b-loss_per_sample_a
            
        # 计算扰动大小
        final_inputs_perturbation = delta.detach()
            
        final_inputs_perturbation = final_inputs_perturbation[:, :, :, self.forward_features]
        # 返回结果
        reduced_samples_count = (loss_reduction_per_sample > 0).sum().item()
        # 返回结果
        return {
            'inputs': outputs_final['inputs'],
            'original_inputs': ori_history_data,
            'prediction': outputs_final['prediction'].detach(),
            'target': outputs_final['target'].detach(),
            'perturbation': final_inputs_perturbation,
            'count':reduced_samples_count
        }


    def batch_remove_dominant_frequencies(self, batch_data: np.ndarray, 
                                     top_k_to_remove: int = 1,
                                     removal_strength: float = 0.8) -> np.ndarray:
        """
        自动统一强度的频率移除（策略一：自适应能量消减）
        
        Args:
            batch_data: 批次数 [B, L, N, C]
            top_k_to_remove: 要操作的前k个主要频率
            target_reduction_ratio: 目标消减总交流能量的比例 (0-1)。
                                    例如 0.3 表示无论主频多强，最终都只砍掉全序列交流能量的 30%。
        """
        B, L, N, C = batch_data.shape
        modified_data = batch_data.copy() # 保护原数据，尤其是时间特征通道
        
        all_adaptive_strengths = []
        all_ratios = []
    
        for b in range(B):
            for n in range(N):
                # 仅针对通道 0 (数值通道)
                series = batch_data[b, :, n, 0]
                
                # 进行频率移除
                modified_series = self.remove_dominant_frequencies_single(
                    series, top_k_to_remove, removal_strength
                )
                
                modified_data[b, :, n, 0] = modified_series
    
        return modified_data
    
    def remove_dominant_frequencies_single(self, time_series: np.ndarray,
                                          top_k_to_remove: int = 1,
                                          removal_strength: float = 0.8) -> np.ndarray:
        """
        对单个序列进行频率移除（核心函数）
        
        Args:
            time_series: 单个时域信号 [L]
            top_k_to_remove: 要移除的前k个主要频率
            removal_strength: 移除强度 (0-1)
        
        Returns:
            重建后的时域信号 [L]
        """
        L = len(time_series)
    
        # 计算FFT
        fft_result = np.fft.rfft(time_series)
        
        # 计算幅度谱（用于找出主要频率）
        magnitudes = np.abs(fft_result)
        
        # 找出主要频率（按幅度排序）
        # 排除直流分量（索引0），因为它代表平均值
        nonzero_indices = np.arange(1, len(magnitudes))  # 跳过直流
        sorted_indices = nonzero_indices[np.argsort(magnitudes[1:])[::-1]]  # 降序
        
        # 创建修改后的FFT结果
        modified_fft = fft_result.copy()

        # 3. 执行移除 (核心修正位置)
        modified_fft = fft_result.copy()
        
        # 移除/减弱主要频率成分
        for i in range(min(top_k_to_remove, len(sorted_indices))):
            idx = sorted_indices[i]
            # 直接缩放FFT系数
            modified_fft[idx] *= (1 - removal_strength)
        
        # 重建时域信号
        reconstructed_series = np.fft.irfft(modified_fft, n=L)
        
        # 确保实数性（去除微小虚部）
        reconstructed_series = np.real(reconstructed_series)
        
        return reconstructed_series

    def enhance_dominant_frequencies_single(self, time_series: np.ndarray,
                                            top_k_to_enhance: int = 1,
                                            enhancement_factor: float = 1.5) -> np.ndarray:
        """
        对单个序列增强主要频率（除直流外）
        
        Args:
            time_series: 单个时域信号 [L]
            top_k_to_enhance: 要增强的前k个主要频率（按幅度排序）
            enhancement_factor: 增强因子（>1 增强，=1 不变，<1 减弱）
                              建议值 1.2 ~ 2.0
        
        Returns:
            增强后的时域信号 [L]
        """
        L = len(time_series)
        
        # 计算FFT（只取正频率部分）
        fft_result = np.fft.rfft(time_series)
        magnitudes = np.abs(fft_result)
        
        # 排除直流分量（索引0），从索引1开始
        if L <= 1:
            return time_series.copy()
        
        # 获取非直流分量的幅度
        non_dc_magnitudes = magnitudes[1:]
        if len(non_dc_magnitudes) == 0:
            return time_series.copy()
        
        # 按幅度降序排序，得到索引（相对于非直流数组的索引）
        sorted_indices_in_non_dc = np.argsort(non_dc_magnitudes)[::-1]  # 降序
        
        # 转换为原始fft数组中的索引（+1 因为跳过了直流）
        top_indices = [idx + 1 for idx in sorted_indices_in_non_dc[:top_k_to_enhance]]
        
        # 增强选中的频率成分
        modified_fft = fft_result.copy()
        for idx in top_indices:
            modified_fft[idx] *= enhancement_factor
        
        # 重建时域信号
        reconstructed = np.fft.irfft(modified_fft, n=L)
        reconstructed = np.real(reconstructed)  # 去除微小虚部
        
        return reconstructed
    
    
    def batch_enhance_dominant_frequencies(self, batch_data: np.ndarray,
                                           top_k_to_enhance: int = 1,
                                           enhancement_factor: float = 1.5) -> np.ndarray:
        """
        对整批数据增强主要频率
        
        Args:
            batch_data: 批次数 [B, L, N, C]
            top_k_to_enhance: 每个序列要增强的前k个主要频率
            enhancement_factor: 增强因子（>1 增强）
        
        Returns:
            增强后的批量数据 [B, L, N, C]
        """
        B, L, N, C = batch_data.shape
        enhanced_data = np.zeros_like(batch_data)
        
        for b in range(B):
            for n in range(N):
                series = batch_data[b, :, n, 0]
                enhanced_series = self.enhance_dominant_frequencies_single(
                    series, top_k_to_enhance, enhancement_factor
                )
                enhanced_data[b, :, n, 0] = enhanced_series
        
        return enhanced_data
    

    def fgsm_attack(self, time_eps: float = 0.03, alpha: float = 0.01, 
                          iters: int = 40, data: Dict = None) -> Dict:
        """
        在时域上进行标准的FGSM攻击，不使用频域变换。
        FGSM是单步攻击，比BIM更简单快速。
        """
        
        ori_history_data = self.to_running_device(data['inputs'])
        future_data = self.to_running_device(data['target'])
        history_data_normalized = self.scaler.transform(ori_history_data)
        future_data_normalized = self.scaler.transform(future_data)
        future_data_normalized2=future_data_normalized.clone()
        future_data_normalized = self._generate_attack_target(history_data_normalized, future_data_normalized)
        
        assert ori_history_data.dim() == 4  # 确保输入数据是四维的 [B, L, N, C]
        feature_mask = torch.zeros_like(ori_history_data)
        feature_mask[:, :, :, 0:1] = 1  # 只对第一个特征通道启用扰动
        attack_mode = self.attack_cfg.get("MODE", "untargeted")
        loss_sign = -1 if attack_mode == "targeted" else 1
        # 直接在时域初始化扰动
        X_adv = ori_history_data.clone().detach().requires_grad_(True)
        
        temp_data1 = {'inputs': X_adv, 'target': future_data_normalized2}
        # 调用模型前向传播
        outputs1 = self._forward_without_preprocessing(data=temp_data1, epoch=None, iter_num=None, train=False)
        loss_per_sample_b = masked_mae_per_sample(outputs1['prediction'], outputs1['target'], null_val=np.nan)
        # FGSM单步前向传播
        temp_data = {'inputs': X_adv, 'target': future_data_normalized}
        
        # 调用模型前向传播
        outputs = self._forward_without_preprocessing(data=temp_data, epoch=None, iter_num=None, train=False)
        # 计算损失
        loss = self.loss(outputs['prediction'], outputs['target'])
        # 清空梯度并执行反向传播
        self.model.zero_grad()
        loss.backward()
    
        # FGSM单步更新 - 使用梯度的符号乘以epsilon
        with torch.no_grad():
            grad_sign = torch.sign(X_adv.grad)  # 获取梯度的符号
            perturbation = time_eps * grad_sign*loss_sign  # FGSM直接使用epsilon乘以梯度符号
            perturbation = perturbation * feature_mask
            # 生成对抗样本
            X_adv = ori_history_data + perturbation
            
        # 计算最终输出
        temp_data = {'inputs': X_adv, 'target': future_data_normalized}
        outputs = self._forward_without_preprocessing(data=temp_data, epoch=None, iter_num=None, train=False)
        loss_per_sample_a = masked_mae_per_sample(outputs['prediction'], outputs['target'], null_val=np.nan)
        loss_reduction_per_sample=loss_per_sample_b-loss_per_sample_a
        prediction = outputs['prediction']
        final_inputs_perturbation = X_adv - ori_history_data
        ori_history_data = ori_history_data[:, :, :, self.target_features]
        final_inputs_perturbation = final_inputs_perturbation[:, :, :, self.forward_features]
        reduced_samples_count = (loss_reduction_per_sample > 0).sum().item()
        # 返回字典
        return {
            'inputs': outputs['inputs'],
            'original_inputs': ori_history_data,  # 原始输入数据
            'prediction': prediction.detach(),  # 预测结果
            'target': outputs['target'].detach(),  # 目标数据
            'perturbation': final_inputs_perturbation,
            'count':reduced_samples_count
        }

    def _inject_adversarial_features(self, X_original, X_adv_partial):
        """
        动态将攻击产生的局部特征注入到全量特征矩阵中。
        X_original: 原始全量输入 [B, L, N, C_total] (如 7 通道)
        X_adv_partial: 攻击产生的对抗特征 [B, L, N, C_attacked] (如 1 通道)
        """
        
        # 2. 动态创建副本
        X_full_adv = X_original.clone()
        
        # 3. 动态定位并替换受攻击的通道
        # 假设你攻击的是 forward_features 里的特征（通常是第一个，即特征0）
        # 如果 self.forward_features = [0, 1, 2, 3, 4]，这里会把 X_adv_partial 填入索引 0
        target_idx = self.forward_features[0] 
        X_full_adv[..., target_idx : target_idx + X_adv_partial.shape[-1]] = X_adv_partial
        
        return X_full_adv
    def aaim_attack(self, time_eps: float = 0.03, data: Dict = None, alpha:float=0,iters:int=0,
                importance_ratio: float = 0.05, use_l1_importance: bool = True) -> Dict:
        """
        基于重要性度量的对抗攻击 (AAIM)
        
        Args:
            time_eps: FGSM扰动强度
            data: 输入数据
            importance_ratio: 重要性比例 (P%，如0.05表示5%)
            use_l1_importance: 是否使用L1距离计算重要性
        """
        # 1. 首先用FGSM生成完整的对抗序列（原始方法）
        fgsm_result = self.fgsm_attack(time_eps=time_eps, data=data)
        ori_history_data = self.to_running_device(data['inputs'])
        future_data = self.to_running_device(data['target'])
        history_data_normalized = self.scaler.transform(ori_history_data)
        future_data_normalized = self.scaler.transform(future_data)
        future_data_normalized2=future_data_normalized.clone()
        future_data_normalized = self._generate_attack_target(history_data_normalized, future_data_normalized)
        # 获取关键数据
        X_original = self.to_running_device(data['inputs']) # 始终拿全量的 [B, L, N, 7]
        X_original = self.scaler.transform(X_original)
        X_adv_only = fgsm_result['inputs'] # 可能是 [B, L, N, 1]
        
        X_adv_full = self._inject_adversarial_features(X_original, X_adv_only)
        future_target = future_data_normalized  # 目标数据
        full_perturbation = fgsm_result['perturbation']  # 完整扰动
        
        batch_size, seq_len, num_nodes, num_features = X_original.shape
        temp_data1 = {'inputs': X_original, 'target': future_data_normalized2}
        
        # 调用模型前向传播
        outputs1 = self._forward_without_preprocessing(data=temp_data1, epoch=None, iter_num=None, train=False)
        loss_per_sample_b = masked_mae_per_sample(outputs1['prediction'], outputs1['target'], null_val=np.nan)
        # 2. 计算每个时间点的重要性
        importance_scores = self._compute_importance_scores(
            X_original=X_original,
            X_adv_full=X_adv_full,
            future_target=future_target,
            use_l1=use_l1_importance
        )
        
        # 3. 选择最重要的时间点
        num_important_points = int(seq_len * importance_ratio)
        
        # 对每个样本单独选择重要时间点
        selected_masks = torch.zeros_like(X_original, dtype=torch.bool)
        
        for b in range(batch_size):
            # 获取当前样本的重要性分数
            sample_importance = importance_scores[b]  # [seq_len]
            
            # 选择重要性最高的时间点
            _, top_indices = torch.topk(sample_importance, 
                                       k=num_important_points, 
                                       dim=0)
            
            # 创建掩码
            selected_masks[b, top_indices, :, 0] = True
        
        # 4. 生成稀疏扰动的对抗样本
        # 只保留重要时间点的扰动，其他点用原始值
        X_adv_sparse = X_original.clone()
        X_adv_sparse[selected_masks] = X_adv_full[selected_masks]
        
        # 计算稀疏扰动
        sparse_perturbation = X_adv_sparse - X_original
        
        # 5. 评估稀疏扰动攻击的效果
        temp_data = {'inputs': X_adv_sparse, 'target': future_target}
        outputs = self._forward_without_preprocessing(
            data=temp_data, epoch=None, iter_num=None, train=False
        )
        loss_per_sample_a = masked_mae_per_sample(outputs['prediction'], outputs['target'], null_val=np.nan)
        loss_reduction_per_sample=loss_per_sample_b-loss_per_sample_a
        reduced_samples_count = (loss_reduction_per_sample > 0).sum().item()
        return {
            'inputs': X_adv_sparse[:, :, :, 0:1].detach(),
            'original_inputs': X_original,
            'prediction': outputs['prediction'].detach(),
            'target': outputs['target'].detach(),
            'perturbation': sparse_perturbation,
            'count':reduced_samples_count
        }
    
    def _compute_importance_scores(self, X_original, X_adv_full, future_target, use_l1=True):
        """
        计算每个时间点的重要性分数（对应AAIM算法步骤1-4）
        
        论文中的方法：对于每个时间点t，构造混合序列，计算预测误差
        """
        batch_size, seq_len, num_nodes, num_features = X_original.shape
        importance_scores = torch.zeros(batch_size, seq_len, device=X_original.device)
        
        # ---------------------------------------------------------
        # 设置子批次大小 (Sub-batch size)
        # 对于 Informer，建议设为 4 或 8。
        # ---------------------------------------------------------
        sub_batch_size = 32 
    
        with torch.no_grad():
            # 1. 先分段计算基准误差 (error_full)
            all_error_full = []
            for i in range(0, batch_size, sub_batch_size):
                end_i = min(i + sub_batch_size, batch_size)
                sub_X_adv = X_adv_full[i:end_i]
                sub_target = future_target[i:end_i]
                
                out = self._forward_without_preprocessing(
                    data={'inputs': sub_X_adv, 'target': sub_target}, train=False
                )['prediction']
                
                if use_l1:
                    err = torch.abs(out - sub_target).mean(dim=(1,2,3))
                else:
                    err = ((out - sub_target)**2).mean(dim=(1,2,3))
                all_error_full.append(err)
            
            error_full = torch.cat(all_error_full) # [B]
    
            # 2. 逐个子批次、逐个时间步计算
            # 虽然慢，但绝对不会 OOM
            for i in range(0, batch_size, sub_batch_size):
                end_i = min(i + sub_batch_size, batch_size)
                
                # 提取当前子批次的数据
                curr_orig = X_original[i:end_i]
                curr_adv = X_adv_full[i:end_i]
                curr_target = future_target[i:end_i]
                curr_error_full = error_full[i:end_i]
                
                for t in range(seq_len):
                    # 构造混合样本：当前子批次中，只有第 t 个点恢复原始
                    X_mixed = curr_adv.clone()
                    X_mixed[:, t, :, :] = curr_orig[:, t, :, :]
                    
                    out_mixed = self._forward_without_preprocessing(
                        data={'inputs': X_mixed, 'target': curr_target}, train=False
                    )['prediction']
                    
                    if use_l1:
                        error_mixed = torch.abs(out_mixed - curr_target).mean(dim=(1,2,3))
                    else:
                        error_mixed = ((out_mixed - curr_target)**2).mean(dim=(1,2,3))
                    
                    # 计算重要性
                    importance_scores[i:end_i, t] = torch.abs(curr_error_full - error_mixed)
                
                # 每处理完一个子批次，强制清理碎片
                torch.cuda.empty_cache()
                print(f" AAIM 进度: Batch {end_i}/{batch_size} 处理完毕")
    
        return importance_scores

    
    def generate_equal_inheritance_weights(self, X_adv, grad, ori_history, time_eps, iteration):
        """
        逻辑：
        1. 撞墙的点（扰动达到 time_eps）权重设为 0。
        2. 没撞墙的点平分总权重预算（L * N）。
        3. 这样没撞墙的点步长会变大，从而“瓜分”了撞墙点的能量。
        """
        
        with torch.no_grad():
            # 获取维度信息
            B, L, N, C = grad.shape
            num_total = L * N # 通道 0 的总预算
    
            # 1. 计算当前扰动量
            pert_0 = (X_adv - ori_history)[:, :, :, 0:1]
            
            # 2. 识别活跃点（还没撞墙的点）
            # 使用 1e-7 防止浮点数精度误差导致提前判定撞墙
            active_mask = (pert_0.abs() < (time_eps - 1e-7)).float()
            
            # 3. 统计每张图有多少活着的点
            num_active = active_mask.sum(dim=(1, 2), keepdim=True)
            
            # 4. 计算分配权重
            # 如果还有活着的点，每个活点分到的权重 = 总预算 / 活点数
            # 如果全部撞墙（num_active=0），则权重设为 0，防止除以零
            redivided_weight = num_total / (num_active + 1e-9)
            
            # 5. 生成通道 0 的最终权重
            # 撞墙的点因为 active_mask 为 0，权重变为 0
            # 没撞墙的点权重全部等于 redivided_weight（权重大小一样）
            w0 = active_mask * redivided_weight
            
            # 6. 组合最终权重矩阵
            # 其他通道保持 1.0（如果不攻击其他通道，1.0 无所谓，因为梯度是 0）
            final_weights = torch.ones_like(grad)
            final_weights[:, :, :, 0:1] = w0
            
            # 调试打印：观察权重是如何随迭代增加而“膨胀”的
            if iteration % 10 == 0 or iteration == 49:
                avg_w = w0[w0 > 0].mean().item() if (num_active > 0).any() else 0
                print(f"Iter {iteration}: 活点数 = {num_active.min().item():.0f}/{num_total}, "
                      f"活点分得权重 = {avg_w:.4f} (原始为 1.0)")
                
            return final_weights

    
    def generate_robin_hood_weights(self, X_adv, grad, ori_history, time_eps, iteration):

        with torch.no_grad():
            B, L, N, C = grad.shape

            num_total_per_node = L 
            
            # 1. 识别活跃点（还没撞墙的点）
            pert_0 = (X_adv - ori_history)[:, :, :, 0:1]
            active_mask = (pert_0.abs() < (time_eps - 1e-7)).float()
            

            num_active = active_mask.sum(dim=1, keepdim=True)
            g0_abs = grad[:, :, :, 0:1].abs()
            
            # 2. 提取梯度相对强度
            g_active = g0_abs * active_mask
            

            avg_g_active = g_active.sum(dim=1, keepdim=True) / (num_active + 1e-9)
            
            # 计算相对强度比率
            relative_ratio = g_active / (avg_g_active + 1e-9)
            
            # 自然对比度压缩
            natural_importance = torch.sqrt(relative_ratio)
            
            # 3. 能量投影 (Energy Projection)
            # 使得每个节点活跃点的权重分布在 1.0 附近
            # dim=1 保证了归一化是在节点内部完成的
            node_internal_mean = natural_importance.sum(dim=1, keepdim=True) / (num_active + 1e-9)
            w_dist = natural_importance / (node_internal_mean + 1e-9)
            
            # 4. 显式罗宾汉重分配 (针对每个节点)
            # 如果某个节点活着的点少，那个点的基础放大倍数就高
            redistribution_factor = num_total_per_node / (num_active + 1e-9)
            
            # 最终权重
            w0 = active_mask * redistribution_factor * w_dist

            # 5. 最终能量对齐 (针对每个节点内部)
            # 确保每个节点在 Channel 0 上的权重总和严格等于 L
            current_sum = w0.sum(dim=1, keepdim=True)
            w0 = w0 * (num_total_per_node / (current_sum + 1e-9))
            
            # 6. 组合并返回
            final_weights = torch.ones_like(grad)
            final_weights[:, :, :, 0:1] = w0

            return final_weights


    def _generate_attack_target(self, history_norm, future_norm):
        """
        根据配置动态生成攻击目标 (Targeted Attack Ground Truth)
        history_norm: 归一化后的历史数据 [B, L, N, C]
        future_norm: 归一化后的未来真实数据 [B, L_f, N, C]
        """
        t_type = self.attack_cfg.get("TARGET_TYPE", "original")
        
        # 1. 趋势反转 (Semantic Trend Reversal)
        if t_type == "trend_reversal":
            # alpha 越小，趋势提取越平滑；0.8 是旋转强度
            tool = SemanticReversalTool(alpha=0.2)
            # 注意：这里的 0.8 建议通过配置传参，或者固定。返回的是全量维度的 Target
            target_reversed = tool.get_reversal_target(history_norm, future_norm,0.2)
            # 只选择目标特征 (e.g., 通道 0) 返回，用于计算 Loss
            return target_reversed
            
        # 1. 趋势反转 (Semantic Trend Reversal)
        elif t_type == "trend_enhance":
            # alpha 越小，趋势提取越平滑；0.8 是旋转强度
            tool = SemanticReversalTool(alpha=0.2)
            # 注意：这里的 0.8 建议通过配置传参，或者固定。返回的是全量维度的 Target
            target_reversed = tool.get_reversal_target(history_norm, future_norm,1.8)
            # 只选择目标特征 (e.g., 通道 0) 返回，用于计算 Loss
            return target_reversed

        # 2. 频率成分移除 (Dominant Frequency Removal)
        elif t_type == "freq_removal":
            # a. 先选择需要攻击的特征进行操作
            future_selected = self.select_input_features(future_norm) 
            # b. 转换成 numpy 运行你的移除函数
            future_np = future_selected.cpu().numpy()
            modified_np = self.batch_remove_dominant_frequencies(
                future_np, 
                top_k_to_remove=1, 
                removal_strength=0.85
            )
            # c. 转回 Tensor
            modified_target = torch.from_numpy(modified_np).to(future_norm.device)
            # 确保返回的特征维度与预测值一致
            return modified_target

        # 2. 频率成分移除 (Dominant Frequency Removal)
        elif t_type == "freq_enhance":
            # a. 先选择需要攻击的特征进行操作
            future_selected = self.select_input_features(future_norm) 
            # b. 转换成 numpy 运行你的移除函数
            future_np = future_selected.cpu().numpy()
            modified_np = self.batch_enhance_dominant_frequencies(
                future_np, 
                top_k_to_enhance=1,           
                enhancement_factor=1.4
            )
            # c. 转回 Tensor
            modified_target = torch.from_numpy(modified_np).to(future_norm.device)
            # 确保返回的特征维度与预测值一致
            return modified_target
        
        # 3. 季节性强度调节 (Seasonal Strength Adjustment)
        elif t_type == "seasonal":
            # 因子 > 1 增强季节性，< 1 削弱季节性
            strength_factor = self.attack_cfg.get("SEASONAL_FACTOR", 1.35)
            augmentor = SeasonalAugmentor(alpha=0.2)
            # 对未来数据进行分解并重新合成
            adjusted_target = augmentor.adjust_seasonal_strength(future_norm, strength_factor)
            return adjusted_target

        # 4. 默认/无目标 (Original Ground Truth)
        else:
            return future_norm

    def bim_attack(self, time_eps: float = 0.03, l2_eps: float = 1.0,alpha: float = 0.01, 
                          iters: int = 40, data: Dict = None) -> Dict:
        """
        在时域上进行标准的BIM攻击
        """
        ori_history_data = self.to_running_device(data['inputs'])
        future_data = self.to_running_device(data['target'])
        history_data_normalized = self.scaler.transform(ori_history_data)
        future_data_normalized = self.scaler.transform(future_data)
        future_data_normalized2=future_data_normalized.clone()
        future_data_normalized = self._generate_attack_target(history_data_normalized, future_data_normalized)
        assert ori_history_data.dim() == 4  # 确保输入数据是四维的 [B, L, N, C]
        feature_mask = torch.zeros_like(ori_history_data)
        feature_mask[:, :, :, 0:1] = 1  # 只对第一个特征通道启用扰动
        X_adv = ori_history_data.clone().detach().requires_grad_(True)
        
        attack_mode = self.attack_cfg.get("MODE", "untargeted")
        loss_sign = -1 if attack_mode == "targeted" else 1
        
        temp_data1 = {'inputs': X_adv, 'target': future_data_normalized2}
        # 调用模型前向传播
        outputs1 = self._forward_without_preprocessing(data=temp_data1, epoch=None, iter_num=None, train=False)
        loss_per_sample_b = masked_mae_per_sample(outputs1['prediction'], outputs1['target'], null_val=np.nan)

        for i in range(iters):
            X_adv.requires_grad_(True)
            temp_data = {'inputs': X_adv, 'target': future_data_normalized}
            outputs = self._forward_without_preprocessing(data=temp_data, epoch=None, iter_num=None, train=False)
            
            loss = self.loss(outputs['prediction'], outputs['target'])
            
            self.model.zero_grad()
            loss.backward(retain_graph=True)
           
            with torch.no_grad():
                
                time_update = alpha * torch.sign(X_adv.grad)*loss_sign
                
                X_adv = X_adv + time_update  # 按照梯度符号更新扰动，步长为alpha
                
                # 计算扰动
                perturbation = X_adv - ori_history_data
                perturbation = perturbation * feature_mask
                # 限制扰动的最大幅度
                perturbation_clamped = torch.clamp(perturbation, min=-time_eps, max=time_eps)
                # 更新对抗样本
                X_adv = ori_history_data + perturbation_clamped
                
            # 重新启用梯度计算，为下一次迭代做好准备
            X_adv = X_adv.detach().requires_grad_(True)
        
        temp_data = {'inputs': X_adv, 'target': future_data_normalized}
        outputs = self._forward_without_preprocessing(data=temp_data, epoch=None, iter_num=None, train=False)
        prediction = outputs['prediction']
        loss_per_sample_a = masked_mae_per_sample(outputs['prediction'], outputs['target'], null_val=np.nan)
        loss_reduction_per_sample=loss_per_sample_b-loss_per_sample_a
        final_inputs_perturbation = X_adv - ori_history_data
        
        ori_history_data=ori_history_data[:, :, :, self.target_features]
        final_inputs_perturbation = final_inputs_perturbation[:, :, :, self.forward_features]
        reduced_samples_count = (loss_reduction_per_sample > 0).sum().item()
        # 返回字典
        return {
            'inputs': outputs['inputs'],
            'original_inputs': ori_history_data,  # 原始输入数据
            'prediction': prediction.detach(),  # 预测结果
            'target': outputs['target'].detach(),  # 目标数据
            'perturbation': final_inputs_perturbation,
            'count':reduced_samples_count
        }    

    def tca_attack(self, time_eps: float = 0.03, alpha: float = 0.01, 
               iters: int = 200, data: Dict = None, 
               use_cosine_constraint: bool = True,
               ) -> Dict:
        """
        基于时态特性的对抗攻击 (TCA)
        
        Args:
            time_eps: 最大扰动阈值
            alpha: 迭代步长
            iters: 迭代次数
            data: 输入数据
            use_cosine_constraint: 是否使用时态特性约束（余弦相似度）
        """
        # 获取原始数据
        ori_history_data = self.to_running_device(data['inputs'])
        future_data = self.to_running_device(data['target'])
        # 数据归一化
        history_data_normalized = self.scaler.transform(ori_history_data)
        future_data_normalized = self.scaler.transform(future_data)
        future_data_normalized2=future_data_normalized.clone()
        # 获取目标特征
        future_data_normalized = self._generate_attack_target(history_data_normalized, future_data_normalized)
        
        # 创建特征掩码（只扰动第一个特征通道）
        feature_mask = torch.zeros_like(ori_history_data)
        feature_mask[:, :, :, 0:1] = 1
        
        attack_mode = self.attack_cfg.get("MODE", "untargeted")
        loss_sign = -1 if attack_mode == "targeted" else 1
        # 初始化对抗样本
        X_adv = ori_history_data.clone().detach().requires_grad_(True)
        X_original = ori_history_data.clone().detach()
        
        xsim_values = []  # 对抗样本与原始样本的相似度
        
        # 提前计算边界样本
        X_plus = X_original + time_eps
        X_minus = X_original - time_eps
        temp_data1 = {'inputs': X_adv, 'target': future_data_normalized2}
        
        # 调用模型前向传播
        outputs1 = self._forward_without_preprocessing(data=temp_data1, epoch=None, iter_num=None, train=False)
        loss_per_sample_b = masked_mae_per_sample(outputs1['prediction'], outputs1['target'], null_val=np.nan)
        # 计算边界样本的余弦相似度
        def compute_cosine_similarity(x1, x2):
            """计算两个张量之间的余弦相似度"""
            # 使用reshape替代view
            x1_flat = x1.reshape(x1.size(0), -1)
            x2_flat = x2.reshape(x2.size(0), -1)
            
            # 计算余弦相似度
            cosine_sim = F.cosine_similarity(x1_flat, x2_flat, dim=1)
            return cosine_sim.mean().item()
        
        # 计算边界相似度
        sim_plus = compute_cosine_similarity(X_original, X_plus)
        sim_minus = compute_cosine_similarity(X_original, X_minus)
        # TCA迭代攻击
        for i in range(iters):
            # 前向传播计算损失
            temp_data = {'inputs': X_adv, 'target': future_data_normalized}
            outputs = self._forward_without_preprocessing(data=temp_data, epoch=None, iter_num=None, train=False)
            loss = self.loss(outputs['prediction'], outputs['target'])
            # 清空梯度并反向传播
            self.model.zero_grad()
            loss.backward()
            
            # 更新对抗样本（TCA核心部分）
            with torch.no_grad():
                # 1. 计算梯度符号
                grad_sign = torch.sign(X_adv.grad)
                
                # 2. 应用梯度更新
                time_update = grad_sign * alpha*loss_sign
                X_candidate = X_adv + time_update
                
                # 3. 裁剪到扰动边界内
                X_candidate = torch.clamp(X_candidate, 
                                         min=X_original - time_eps, 
                                         max=X_original + time_eps)
                
                # 4. 应用特征掩码
                X_candidate = X_original + (X_candidate - X_original) * feature_mask
                
                # 5. 时态特性约束（TCA的关键创新）
                if use_cosine_constraint:
                    # 计算候选样本与原始样本的余弦相似度
                    sim_candidate = compute_cosine_similarity(X_original, X_candidate)
                    
                    # 根据论文Algorithm 1的约束条件
                    if sim_candidate > sim_plus:
                        X_adv = X_candidate
                    elif sim_candidate > sim_minus:
                        X_adv = X_candidate
                    else:
                        # 选择相似度更高的边界
                        if sim_plus > sim_minus:
                            X_adv = torch.clamp(X_original + time_eps, 
                                               min=X_original - time_eps, 
                                               max=X_original + time_eps)
                        else:
                            X_adv = torch.clamp(X_original - time_eps, 
                                               min=X_original - time_eps, 
                                               max=X_original + time_eps)
                else:
                    # 不使用时态约束，直接使用候选样本
                    X_adv = X_candidate
                
                # 计算当前相似度
                current_xsim = compute_cosine_similarity(X_original, X_adv)
                xsim_values.append(current_xsim)
                
            # 重新启用梯度计算
            X_adv = X_adv.detach().requires_grad_(True)
            
        # 最终评估
        with torch.no_grad():
            # 计算最终对抗样本的预测
            temp_final = {'inputs': X_adv, 'target': future_data_normalized}
            outputs_final = self._forward_without_preprocessing(
                data=temp_final, epoch=None, iter_num=None, train=False
            )
        # 计算扰动
        final_inputs_perturbation = X_adv - X_original
        loss_per_sample_a = masked_mae_per_sample(outputs_final['prediction'], outputs_final['target'], null_val=np.nan)
        loss_reduction_per_sample=loss_per_sample_b-loss_per_sample_a
        reduced_samples_count = (loss_reduction_per_sample > 0).sum().item()
        ori_history_data=ori_history_data[:, :, :, self.target_features]
        final_inputs_perturbation = final_inputs_perturbation[:, :, :, self.forward_features]
        # 返回结果
        return {
            'inputs': outputs_final['inputs'],
            'original_inputs': ori_history_data,
            'prediction': outputs_final['prediction'].detach(),
            'target': outputs_final['target'].detach(),
            'perturbation': final_inputs_perturbation,
            'count':reduced_samples_count
        }
    def mapgd_tsf_attack(self, time_eps: float = 0.03, 
                                alpha: float = 0.01, momentum: float = 0.75,
                                iters: int = 100, data: Dict = None,
                                checkpoint_interval: int = 10) -> Dict:
        """
        在时域上进行无目标版本的mAPGD-TSF攻击。
        基于mAPGD-TSF算法，包含自适应步长调整和动量机制。
        """
        # 准备数据
        ori_history_data = self.to_running_device(data['inputs'])
        future_data = self.to_running_device(data['target'])
        history_data_normalized = self.scaler.transform(ori_history_data)
        future_data_normalized = self.scaler.transform(future_data)
        future_data_normalized2=future_data_normalized.clone()
        future_data_normalized = self._generate_attack_target(history_data_normalized, future_data_normalized)
        assert ori_history_data.dim() == 4  # 确保输入数据是四维的 [B, L, N, C]
        
        # 特征掩码：只对某些特征通道启用扰动
        feature_mask = torch.zeros_like(ori_history_data)
        feature_mask[:, :, :, 0:1] = 1  # 只对第一个特征通道启用扰动
        attack_mode = self.attack_cfg.get("MODE", "untargeted")
        loss_sign = -1 if attack_mode == "targeted" else 1
        # 初始化对抗样本
        x_adv = ori_history_data.clone().detach().requires_grad_(True)
        x_original = ori_history_data.clone().detach()

        temp_data1 = {'inputs': x_adv, 'target': future_data_normalized2}
        # 调用模型前向传播
        outputs1 = self._forward_without_preprocessing(data=temp_data1, epoch=None, iter_num=None, train=False)
        loss_per_sample_b = masked_mae_per_sample(outputs1['prediction'], outputs1['target'], null_val=np.nan)
        
        # 记录最佳解
        x_best = x_adv.clone().detach()
        f_best = -float('inf') if attack_mode == "untargeted" else float('inf')
        # 初始化步长
        step_size = alpha
        # 动量相关变量
        x_prev = x_adv.clone().detach()  # 用于动量计算
        
        # 检查点相关变量
        checkpoint_losses = []  # 记录每个检查点的损失
        window_size = checkpoint_interval
        rho = 0.75  # 条件1中的比例参数，来自原论文
        
        # 定义投影函数：将对抗样本投影到扰动约束范围内
        def project_to_constraint(x_adv, x_original, eps):
            """将对抗样本投影到L∞约束范围内"""
            perturbation = x_adv - x_original
            perturbation = perturbation * feature_mask  # 应用特征掩码
            perturbation_clamped = torch.clamp(perturbation, min=-eps, max=eps)
            return x_original + perturbation_clamped
        
        # 计算初始损失
        temp_data = {'inputs': x_adv, 'target': future_data_normalized}
        outputs = self._forward_without_preprocessing(data=temp_data, epoch=None, iter_num=None, train=False)
        loss_initial = self.loss(outputs['prediction'], outputs['target'])
        
        # 计算梯度
        self.model.zero_grad()
        loss_initial.backward(retain_graph=True)
        
        # 获取梯度
        grad = x_adv.grad
        
        # 第一步PGD更新
        with torch.no_grad():
            # 使用梯度符号进行更新
            grad_sign = torch.sign(grad)
            x_update = x_adv + step_size * grad_sign*loss_sign  
            # 投影到约束范围内
            x_adv_step1 = project_to_constraint(x_update, x_original, time_eps)
            # 计算第一步后的损失
            temp_data_step1 = {'inputs': x_adv_step1, 'target': future_data_normalized}
            outputs_step1 = self._forward_without_preprocessing(data=temp_data_step1, epoch=None, iter_num=None, train=False)
            loss_step1 = self.loss(outputs_step1['prediction'], outputs_step1['target'])
            
            # 记录最佳解
            if attack_mode == "untargeted":
                if loss_step1.item() > f_best:
                    f_best = loss_step1.item()
                    x_best = x_adv_step1.clone().detach()
            else: # targeted
                if loss_step1.item() < f_best:
                    f_best = loss_step1.item()
                    x_best = x_adv_step1.clone().detach()
            
            # 设置初始状态
            x_adv = x_adv_step1.clone().detach().requires_grad_(True)
            x_prev = x_original.clone().detach()  # 用于动量计算
        
        # ============= 主循环：执行自适应PGD =============
        # 用于条件判断的变量
        window_success_count = 0  # 当前窗口内损失下降的次数
        last_checkpoint_step_size = step_size
        last_checkpoint_f_best = f_best
        
        for iteration in range(1, iters):
            # 重新启用梯度计算
            x_adv = x_adv.detach().requires_grad_(True)
            # 计算当前损失和梯度
            temp_data = {'inputs': x_adv, 'target': future_data_normalized}
            outputs = self._forward_without_preprocessing(data=temp_data, epoch=None, iter_num=None, train=False)
            loss_current = self.loss(outputs['prediction'], outputs['target'])
            
            # 计算梯度
            self.model.zero_grad()
            loss_current.backward(retain_graph=True)
            grad = x_adv.grad
            
            with torch.no_grad():
                # ============= 算法步骤：计算中间变量z =============
                grad_sign = torch.sign(grad)
                x_update_z = x_adv + step_size * grad_sign*loss_sign
                z = project_to_constraint(x_update_z, x_original, time_eps)
                # ============= 算法步骤：动量更新 =============
                # x_{n+1} = P_S(x_n + α*(z - x_n) + (1-α)*(x_n - x_{n-1}))
                momentum_term = momentum * (z - x_adv) + (1 - momentum) * (x_adv - x_prev)
                x_update_momentum = x_adv + momentum_term
                x_next = project_to_constraint(x_update_momentum, x_original, time_eps)
                
                # 计算更新后的损失
                temp_data_next = {'inputs': x_next, 'target': future_data_normalized}
                outputs_next = self._forward_without_preprocessing(data=temp_data_next, epoch=None, iter_num=None, train=False)
                loss_next = self.loss(outputs_next['prediction'], outputs_next['target'])
                
                # ============= 更新最佳解 =============
                if attack_mode == "untargeted":
                    is_improved = loss_next > loss_current
                    is_better_than_best = loss_next.item() > f_best
                else: # targeted
                    is_improved = loss_next < loss_current
                    is_better_than_best = loss_next.item() < f_best
    
                if is_improved:
                    window_success_count += 1
                
                if is_better_than_best:
                    f_best = loss_next.item()
                    x_best = x_next.clone().detach()
    
                # 更新状态
                x_prev = x_adv.clone().detach()
                x_adv = x_next.clone().detach()
                
                # ============= 检查点：自适应步长调整 =============
                if iteration % checkpoint_interval == 0:
                    
                    # 条件1：损失下降次数不足
                    condition1 = window_success_count < rho * checkpoint_interval
                    
                    # 条件2：步长和最佳损失在两个连续检查点无变化
                    condition2 = (abs(step_size - last_checkpoint_step_size) < 1e-6 and 
                                abs(f_best - last_checkpoint_f_best) < 1e-6)
                    
                    if condition1 or condition2:
                        # 步长减半并回滚到最佳解
                        step_size = step_size / 2.0
                        x_adv = x_best.clone().detach()
                        x_prev = x_best.clone().detach()  # 重置动量状态
                        # 重置窗口计数
                        window_success_count = 0
                    
                    # 更新检查点记录
                    last_checkpoint_step_size = step_size
                    last_checkpoint_f_best = f_best
                    
                    # 重置窗口计数（为下一个窗口准备）
                    window_success_count = 0
        
        # 使用最佳解作为最终对抗样本
        x_final_adv = x_best
        
        # 计算最终预测
        temp_data_final = {'inputs': x_final_adv, 'target': future_data_normalized}
        outputs_final = self._forward_without_preprocessing(data=temp_data_final, epoch=None, iter_num=None, train=False)
        prediction = outputs_final['prediction']
        loss_per_sample_a = masked_mae_per_sample(outputs_final['prediction'], outputs_final['target'], null_val=np.nan)
        loss_reduction_per_sample=loss_per_sample_b-loss_per_sample_a
        # 计算扰动
        final_inputs_perturbation = x_final_adv - ori_history_data
        reduced_samples_count = (loss_reduction_per_sample > 0).sum().item()
        # 恢复原始特征的维度
        ori_history_data_original = ori_history_data[:, :, :, self.target_features]
        final_inputs_perturbation = final_inputs_perturbation[:, :, :, self.forward_features]
        # 返回结果
        return {
            'inputs': outputs_final['inputs'],
            'original_inputs': ori_history_data_original,
            'prediction': outputs_final['prediction'].detach(),
            'target': outputs_final['target'].detach(),
            'perturbation': final_inputs_perturbation,
            'count':reduced_samples_count
        }

    def mi_fgsm_attack(self, time_eps: float = 0.03, alpha: float = 0.01, 
                              iters: int = 40, data: Dict = None, momentum: float = 1.0) -> Dict:
        """
        在时域上进行MI-FGSM攻击，使用L₁归一化。
        """
        
        ori_history_data = self.to_running_device(data['inputs'])
        future_data = self.to_running_device(data['target'])
        history_data_normalized = self.scaler.transform(ori_history_data)
        future_data_normalized = self.scaler.transform(future_data)
        future_data_normalized2=future_data_normalized.clone()
        future_data_normalized = self._generate_attack_target(history_data_normalized, future_data_normalized)
        
        assert ori_history_data.dim() == 4  # 确保输入数据是四维的 [B, L, N, C]
        feature_mask = torch.zeros_like(ori_history_data)
        feature_mask[:, :, :, 0:1] = 1  # 只对第一个特征通道启用扰动
        
        attack_mode = self.attack_cfg.get("MODE", "untargeted")
        loss_sign = -1 if attack_mode == "targeted" else 1
        
        # 初始化对抗样本和动量
        X_adv = ori_history_data.clone().detach().requires_grad_(True)
        momentum_buffer = torch.zeros_like(X_adv)  # 动量缓冲区
        
        temp_data1 = {'inputs': X_adv, 'target': future_data_normalized2}
        
        # 调用模型前向传播
        outputs1 = self._forward_without_preprocessing(data=temp_data1, epoch=None, iter_num=None, train=False)
        loss_per_sample_b = masked_mae_per_sample(outputs1['prediction'], outputs1['target'], null_val=np.nan)
        for i in range(iters):
            temp_data = {'inputs': X_adv, 'target': future_data_normalized}
            # 调用模型前向传播
            outputs = self._forward_without_preprocessing(data=temp_data, epoch=None, iter_num=None, train=False)
            # 计算损失
            loss = self.loss(outputs['prediction'], outputs['target'])
            
            self.model.zero_grad()
            loss.backward()
    
            # 更新扰动 - MI-FGSM核心逻辑
            with torch.no_grad():
                # 获取当前梯度
                current_grad = X_adv.grad.data
                
                grad_data = current_grad[:, :, :, 0:1] # 只切出有数据的那个通道
                # dim=[1, 2] 表示在时间轴和节点轴上求平均
                l1_norm_data = torch.mean(torch.abs(grad_data), dim=[1, 2], keepdim=True)
                # 避免除零
                l1_norm_data = torch.where(l1_norm_data == 0, torch.ones_like(l1_norm_data), l1_norm_data)
                
                # 3. 归一化整个梯度
                # 这样 feature 0 的梯度会被缩放到一个合理的量级，而 feature 1-4 依然是 0
                normalized_grad = current_grad / l1_norm_data
                # 更新动量缓冲区 - MI-FGSM核心
                momentum_buffer = momentum * momentum_buffer + normalized_grad
                
                # 按照动量符号更新扰动
                time_update = alpha * torch.sign(momentum_buffer)*loss_sign
                
                X_adv = X_adv + time_update  # 按照动量符号更新扰动
    
                # 计算扰动
                perturbation = X_adv - ori_history_data
                perturbation = perturbation * feature_mask
                
                perturbation_clamped =torch.clamp(perturbation, min=-time_eps, max=time_eps)
                # 更新对抗样本
                X_adv = ori_history_data + perturbation_clamped
                
            # 重新启用梯度计算，为下一次迭代做好准备
            X_adv.requires_grad_(True)
        
        temp_data = {'inputs': X_adv, 'target': future_data_normalized}
        outputs = self._forward_without_preprocessing(data=temp_data, epoch=None, iter_num=None, train=False)
        prediction = outputs['prediction']
        loss_per_sample_a = masked_mae_per_sample(outputs['prediction'], outputs['target'], null_val=np.nan)
        loss_reduction_per_sample=loss_per_sample_b-loss_per_sample_a
        final_inputs_perturbation = X_adv - ori_history_data
        ori_history_data=ori_history_data[:, :, :, self.target_features]
        final_inputs_perturbation = final_inputs_perturbation[:, :, :, self.forward_features]
        reduced_samples_count = (loss_reduction_per_sample > 0).sum().item()
        # 返回字典
        return {
            'inputs': outputs['inputs'],
            'original_inputs': ori_history_data,  # 原始输入数据
            'prediction': prediction.detach(),  # 预测结果
            'target': outputs['target'].detach(),  # 目标数据
            'perturbation': final_inputs_perturbation,
            'count':reduced_samples_count
        }

    def ni_fgsm_attack(self, time_eps: float = 0.03, alpha: float = 0.01, 
                              iters: int = 40, momentum: float = 1.0, data: Dict = None) -> Dict:
        """
        在时域上进行NI-FGSM攻击
        """
        # 数据准备
        ori_history_data = self.to_running_device(data['inputs'])
        future_data = self.to_running_device(data['target'])
        history_data_normalized = self.scaler.transform(ori_history_data)
        future_data_normalized = self.scaler.transform(future_data)
        future_data_normalized2=future_data_normalized.clone()
        future_data_normalized = self._generate_attack_target(history_data_normalized, future_data_normalized)
        
        assert ori_history_data.dim() == 4  # 确保输入数据是四维的 [B, L, N, C]
        feature_mask = torch.zeros_like(ori_history_data)
        feature_mask[:, :, :, 0:1] = 1  # 只对第一个特征通道启用扰动
        
        attack_mode = self.attack_cfg.get("MODE", "untargeted")
        loss_sign = -1 if attack_mode == "targeted" else 1
        
        # NI-FGSM 初始化
        X_adv = ori_history_data.clone().detach()
        
        # NI-FGSM 核心：累积梯度（动量）
        accumulated_grad = torch.zeros_like(ori_history_data)
        temp_data1 = {'inputs': X_adv, 'target': future_data_normalized2}
        
        # 调用模型前向传播
        outputs1 = self._forward_without_preprocessing(data=temp_data1, epoch=None, iter_num=None, train=False)
        loss_per_sample_b = masked_mae_per_sample(outputs1['prediction'], outputs1['target'], null_val=np.nan)
        for i in range(iters):
            # NI-FGSM 核心步骤1：前瞻跳跃
            with torch.no_grad():
                x_nes = X_adv + alpha * momentum * accumulated_grad*feature_mask*loss_sign
            
            # 关键修改：让 x_nes 需要梯度
            x_nes = x_nes.detach().requires_grad_(True)
            
            # 在前瞻点计算前向传播
            temp_data = {'inputs': x_nes, 'target': future_data_normalized}
            outputs = self._forward_without_preprocessing(data=temp_data, epoch=None, iter_num=None, train=False)
            # 计算损失
            loss = self.loss(outputs['prediction'], outputs['target'])
            # 清空梯度
            self.model.zero_grad()
            
            # 关键修改：按照论文，计算损失对 x_nes 的梯度
            grad = torch.autograd.grad(
                outputs=loss, 
                inputs=x_nes,  # ← 对前瞻点求导，不是对 X_adv
                grad_outputs=torch.ones_like(loss),
                retain_graph=False,
                create_graph=False,
                only_inputs=True
            )[0]
            
            # NI-FGSM 核心步骤2：更新累积梯度
            with torch.no_grad():
                # 梯度归一化（按L1范数，与论文一致）
                active_grad = grad[:, :, :, 0:1] 
                grad_norm = torch.mean(torch.abs(active_grad), dim=(1,2), keepdim=True)
                normalized_grad = grad / (grad_norm + 1e-8)
                
                # 更新累积梯度：momentum * accumulated_grad + normalized_grad
                accumulated_grad = momentum * accumulated_grad + normalized_grad
                
                # NI-FGSM 核心步骤3：沿着累积梯度方向更新
                perturbation_update = alpha * torch.sign(accumulated_grad)
                X_adv = X_adv + perturbation_update*loss_sign
                
                # 计算当前总扰动并裁剪
                total_perturbation = X_adv - ori_history_data
                total_perturbation = total_perturbation * feature_mask
                perturbation_clamped = torch.clamp(total_perturbation, min=-time_eps, max=time_eps)
                # 更新对抗样本，确保在扰动范围内
                X_adv = ori_history_data + perturbation_clamped
                
            # 为下一次迭代准备梯度计算
            X_adv = X_adv.detach().requires_grad_(True)
        temp_data = {'inputs': X_adv, 'target': future_data_normalized}
        outputs = self._forward_without_preprocessing(data=temp_data, epoch=None, iter_num=None, train=False)
        prediction = outputs['prediction']
        loss_per_sample_a = masked_mae_per_sample(outputs['prediction'], outputs['target'], null_val=np.nan)
        loss_reduction_per_sample=loss_per_sample_b-loss_per_sample_a
        final_inputs_perturbation = X_adv - ori_history_data
        
        ori_history_data=ori_history_data[:, :, :, self.target_features]
        final_inputs_perturbation = final_inputs_perturbation[:, :, :, self.forward_features]
        reduced_samples_count = (loss_reduction_per_sample > 0).sum().item()
        # 返回字典
        return {
            'inputs': outputs['inputs'],
            'original_inputs': ori_history_data,  # 原始输入数据
            'prediction': prediction.detach(),  # 预测结果
            'target': outputs['target'].detach(),  # 目标数据
            'perturbation': final_inputs_perturbation,
            'count':reduced_samples_count
        }

    def fuse_attack(self, time_eps: float = 0.03, alpha: float = 0.01,
                          iters: int = 40, data: Dict = None, global_optimization: bool = False) -> Dict:
        
        ori_history_data = self.to_running_device(data['inputs'])
        future_data = self.to_running_device(data['target'])
        history_data_normalized = self.scaler.transform(ori_history_data)
        future_data_normalized = self.scaler.transform(future_data)
        future_data_normalized2=future_data_normalized.clone()
        future_data_normalized = self._generate_attack_target(history_data_normalized, future_data_normalized)
    
        assert ori_history_data.dim() == 4  # 确保输入数据是四维的 [B, L, N, C]
        feature_mask = torch.zeros_like(ori_history_data)
        feature_mask[:, :, :, 0:1] = 1  # 只对第一个特征通道启用扰动
        L = ori_history_data.shape[1]
        B, L, N, C = ori_history_data.shape
        # 初始化随机扰动
        delta = torch.zeros_like(ori_history_data)
        delta = delta * feature_mask  # 只对特定特征通道添加扰动
        delta.requires_grad_(True)
        X_adv = ori_history_data.clone().detach().requires_grad_(True)
        # 获取频率数量
        X_original = self._rfft_time(ori_history_data, time_dim=1)
        
        attack_mode = self.attack_cfg.get("MODE", "untargeted")
        loss_sign = -1 if attack_mode == "targeted" else 1
        
        temp_data1 = {'inputs': X_adv, 'target': future_data_normalized2}
        
        # 调用模型前向传播
        outputs1 = self._forward_without_preprocessing(data=temp_data1, epoch=None, iter_num=None, train=False)
        loss_per_sample_b = masked_mae_per_sample(outputs1['prediction'], outputs1['target'], null_val=np.nan)

        freq_dim = L // 2 + 1
        adaptive_weights_param = torch.nn.Parameter(
            torch.zeros(1, freq_dim, N, 1, device=ori_history_data.device),
            requires_grad=True
        )
        adaptive_optimizer = torch.optim.Adam([adaptive_weights_param], lr=0.01)
        
        def get_adaptive_mask(weights_param):
            """
            使用softmax将权重转换为概率分布，然后乘以freq_dim
            这样权重总和为freq_dim，平均值为1.0
            """
            
            # 应用softmax得到概率分布
            prob_weights = torch.softmax(weights_param, dim=1)
            # 乘以freq_dim使得权重总和为freq_dim，平均值为1.0
            return prob_weights * freq_dim
        
        X_adv.requires_grad_(True)
        
        for i in range(iters):
            temp_data = {'inputs': X_adv, 'target': future_data_normalized}
            adaptive_mask = get_adaptive_mask(adaptive_weights_param)
            # 调用模型前向传播
            outputs = self._forward_without_preprocessing(data=temp_data, epoch=None, iter_num=None, train=False)
            # 计算损失
            loss = self.loss(outputs['prediction'], outputs['target'])
            
            # 清空梯度并执行反向传播
            self.model.zero_grad()
            loss.backward()
            # 更新扰动
            with torch.no_grad():
                grad=X_adv.grad # 获取梯度的符号
                # 将时域更新步长转换到频域
                direction = torch.sign(grad)
                direction_freq = self._rfft_time(direction, time_dim=1)
                
                temporal_weights = self.generate_robin_hood_weights(
                    X_adv,X_adv.grad,ori_history_data,0.2,i)

            update = self._irfft_time(direction_freq * adaptive_mask, n_time=L, time_dim=1) * alpha*temporal_weights
            X_adv_differentiable = X_adv + update*loss_sign # 此时 X_adv_diff 链接着 weight_param  
            perturbation = X_adv_differentiable - ori_history_data
            perturbation = perturbation * feature_mask
            # 限制扰动的最大幅度
            perturbation_clamped = torch.clamp(perturbation, min=-time_eps, max=time_eps)
                
            # 更新对抗样本
            X_adv_differentiable = ori_history_data + perturbation_clamped
            
            delta = perturbation_clamped    
            if (i + 1) % 1 == 0:
                temp_data = {'inputs': X_adv_differentiable, 'target': future_data_normalized}
                outputs = self._forward_without_preprocessing(data=temp_data, epoch=None, iter_num=None, train=False)
                loss_weight = -self.loss(outputs['prediction'], outputs['target'])*loss_sign
                
                # 清空权重梯度并执行反向传播
                self.model.zero_grad()
                if adaptive_weights_param.grad is not None:
                    adaptive_weights_param.grad.zero_()
                    
                loss_weight.backward()
                
                # 更新自适应权重参数
                adaptive_optimizer.step()
                # 5. 更新 X_adv 本身
                adaptive_mask = get_adaptive_mask(adaptive_weights_param)
            
            X_adv = X_adv_differentiable.detach().requires_grad_(True)
            
            current_adaptive_mask = get_adaptive_mask(adaptive_weights_param)
                    
            
                
        final_adaptive_mask = get_adaptive_mask(adaptive_weights_param)
    
        temp_data = {'inputs': X_adv, 'target': future_data_normalized}
        outputs = self._forward_without_preprocessing(data=temp_data, epoch=None, iter_num=None, train=False)
        loss_per_sample_a = masked_mae_per_sample(outputs['prediction'], outputs['target'], null_val=np.nan)
        loss_reduction_per_sample=loss_per_sample_b-loss_per_sample_a
        prediction = outputs['prediction']
        final_inputs_perturbation = X_adv - ori_history_data
    
        if final_adaptive_mask.shape[1] >= 47:
            weights_np = final_adaptive_mask[0, :47, 0, 0].detach().cpu().numpy()
            print(f"  前47个频率权重: {[f'{w:.4f}' for w in weights_np]}")
        
        loss = self.loss(outputs['prediction'], outputs['target'])
        ori_history_data = ori_history_data[:, :, :, self.target_features]
        final_inputs_perturbation = final_inputs_perturbation[:, :, :, self.forward_features]
        
        reduced_samples_count = (loss_reduction_per_sample > 0).sum().item()
        # 返回字典
        return {
            'inputs': outputs['inputs'],
            'original_inputs': ori_history_data,  # 原始输入数据
            'prediction': prediction.detach(),  # 预测结果
            'target': outputs['target'].detach(),  # 目标数据
            'perturbation': final_inputs_perturbation,
            'count': reduced_samples_count,
        }

    def forward(self, data: Dict, epoch: Optional[int] = None, iter_num: Optional[int] = None, train: bool = True, **kwargs) -> Dict:
        """
        Performs the forward pass for training, validation, and testing. 

        Args:
            data (Dict): A dictionary containing 'target' (future data) and 'inputs' (history data) (normalized by self.scaler).
            epoch (int, optional): Current epoch number. Defaults to None.
            iter_num (int, optional): Current iteration number. Defaults to None.
            train (bool, optional): Indicates whether the forward pass is for training. Defaults to True.

        Returns:
            Dict: A dictionary containing the keys:
                  - 'inputs': Selected input features.
                  - 'prediction': Model predictions.
                  - 'target': Selected target features.

        Raises:
            AssertionError: If the shape of the model output does not match [B, L, N].
        """

        data = self.preprocessing(data)

        # Preprocess input data
        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)  # Shape: [B, L, N, C]
        future_data = self.to_running_device(future_data)    # Shape: [B, L, N, C]
        batch_size, length, num_nodes, _ = future_data.shape

        # Select input features
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            # For non-training phases, use only temporal features
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # Forward pass through the model
        model_return = self.model(history_data=history_data, future_data=future_data_4_dec,
                                  batch_seen=iter_num, epoch=epoch, train=train)

        # Parse model return
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        # Ensure the output shape is correct
        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            "The shape of the output is incorrect. Ensure it matches [B, L, N, C]."

        model_return = self.postprocessing(model_return)

        return model_return
    
    def _forward_without_preprocessing(self, data: Dict, epoch: Optional[int] = None, iter_num: Optional[int] = None, train: bool = True, **kwargs) -> Dict:
        """
        Performs the forward pass for training, validation, and testing. 

        Args:
            data (Dict): A dictionary containing 'target' (future data) and 'inputs' (history data) (normalized by self.scaler).
            epoch (int, optional): Current epoch number. Defaults to None.
            iter_num (int, optional): Current iteration number. Defaults to None.
            train (bool, optional): Indicates whether the forward pass is for training. Defaults to True.

        Returns:
            Dict: A dictionary containing the keys:
                  - 'inputs': Selected input features.
                  - 'prediction': Model predictions.
                  - 'target': Selected target features.

        Raises:
            AssertionError: If the shape of the model output does not match [B, L, N].
        """


        # Preprocess input data
        future_data, history_data = data['target'], data['inputs']
        history_data = self.to_running_device(history_data)  # Shape: [B, L, N, C]
        future_data = self.to_running_device(future_data)    # Shape: [B, L, N, C]
        batch_size, length, num_nodes, _ = future_data.shape

        # Select input features
        history_data = self.select_input_features(history_data)
        future_data_4_dec = self.select_input_features(future_data)

        if not train:
            # For non-training phases, use only temporal features
            future_data_4_dec[..., 0] = torch.empty_like(future_data_4_dec[..., 0])

        # Forward pass through the model
        model_return = self.model(history_data=history_data, future_data=future_data_4_dec,
                                  batch_seen=iter_num, epoch=epoch, train=train)

        # Parse model return
        if isinstance(model_return, torch.Tensor):
            model_return = {'prediction': model_return}
        if 'inputs' not in model_return:
            model_return['inputs'] = self.select_target_features(history_data)
        if 'target' not in model_return:
            model_return['target'] = self.select_target_features(future_data)

        # Ensure the output shape is correct
        assert list(model_return['prediction'].shape)[:3] == [batch_size, length, num_nodes], \
            "The shape of the output is incorrect. Ensure it matches [B, L, N, C]."

        model_return = self.postprocessing(model_return)

        return model_return
    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Selects input features based on the forward features specified in the configuration.

        Args:
            data (torch.Tensor): Input history data with shape [B, L, N, C1].

        Returns:
            torch.Tensor: Data with selected features with shape [B, L, N, C2].
        """

        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Selects target features based on the target features specified in the configuration.

        Args:
            data (torch.Tensor): Model prediction data with shape [B, L, N, C1].

        Returns:
            torch.Tensor: Data with selected target features and shape [B, L, N, C2].
        """

        data = data[:, :, :, self.target_features]
        return data

    def select_target_time_series(self, data: torch.Tensor) -> torch.Tensor:
        """
        Select target time series based on the target time series specified in the configuration.

        Args:
            data (torch.Tensor): Model prediction data with shape [B, L, N1, C].

        Returns:
            torch.Tensor: Data with selected target time series and shape [B, L, N2, C].
        """

        data = data[:, :, self.target_time_series, :]
        return data
