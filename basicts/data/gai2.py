import inspect
import json
import logging
from typing import List

import numpy as np
import torch

from .base_dataset import BaseDataset


class TimeSeriesForecastingDatasetgai2(BaseDataset):
    """
    时间序列预测数据集类，支持频域对抗攻击功能。
    
    此类扩展了基础数据集类，支持在数据加载过程中进行频域扰动，
    允许直接在数据流水线中生成对抗样本。
    
    属性:
        data_file_path (str): 时间序列数据文件的路径。
        description_file_path (str): 数据集描述JSON文件的路径。
        data (np.ndarray): 加载的时间序列数据数组，根据指定模式分割。
        description (dict): 数据集的元数据，如形状和其他属性。
        attack_params (dict): 控制频域攻击行为的参数。
    """

    def __init__(self, dataset_name: str, train_val_test_ratio: List[float], mode: str, input_len: int, output_len: int, \
        overlap: bool = False, logger: logging.Logger = None, 
        attack_mode: str = None, attack_freq_band: str = 'low', 
        attack_epsilon: float = 0.1, attack_prob: float = 0.0) -> None:
        """
        初始化时间序列预测数据集，可选择启用频域攻击功能。

        参数:
            dataset_name (str): 数据集的名称。
            train_val_test_ratio (List[float]): 将数据集分割为训练、验证和测试集的比率。
            mode (str): 数据集的操作模式。有效值为 'train', 'valid' 或 'test'。
            input_len (int): 输入序列的长度（历史点数）。
            output_len (int): 输出序列的长度（要预测的未来点数）。
            overlap (bool): 确定训练/验证/测试分割是否应重叠的标志。
            logger (logging.Logger): 用于记录消息的日志记录器实例。
            attack_mode (str): 要应用的攻击类型。None表示无攻击，'random'表示随机扰动。
            attack_freq_band (str): 要攻击的频带。选项: 'low', 'medium', 'high'。
            attack_epsilon (float): 扰动的强度。
            attack_prob (float): 对样本应用攻击的概率。

        异常:
            AssertionError: 如果`mode`不是['train', 'valid', 'test']中的一个。
        """
        assert mode in ['train', 'valid', 'test'], f"无效模式: {mode}. 必须是 ['train', 'valid', 'test'] 中的一个。"
        super().__init__(dataset_name, train_val_test_ratio, mode, input_len, output_len, overlap)
        self.logger = logger

        self.data_file_path = f'datasets/{dataset_name}/data.dat'
        self.description_file_path = f'datasets/{dataset_name}/desc.json'
        self.description = self._load_description()
        self.data = self._load_data()
        
        # 频域攻击参数 - 仅在测试模式下启用
        self.attack_mode = attack_mode if mode == 'test' else None
        self.attack_freq_band = attack_freq_band
        self.attack_epsilon = attack_epsilon
        self.attack_prob = attack_prob if mode == 'test' else 0.0  # 非测试模式下概率为0
        
        # 如果启用了攻击，预计算频率掩码
        if self.attack_mode is not None:
            self.freq_mask = self._create_frequency_mask()
            
            if self.logger is not None:
                self.logger.info(f"初始化数据集并启用频域攻击: 模式={attack_mode}, " +
                                f"频带={attack_freq_band}, 强度={attack_epsilon}, 概率={attack_prob}")
            elif mode == 'test':
                print(f"测试模式启用频域攻击: 模式={attack_mode}, 频带={attack_freq_band}, 强度={attack_epsilon}, 概率={attack_prob}")

    def _create_frequency_mask(self) -> np.ndarray:
        """
        为指定的频带创建频率掩码。
        
        返回:
            np.ndarray: 布尔掩码，其中True表示要攻击的频率。
        """
        n_freq = self.input_len
        
        print(f"输入长度 n_freq: {n_freq}")
        
        if n_freq % 2 == 0:
            norm_freq = np.linspace(0, 0.5, n_freq // 2 + 1)
        else:
            norm_freq = np.linspace(0, 0.5, (n_freq + 1) // 2)
        
        print(f"归一化频率 norm_freq: {norm_freq}")
        print(f"归一化频率 norm_freq 的长度: {len(norm_freq)}")
        
        full_norm_freq = np.zeros(n_freq)
        full_norm_freq[:len(norm_freq)] = norm_freq
        
        print(f"完整归一化频率 full_norm_freq: {full_norm_freq}")
        print(f"完整归一化频率 full_norm_freq 的长度: {len(full_norm_freq)}")
        
        if n_freq % 2 == 0:
            # 对于偶数长度，需要确保 norm_freq[1:][::-1] 的长度与 full_norm_freq[len(norm_freq):] 一致
            full_norm_freq[len(norm_freq):] = norm_freq[1:-1][::-1]
        else:
            full_norm_freq[len(norm_freq):] = norm_freq[1:][::-1]
        
        print(f"完整归一化频率 full_norm_freq (扩展后): {full_norm_freq}")
        print(f"完整归一化频率 full_norm_freq (扩展后) 的长度: {len(full_norm_freq)}")
        
        # 根据指定的频带创建掩码
        if self.attack_freq_band == 'low':
            mask = full_norm_freq <= 0.1  # 攻击最低10%的频率
        elif self.attack_freq_band == 'medium':
            mask = (full_norm_freq > 0.1) & (full_norm_freq <= 0.3)  # 攻击10%-30%的频率
        elif self.attack_freq_band == 'high':
            mask = full_norm_freq > 0.3  # 攻击高于30%的频率
        else:
            mask = np.ones(n_freq, dtype=bool)  # 攻击所有频率
        
        print(f"生成的掩码 mask: {mask}")
        print(f"掩码 mask 的长度: {len(mask)}")
        
        return mask

    def _apply_frequency_attack(self, signal: np.ndarray) -> np.ndarray:
        """
        对输入信号应用频域扰动。
        
        参数:
            signal (np.ndarray): 输入时间序列信号。
            
        返回:
            np.ndarray: 扰动后的信号。
        """
        print(f"输入信号 signal 的形状: {signal.shape}")
        
        signal_tensor = torch.from_numpy(signal).float()
        print(f"信号张量 signal_tensor 的形状: {signal_tensor.shape}")
        
        # 执行离散傅里叶变换(DFT)
        dft_output = torch.fft.fft(signal_tensor, dim=0)
        print(f"DFT 输出 dft_output 的形状: {dft_output.shape}")
        
        # 获取幅度和相位
        amp = torch.abs(dft_output)
        phase = torch.angle(dft_output)
        print(f"幅度 amp 的形状: {amp.shape}")
        print(f"相位 phase 的形状: {phase.shape}")
        
        if self.attack_mode == 'random':
            # 生成随机扰动
            perturbation = torch.randn_like(amp).real
            print(f"扰动 perturbation 的形状: {perturbation.shape}")
            
            # 应用频率掩码，只扰动指定频带
            freq_mask_tensor = torch.from_numpy(self.freq_mask).float().unsqueeze(-1).unsqueeze(-1)
            print(f"频率掩码 freq_mask_tensor 的形状: {freq_mask_tensor.shape}")
            
            # 只对第一个维度（时间步）应用扰动
            amp_perturbed = amp + self.attack_epsilon * perturbation * freq_mask_tensor
            print(f"扰动后的幅度 amp_perturbed 的形状: {amp_perturbed.shape}")
            
            # 使用扰动后的幅度和原始相位重建信号
            dft_perturbed = amp_perturbed * torch.exp(1j * phase)
            print(f"扰动后的 DFT dft_perturbed 的形状: {dft_perturbed.shape}")
            
            # 执行逆离散傅里叶变换(IDFT)
            perturbed_signal = torch.fft.ifft(dft_perturbed, dim=0).real
            print(f"扰动后的信号 perturbed_signal 的形状: {perturbed_signal.shape}")
            
            return perturbed_signal.numpy()
        
        else:
            return signal  # 如果攻击模式不被识别，返回原始信号
    def _load_description(self) -> dict:
        """
        从JSON文件加载数据集的描述。

        返回:
            dict: 包含数据集元数据的字典，如其形状和其他属性。

        异常:
            FileNotFoundError: 如果找不到描述文件。
            json.JSONDecodeError: 如果解码JSON数据时出错。
        """

        try:
            with open(self.description_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'找不到描述文件: {self.description_file_path}') from e
        except json.JSONDecodeError as e:
            raise ValueError(f'解码JSON文件时出错: {self.description_file_path}') from e

    def _load_data(self) -> np.ndarray:
        """
        从文件加载时间序列数据，并根据所选模式进行分割。

        返回:
            np.ndarray: 指定模式（训练、验证或测试）的数据数组。

        异常:
            ValueError: 如果加载数据文件有问题或数据形状不符合预期。
        """

        try:
            data = np.memmap(self.data_file_path, dtype='float32', mode='r', shape=tuple(self.description['shape']))
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f'加载数据文件时出错: {self.data_file_path}') from e

        total_len = len(data)
        valid_len = int(total_len * self.train_val_test_ratio[1])
        test_len = int(total_len * self.train_val_test_ratio[2])
        train_len = total_len - valid_len - test_len

        # 自动配置重叠参数
        minimal_len = self.input_len + self.output_len
        if minimal_len > {'train': train_len, 'valid': valid_len, 'test': test_len}[self.mode]:
            self.overlap = True  # 当训练、验证或测试集太短时启用重叠
            current_frame = inspect.currentframe()
            file_name = inspect.getfile(current_frame)
            line_number = current_frame.f_lineno - 7
            dataset = {'train': '训练', 'valid': '验证', 'test': '测试'}[self.mode]
            if self.logger is not None:
                self.logger.info(f'{dataset}数据集太短，启用重叠。详细信息见{file_name}第{line_number}行。')
            else:
                print(f'{dataset}数据集太短，启用重叠。详细信息见{file_name}第{line_number}行。')

        if self.mode == 'train':
            offset = self.output_len if self.overlap else 0
            return data[:train_len + offset].copy()
        elif self.mode == 'valid':
            offset_left = self.input_len - 1 if self.overlap else 0
            offset_right = self.output_len if self.overlap else 0
            return data[train_len - offset_left : train_len + valid_len + offset_right].copy()
        else:  # self.mode == 'test'
            offset = self.input_len - 1 if self.overlap else 0
            return data[train_len + valid_len - offset:].copy()

    def __getitem__(self, index: int) -> dict:
        """
        从数据集中检索指定索引处的样本，可选择应用频域攻击。

        参数:
            index (int): 数据集中所需样本的索引。

        返回:
            dict: 包含'inputs'和'target'的字典，其中'inputs'可能会被扰动（如果启用攻击并由概率触发）。
        """
        history_data = self.data[index:index + self.input_len]
        future_data = self.data[index + self.input_len:index + self.input_len + self.output_len]
        
        # 仅在测试模式下且有概率时应用频域攻击
        if (self.attack_mode is not None and 
            self.mode == 'test' and 
            np.random.rand() < self.attack_prob):
            try:
                history_data = self._apply_frequency_attack(history_data)
            except Exception as e:
                if self.logger is not None:
                    self.logger.error(f"应用频域攻击时出错: {e}")
                # 如果攻击失败，返回原始数据
                pass
        
        return {'inputs': history_data, 'target': future_data}

    def __len__(self) -> int:
        """
        计算数据集中可用的样本总数。

        返回:
            int: 可以从数据集中提取的有效样本数，基于输入和输出长度的配置。
        """
        return len(self.data) - self.input_len - self.output_len + 1