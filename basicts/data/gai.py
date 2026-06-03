import inspect
import json
import logging
from typing import List

import numpy as np
import torch
import torch.fft

from .base_dataset import BaseDataset


class TimeSeriesForecastingDatasetgai(BaseDataset):
    """
    A dataset class for time series forecasting problems, handling the loading, parsing, and partitioning
    of time series data into training, validation, and testing sets based on provided ratios.
    
    This class supports configurations where sequences may or may not overlap, accommodating scenarios
    where time series data is drawn from continuous periods or distinct episodes, affecting how
    the data is split into batches for model training or evaluation.
    
    Attributes:
        data_file_path (str): Path to the file containing the time series data.
        description_file_path (str): Path to the JSON file containing the description of the dataset.
        data (np.ndarray): The loaded time series data array, split according to the specified mode.
        description (dict): Metadata about the dataset, such as shape and other properties.
        test_perturbation (torch.Tensor): Precomputed perturbation for the test set.
    """

    def __init__(self, dataset_name: str, train_val_test_ratio: List[float], mode: str, input_len: int, output_len: int, \
        overlap: bool = False, logger: logging.Logger = None, freq_range: List[int] = None, noise_level: float = 0.1) -> None:
        """
        Initializes the TimeSeriesForecastingDataset by setting up paths, loading data, and 
        preparing it according to the specified configurations.

        Args:
            dataset_name (str): The name of the dataset.
            train_val_test_ratio (List[float]): Ratios for splitting the dataset into train, validation, and test sets.
                Each value should be a float between 0 and 1, and their sum should ideally be 1.
            mode (str): The operation mode of the dataset. Valid values are 'train', 'valid', or 'test'.
            input_len (int): The length of the input sequence (number of historical points).
            output_len (int): The length of the output sequence (number of future points to predict).
            overlap (bool): Flag to determine if training/validation/test splits should overlap. 
                Defaults to False for strictly non-overlapping periods. Set to True to allow overlap.
            logger (logging.Logger): logger.
            freq_range (List[int]): Frequency range for adding perturbation. Only used in 'test' mode.
            noise_level (float): Noise level for perturbation. Defaults to 0.1.
        """
        assert mode in ['train', 'valid', 'test'], f"Invalid mode: {mode}. Must be one of ['train', 'valid', 'test']."
        super().__init__(dataset_name, train_val_test_ratio, mode, input_len, output_len, overlap)
        self.logger = logger

        self.data_file_path = f'datasets/{dataset_name}/data.dat'
        self.description_file_path = f'datasets/{dataset_name}/desc.json'
        self.description = self._load_description()
        self.data = self._load_data()
        self.freq_range = freq_range
        if mode == 'test' and freq_range is not None:
            self.test_perturbation = self._generate_test_perturbation(freq_range, noise_level)
        else:
            self.test_perturbation = None

    def _load_description(self) -> dict:
        """
        Loads the description of the dataset from a JSON file.

        Returns:
            dict: A dictionary containing metadata about the dataset, such as its shape and other properties.

        Raises:
            FileNotFoundError: If the description file is not found.
            json.JSONDecodeError: If there is an error decoding the JSON data.
        """
        try:
            with open(self.description_file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Description file not found: {self.description_file_path}') from e
        except json.JSONDecodeError as e:
            raise ValueError(f'Error decoding JSON file: {self.description_file_path}') from e

    def _load_data(self) -> np.ndarray:
        """
        Loads the time series data from a file and splits it according to the selected mode.

        Returns:
            np.ndarray: The data array for the specified mode (train, validation, or test).

        Raises:
            ValueError: If there is an issue with loading the data file or if the data shape is not as expected.
        """
        try:
            data = np.memmap(self.data_file_path, dtype='float32', mode='r', shape=tuple(self.description['shape']))
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(f'Error loading data file: {self.data_file_path}') from e

        total_len = len(data)
        valid_len = int(total_len * self.train_val_test_ratio[1])
        test_len = int(total_len * self.train_val_test_ratio[2])
        train_len = total_len - valid_len - test_len

        minimal_len = self.input_len + self.output_len
        if minimal_len > {'train': train_len, 'valid': valid_len, 'test': test_len}[self.mode]:
            self.overlap = True
            current_frame = inspect.currentframe()
            file_name = inspect.getfile(current_frame)
            line_number = current_frame.f_lineno - 7
            dataset = {'train': 'Training', 'valid': 'Validation', 'test': 'Test'}[self.mode]
            if self.logger is not None:
                self.logger.info(f'{dataset} dataset is too short, enabling overlap. See details in {file_name} at line {line_number}.')
            else:
                print(f'{dataset} dataset is too short, enabling overlap. See details in {file_name} at line {line_number}.')

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

    def _generate_test_perturbation(self, freq_range: List[float], noise_level: float) -> torch.Tensor:
        """
        为测试集生成指定频率幅值范围内的固定扰动。
    
        Args:
            freq_range (List[float]): 添加扰动的频率幅值范围。
            noise_level (float): 扰动的噪声水平。
    
        Returns:
            torch.Tensor: 预计算的扰动张量。
        """
        # 将数据转换为张量
        data_tensor = torch.tensor(self.data, dtype=torch.float32)
        # 对数据进行傅里叶变换
        data_fft = torch.fft.rfft(data_tensor, dim=1)
        # 计算频率分量的幅值
        freq_magnitude = torch.abs(data_fft)
        
        # 打印频率幅值的范围
        min_magnitude = torch.min(freq_magnitude).item()
        max_magnitude = torch.max(freq_magnitude).item()
        print(f"Frequency magnitude range: min={min_magnitude}, max={max_magnitude}")
        print(1111111111111111)
        # 找到频率幅值在指定范围内的索引
        freq_indices = (freq_magnitude >= freq_range[0]) & (freq_magnitude <= freq_range[1])
        
        # 仅在这些索引上生成扰动
        freq_noise = torch.randn_like(data_fft) * noise_level
        # 将指定范围之外的扰动置为0
        freq_noise[~freq_indices] = 0  # 将指定范围之外的扰动置为0
        
        return freq_noise

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a sample from the dataset at the specified index, considering both the input and output lengths.
        If in 'test' mode, adds a fixed perturbation to the historical data in the frequency domain.
        
        Args:
            index (int): The index of the desired sample in the dataset.
        
        Returns:
            dict: A dictionary containing 'inputs' and 'target', where both are slices of the dataset corresponding to
                  the historical input data and future prediction data, respectively.
        """
        history_data = self.data[index:index + self.input_len]
        future_data = self.data[index + self.input_len:index + self.input_len + self.output_len]
        
        #print(f"history_data shape before perturbation: {history_data.shape}")
        #print(f"future_data shape: {future_data.shape}")
        
        if self.mode == 'test' and self.test_perturbation is not None:
            history_data_tensor = torch.tensor(history_data, dtype=torch.float32)
            history_data_fft = torch.fft.rfft(history_data_tensor, dim=1)
            #print(f"history_data_fft shape: {history_data_fft.shape}")
        
            # Add perturbation to the specified frequency amplitude range
            history_data_fft += self.test_perturbation[index]
        
            history_data_perturbed = torch.fft.irfft(history_data_fft, n=history_data.shape[1], dim=1)
            #print(f"history_data_perturbed shape: {history_data_perturbed.shape}")
        
            history_data = history_data_perturbed.numpy()
            #print(f"history_data shape after perturbation: {history_data.shape}")
        
        return {'inputs': history_data, 'target': future_data}

    def __len__(self) -> int:
        """
        Calculates the total number of samples available in the dataset, adjusted for the lengths of input and output sequences.

        Returns:
            int: The number of valid samples that can be drawn from the dataset, based on the configurations of input and output lengths.
        """
        return len(self.data) - self.input_len - self.output_len + 1
    