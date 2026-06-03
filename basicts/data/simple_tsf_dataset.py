import inspect
import json
import logging
from typing import List

import numpy as np

from .base_dataset import BaseDataset


class TimeSeriesForecastingDataset(BaseDataset):
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
    """

    def __init__(self, dataset_name: str, train_val_test_ratio: List[float], mode: str, input_len: int, output_len: int, \
        overlap: bool = False, logger: logging.Logger = None,**kwargs) -> None:
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

        Raises:
            AssertionError: If `mode` is not one of ['train', 'valid', 'test'].
        """
        assert mode in ['train', 'valid', 'test'], f"Invalid mode: {mode}. Must be one of ['train', 'valid', 'test']."
        super().__init__(dataset_name, train_val_test_ratio, mode, input_len, output_len, overlap)
        self.logger = logger

        self.data_file_path = f'datasets/{dataset_name}/data.dat'
        self.description_file_path = f'datasets/{dataset_name}/desc.json'
        self.description = self._load_description()
        self.data = self._load_data()
        # --- 自动化开关逻辑 ---
        # 从 kwargs 获取参数，默认不开启 NPZ 加载（白盒模式）
        self.use_npz = kwargs.get('use_npz', False)
        self.npz_path = kwargs.get('npz_path', None)
        if self.use_npz and self.npz_path:
            # 关键：直接读取到内存中，变成真正的 numpy array，而不是 NpzFile 对象
            with np.load(self.npz_path) as loaded:
                self.adv_inputs = loaded['inputs'][:]  # [:] 强制加载进内存
                self.adv_target = loaded['target'][:]
            
            # 预处理：确保维度是 3D [Sample, L, N] -> [Sample, L, N, 1]
            if self.adv_inputs.ndim == 3:
                self.adv_inputs = self.adv_inputs[..., np.newaxis]
            if self.adv_target.ndim == 3:
                self.adv_target = self.adv_target[..., np.newaxis]
        else:
            self.adv_inputs = None
            self.adv_target = None


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

        # Automatically configure the overlap parameter
        minimal_len = self.input_len + self.output_len
        if minimal_len > {'train': train_len, 'valid': valid_len, 'test': test_len}[self.mode]:
            self.overlap = True  # Enable overlap when the train, validation, or test set is too short
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

    def __getitem__(self, index: int) -> dict:
        """
        Retrieves a sample from the dataset at the specified index, considering both the input and output lengths.

        Args:
            index (int): The index of the desired sample in the dataset.

        Returns:
            dict: A dictionary containing 'inputs' and 'target', where both are slices of the dataset corresponding to
                  the historical input data and future prediction data, respectively.
        """
        # 1. 计算索引 (保持不变)
        s_begin = index
        s_end = index + self.input_len
        r_begin = s_end
        r_end = r_begin + self.output_len
    
        # 2. 如果是黑盒模式
        if self.adv_inputs is not None:
            # 直接从内存数组取，速度极快
            history_data = self.adv_inputs[index]
            future_data = self.adv_target[index]
    
            # 自动补全逻辑优化：只在必要时拼接
            num_adv_channels = history_data.shape[-1]
            num_total_channels = self.description['shape'][1] # 假设 shape 是 [Total_L, N, C]
    
            if num_adv_channels < num_total_channels:
                # 仅对缺少的通道使用 memmap 访问
                raw_full_in = self.data[s_begin:s_end, :, num_adv_channels:]
                raw_full_out = self.data[r_begin:r_end, :, num_adv_channels:]
                history_data = np.concatenate([history_data, raw_full_in], axis=-1)
                future_data = np.concatenate([future_data, raw_full_out], axis=-1)
        else:
            # 白盒模式：直接切片
            history_data = self.data[s_begin:s_end]
            future_data = self.data[r_begin:r_end]
        return {'inputs': history_data, 'target': future_data}

    def __len__(self) -> int:
        """
        Calculates the total number of samples available in the dataset, adjusted for the lengths of input and output sequences.

        Returns:
            int: The number of valid samples that can be drawn from the dataset, based on the configurations of input and output lengths.
        """
        return len(self.data) - self.input_len - self.output_len + 1
