import os
import numpy as np
import nibabel as nib
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset


class SingleDataset(BaseDataset):
    """ 这个数据集类用于加载单个 CT NIfTI 文件，并转换成适合模型处理的 NumPy 格式 """

    def __init__(self, opt):
        """ 初始化数据集类 """
        super().__init__(opt)
        self.root = opt.dataroot  # CT NIfTI 文件的路径
        self.image_paths = sorted(make_dataset(self.root))  # 获取所有 NIfTI 文件路径
        self.preprocess_gamma = opt.preprocess_gamma  # 预处理参数（如果需要）

    def __getitem__(self, index):
        """ 读取并处理 CT NIfTI 文件 """
        path = self.image_paths[index]
        nii_img = nib.load(path)  # 读取 NIfTI 文件
        image_array = nii_img.get_fdata()  # 转换为 NumPy 数组

        # 归一化到 [0, 1]（如果数据不是这个范围）
        image_array = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array) + 1e-5)

        # 转换为模型需要的格式（PyTorch 默认格式是 C×H×W）
        image_array = np.expand_dims(image_array, axis=0).astype(np.float32)  # 添加通道维度 (1, H, W, D)

        return {'A': image_array, 'A_paths': path}  # 这里的 'A' 代表输入数据

    def __len__(self):
        """ 返回数据集的长度 """
        return len(self.image_paths)
