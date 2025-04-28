import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as tt
from os import listdir, path
from os.path import join
import numpy as np
import logging
from data.base_dataset import BaseDataset
import os


class CTtoPETDataset(BaseDataset):
    def __init__(self, opt):
        self.mode = opt.mode
        self.preprocess_gamma = opt.preprocess_gamma
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        BaseDataset.__init__(self, opt)

        if self.mode == 'test':
            self.CT_dir = join(opt.dataroot, 'temp_folder')
            self.PET_dir = join(opt.dataroot, 'temp_folder')
        else:
            self.CT_dir = join(opt.dataroot, 'trainA')
            self.PET_dir = join(opt.dataroot, 'trainB')

        if not all([self._check_dir(self.CT_dir), self._check_dir(self.PET_dir)]):
            raise FileNotFoundError(f"CT or PET data directory not found: {self.CT_dir}, {self.PET_dir}")

        self.ids = [file for file in listdir(self.CT_dir) if file.endswith('.npy') and not file.startswith('.')]
        self.ids.sort()
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def _check_dir(self, directory):
        return os.path.isdir(directory)

    @classmethod
    def preprocessCT(cls, im, minn=-900.0, maxx=200.0, noise_std=0):
        if len(im.shape) == 3:
            img_np = np.array(im)
        else:
            raise ValueError(f"Unexpected CT shape: {im.shape}")

        if noise_std > 0:
            img_np += noise_std * np.random.randn(*img_np.shape)

        img_np = np.clip(img_np, minn, maxx)
        img_np = (img_np - minn) / (maxx - minn)
        print(f"Preprocessed CT shape: {img_np.shape}")  # 添加调试信息，打印预处理后的 CT 形状
        return img_np

    @classmethod
    def preprocessPET_gamma(cls, img, gamma=1 / 2, maxx=7, noise_std=0, scale=1.):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)  # 添加通道维度 (1, H, W)
        elif len(img.shape) != 3:
            raise ValueError(f"Unexpected PET shape: {img.shape}")

        img = img / scale
        if noise_std > 0:
            img += noise_std * np.random.randn(*img.shape)

        img = np.clip(img, 0, maxx)
        img = img / maxx
        img = np.power(img, gamma)
        print(f"Preprocessed PET shape: {img.shape}")  # 添加调试信息，打印预处理后的 PET 形状
        return img

    @classmethod
    def edge_zero(cls, img):
        img[:, 0, :] = 0
        img[:, -1, :] = 0
        img[:, :, 0] = 0
        img[:, :, -1] = 0
        print(f"PET after edge_zero shape: {img.shape}")  # 添加调试信息，打印边缘置零后的 PET 形状
        return img

    def transform(self, CT, PET):
        affine_params = tt.RandomAffine.get_params(
            degrees=(-45, 45),
            translate=(0.10, 0.10),
            scale_ranges=(0.85, 1.15),
            shears=(-7, 7),
            img_size=CT.shape[-2:]
        )

        CT = TF.affine(CT, *affine_params, interpolation=TF.InterpolationMode.BILINEAR)
        PET = TF.affine(PET, *affine_params, interpolation=TF.InterpolationMode.BILINEAR)

        print(f"Transformed CT shape: {CT.shape}")  # 添加调试信息，打印变换后的 CT 形状
        print(f"Transformed PET shape: {PET.shape}")  # 添加调试信息，打印变换后的 PET 形状

        return CT, PET

    def __len__(self):
        # 确保返回值大于等于 0
        return max(0, len(self.ids) - 6)

    def __getitem__(self, i):
        CT_stack = []
        for j in range(i, i + 7):
            CT_path = join(self.CT_dir, self.ids[j])
            if path.exists(CT_path):
                try:
                    CT = np.load(CT_path)
                    print(f"Loaded CT from {CT_path}, shape: {CT.shape}")  # 添加调试信息，打印加载的 CT 路径和形状
                except Exception as e:
                    print(f"CT 文件 {CT_path} 存在但读取失败，错误信息: {e}，使用默认值")
                    CT = np.zeros((512, 512))
            else:
                print(f"CT 文件 {CT_path} 不存在，使用默认值")
                CT = np.zeros((512, 512))

            # 调整 CT 数据的尺寸
            if CT.shape != (512, 512):
                CT_tensor = torch.from_numpy(CT).unsqueeze(0).float()
                CT_tensor = TF.resize(CT_tensor, size=(512, 512), interpolation=TF.InterpolationMode.BILINEAR)
                CT = CT_tensor.squeeze(0).numpy()
                print(f"Resized CT shape: {CT.shape}")  # 添加调试信息，打印调整尺寸后的 CT 形状

            CT_stack.append(CT)

        CT = np.stack(CT_stack, axis=0)
        print(f"Stacked CT shape: {CT.shape}")  # 添加调试信息，打印堆叠后的 CT 形状

        PET_path = join(self.PET_dir, self.ids[i + 3])
        if path.exists(PET_path):
            try:
                PET = np.load(PET_path)
                print(f"Loaded PET from {PET_path}, shape: {PET.shape}")  # 添加调试信息，打印加载的 PET 路径和形状
            except Exception as e:
                print(f"PET 文件 {PET_path} 存在但读取失败，错误信息: {e}，使用默认值")
                PET = np.zeros((144, 144))
        else:
            print(f"PET 文件 {PET_path} 不存在，使用默认值")
            PET = np.zeros((144, 144))

        PET = self.preprocessPET_gamma(PET)
        PET = self.edge_zero(PET)

        # 保持 PET 为 3 通道
        if PET.shape[0] == 1:
            PET = np.repeat(PET, 3, axis=0)
            print(f"Repeated PET to 3 channels, shape: {PET.shape}")  # 添加调试信息，打印重复通道后的 PET 形状

        CT = torch.from_numpy(CT).float().clone()
        PET = torch.from_numpy(PET).float().clone()

        if self.mode == 'train':
            CT, PET = self.transform(CT, PET)

        # 修改后的打印语句
        print(f"Returning data with keys: {str({'A': CT.shape, 'B': PET.shape, 'A_paths': self.CT_dir, 'B_paths': self.PET_dir, 'name': self.ids[i + 3]})}")

        return {
            'A': CT,
            'B': PET,
            'A_paths': self.CT_dir,
            'B_paths': self.PET_dir,
            'name': self.ids[i + 3]
        }

    def __repr__(self):
        return f"CTtoPETDataset(mode={self.mode}, size={len(self)})"