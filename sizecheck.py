import os
import numpy as np

# 定义要检查的目录路径，这里以你的 CT 数据目录为例，你可以修改为实际的 CT 或 PET 数据目录
data_dir = 'D:/Synthetic-PET-from-CT-main/data_7CHL/pix2pix_7Ch7/trainB'

for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.npy'):
            file_path = os.path.join(root, file)
            try:
                data = np.load(file_path)
                print(f"File: {file_path}, Shape: {data.shape}")
            except:
                print(f"Error loading file: {file_path}")