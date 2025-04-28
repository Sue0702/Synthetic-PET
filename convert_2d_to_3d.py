# -*- coding: utf-8 -*-
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm


def load_slice(file_path):
    """加载单张切片，支持 .npy 格式"""
    if file_path.endswith('.npy'):
        # 读取 .npy 文件
        slice_array = np.load(file_path).astype(np.float32)
        # 这里假设没有切片位置信息，返回默认值 0
        slice_pos = 0
        return slice_array, slice_pos
    else:
        print(f"不支持的文件格式: {file_path}")
        return None, 0


def sort_slices(slice_files):
    """根据文件名或 DICOM 切片位置排序"""
    sorted_slices = []
    for file in slice_files:
        slice_array, slice_pos = load_slice(file)
        if slice_array is not None:
            sorted_slices.append((slice_pos, file, slice_array))

    # 按切片位置排序（从小到大，从头到脚）
    sorted_slices.sort(key=lambda x: x[0])
    return [arr for _, _, arr in sorted_slices]


def convert_2d_to_3d(input_dir, output_dir, output_format='npy', target_channels=1, required_depth=1):
    """
    主函数：将二维切片转换为三维体积
    参数：
      input_dir   : 输入目录（可能包含子文件夹）
      output_dir  : 输出目录（保存为 .npy 或 .nii.gz）
      output_format : 输出格式 ['npy' | 'nii']
      target_channels : 目标通道数（根据模型需求扩展）
      required_depth: 所需的最小切片数量
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 递归查找所有 .npy 文件
    slice_files = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith('.npy'):
                slice_files.append(os.path.join(root, f))

    if len(slice_files) == 0:
        print(f"警告：输入目录 {input_dir} 中无有效切片文件，跳过")
        return

    # 加载并排序切片
    try:
        sorted_arrays = sort_slices(slice_files)
    except Exception as e:
        print(f"排序失败: {str(e)}")
        return

    # 检查切片数量是否足够
    if len(sorted_arrays) < required_depth:
        print(f"警告：切片数量不足，跳过")
        return

    # 检查切片尺寸是否一致
    first_slice = sorted_arrays[0]
    for arr in sorted_arrays[1:]:
        if arr.shape != first_slice.shape:
            print(f"警告：存在尺寸不一致的切片，跳过")
            return

    # 堆叠为3D数组 (Depth, H, W)
    volume_3d = np.stack(sorted_arrays, axis=0)

    # 扩展通道维度 (Depth, H, W) → (Depth, H, W, C)
    if target_channels > 1:
        volume_3d = np.repeat(volume_3d[..., np.newaxis], target_channels, axis=-1)
    else:
        volume_3d = volume_3d[..., np.newaxis]  # 添加单一通道

    # 归一化（示例：CT值转为0~1）
    volume_3d = (volume_3d - np.min(volume_3d)) / (np.max(volume_3d) - np.min(volume_3d))

    # 调整维度顺序为 (C, D, H, W)
    volume_3d = volume_3d.transpose((3, 0, 1, 2))

    # 保存文件
    output_path = os.path.join(output_dir, "combined_3d")
    if output_format == 'npy':
        np.save(output_path, volume_3d)
    elif output_format == 'nii':
        # 创建NIfTI文件（需设定仿射矩阵，此处用单位矩阵示例）
        affine = np.eye(4)
        nii_img = nib.Nifti1Image(volume_3d, affine)
        nib.save(nii_img, f"{output_path}.nii.gz")
    else:
        raise ValueError("输出格式仅支持 'npy' 或 'nii'")

    print(f"三维数组已保存，形状: {volume_3d.shape}")


if __name__ == "__main__":
    # 使用示例 --------------------------
    input_dir = "D:/训练数据/CT"  # 输入目录可能包含子文件夹
    output_dir = "D:/训练数据/CT_output"  # 输出目录
    output_format = "npy"  # 输出格式
    target_channels = 7  # 根据模型 input_nc 设置（示例中需要7通道）
    required_depth = 1  # 所需的最小切片数量

    convert_2d_to_3d(
        input_dir=input_dir,
        output_dir=output_dir,
        output_format=output_format,
        target_channels=target_channels,
        required_depth=required_depth
    )