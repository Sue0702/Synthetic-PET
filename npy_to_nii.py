import os
import numpy as np
import nibabel as nib


def merge_npy_to_nii(npy_folder_path, nii_file_path):
    # 检查文件夹是否存在
    if not os.path.exists(npy_folder_path):
        print(f"错误：未找到 {npy_folder_path} 文件夹。")
        return

    # 获取文件夹中所有的 .npy 文件
    npy_files = [os.path.join(npy_folder_path, f) for f in os.listdir(npy_folder_path) if f.endswith('.npy')]
    if not npy_files:
        print(f"错误：{npy_folder_path} 文件夹中未找到 .npy 文件。")
        return

    # 加载并合并 .npy 文件
    all_data = []
    for npy_file in npy_files:
        try:
            data = np.load(npy_file)
            all_data.append(data)
        except FileNotFoundError:
            print(f"错误：未找到 {npy_file} 文件。")
        except Exception as e:
            print(f"错误：加载 {npy_file} 文件时出现问题：{e}")

    if not all_data:
        print("错误：没有成功加载任何 .npy 文件。")
        return

    # 合并数据
    merged_data = np.stack(all_data, axis=-1)

    # 创建 NIfTI 图像对象
    nii_img = nib.Nifti1Image(merged_data, np.eye(4))

    # 保存为 .nii 文件
    try:
        nib.save(nii_img, nii_file_path)
        print(f"成功将 {npy_folder_path} 中的 .npy 文件合并转换为 {nii_file_path}")
    except Exception as e:
        print(f"错误：保存为 {nii_file_path} 时出现问题：{e}")


if __name__ == "__main__":
    # 请将下面的路径替换为你实际存放 .npy 文件的文件夹路径
    npy_folder_path = 'E:/npy'
    # 请将下面的路径替换为你想要保存的 .nii 文件路径
    nii_file_path = 'E:/npy1'
    merge_npy_to_nii(npy_folder_path, nii_file_path)
