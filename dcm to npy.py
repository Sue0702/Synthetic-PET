import os
import pydicom
import numpy as np

def dcm_to_npy(dicom_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历 DICOM 文件夹中的所有文件
    for filename in os.listdir(dicom_folder):
        if filename.endswith('.dcm'):
            # 构建完整的 DICOM 文件路径
            dcm_path = os.path.join(dicom_folder, filename)
            try:
                # 读取 DICOM 文件
                ds = pydicom.dcmread(dcm_path)
                # 获取图像数据
                image = ds.pixel_array
                # 构建输出的 .npy 文件路径，保持文件名相同，仅更改扩展名
                npy_filename = os.path.splitext(filename)[0] + '.npy'
                npy_path = os.path.join(output_folder, npy_filename)
                # 保存为 .npy 文件
                np.save(npy_path, image)
                print(f"Converted {filename} to {npy_filename}")
            except Exception as e:
                print(f"Error converting {filename}: {e}")

if __name__ == "__main__":
    # 请替换为实际的 DICOM 文件夹路径
    dicom_folder = "D:/manifest-1622561851074/NSCLC Radiogenomics/AMC-001/04-30-1994-NA-PETCT Lung Cancer-74760/6.000000-WB MAC P690-69577"
    # 请替换为实际的输出文件夹路径
    output_folder = "D:/训练数据/PET"
    dcm_to_npy(dicom_folder, output_folder)