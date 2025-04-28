import os
import numpy as np
import pydicom

# 设置输入 DICOM 文件夹和输出 npy 文件夹
dicom_folder = r"E:\2023.6.1-2024.12.31RAW\baoweiping\RAWCT"
output_folder = r"E:\2023.6.1-2024.12.31RAW\hutaiming\CT"
prefix = "A"  # 数据来源标识

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历 DICOM 文件夹，按病人 ID 分组
patient_dict = {}  # 存储病人 ID 对应的序号

for root, _, files in os.walk(dicom_folder):
    # 只处理 .dcm 文件
    dicom_files = sorted([f for f in files if f.endswith(".dcm")])

    if not dicom_files:
        continue  # 跳过没有 DICOM 文件的文件夹

    # 获取病人 ID（从 DICOM metadata 读取）
    sample_dicom = pydicom.dcmread(os.path.join(root, dicom_files[0]))
    patient_id = sample_dicom.PatientID if hasattr(sample_dicom, "PatientID") else os.path.basename(root)

    # 给病人编号（例如 A1, A2, A3）
    if patient_id not in patient_dict:
        patient_dict[patient_id] = len(patient_dict) + 1  # 递增病人编号

    patient_num = patient_dict[patient_id]

    # 读取并转换每个 DICOM（仅保留奇数层）
    for i, file in enumerate(dicom_files):
        if i % 2 == 1:  # 只保留索引为奇数的 DICOM 文件
            dicom_path = os.path.join(root, file)
            ds = pydicom.dcmread(dicom_path)  # 读取 DICOM 文件
            pixel_array = ds.pixel_array.astype(np.int16)  # 转换为 NumPy 数组

            # 生成文件名 A1-001.npy
            new_filename = f"{prefix}{patient_num:02d}-{(i // 2) + 1:03d}.npy"
            output_path = os.path.join(output_folder, new_filename)

            # 保存为 .npy
            np.save(output_path, pixel_array)

            print(f"✅ 已保存: {output_path}")

print("🎉 仅奇数层的 DICOM 文件已转换并重命名为 npy！")