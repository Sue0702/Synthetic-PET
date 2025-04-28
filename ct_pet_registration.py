import SimpleITK as sitk
import os
import uuid

def generate_uid():
    """生成符合 DICOM 规范的唯一标识符"""
    return "2.25." + str(uuid.uuid4().int)[:36]

def load_dicom_series(dicom_dir):
    """
    加载 DICOM 系列数据并收集每个切片的元数据
    """
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)

    # 读取每个DICOM文件的元数据
    original_metadata_list = []
    for file_path in dicom_files:
        reader_meta = sitk.ImageFileReader()
        reader_meta.SetFileName(file_path)
        reader_meta.LoadPrivateTagsOn()
        reader_meta.ReadImageInformation()
        metadata = {}
        for key in reader_meta.GetMetaDataKeys():
            metadata[key] = reader_meta.GetMetaData(key)
        original_metadata_list.append(metadata)

    # 读取整个图像
    reader.SetFileNames(dicom_files)
    try:
        image = reader.Execute()
        return sitk.Cast(image, sitk.sitkFloat32), original_metadata_list
    except Exception as e:
        print(f"加载 DICOM 系列时出错: {e}")
        return None, None

def process_and_resample_ct(ct_image, target_size=(144, 144)):
    """
    处理 CT 图像，按照规则保留和删除层，并进行重采样
    """
    if ct_image is None:
        return None

    original_size = ct_image.GetSize()
    original_spacing = ct_image.GetSpacing()
    original_origin = ct_image.GetOrigin()
    original_direction = ct_image.GetDirection()

    new_size = [
        target_size[0],
        target_size[1],
        original_size[2] // 2  # 保留每隔一层
    ]

    # 计算新的间距
    new_spacing = (
        original_spacing[0] * (original_size[0] / new_size[0]),
        original_spacing[1] * (original_size[1] / new_size[1]),
        original_spacing[2] * 2  # 层间距加倍
    )

    # 创建保留层的图像
    new_ct_image = sitk.Image(new_size, ct_image.GetPixelID())
    new_ct_image.SetOrigin(original_origin)
    new_ct_image.SetSpacing(new_spacing)
    new_ct_image.SetDirection(original_direction)

    # 填充保留的层
    for z in range(new_size[2]):
        src_z = z * 2
        for y in range(min(original_size[1], new_size[1])):
            for x in range(min(original_size[0], new_size[0])):
                new_ct_image[x, y, z] = ct_image[x, y, src_z]

    # 重采样以匹配目标尺寸
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(new_ct_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled_ct = resampler.Execute(new_ct_image)

    return resampled_ct

def save_dicom_image(image, output_dir, original_metadata_list):
    """
    将图像保存为DICOM系列，保留必要元数据
    """
    if image is None or not original_metadata_list:
        return

    os.makedirs(output_dir, exist_ok=True)
    size = image.GetSize()
    num_slices = size[2]
    spacing = image.GetSpacing()

    # 生成新的唯一标识符
    series_instance_uid = generate_uid()
    study_instance_uid = generate_uid()

    for z in range(num_slices):
        # 获取对应的原始元数据
        original_z = z * 2
        if original_z >= len(original_metadata_list):
            continue

        # 创建切片图像
        slice_img = image[:, :, z]
        slice_img = sitk.Cast(slice_img, sitk.sitkInt16)

        # 复制原始元数据
        metadata = original_metadata_list[original_z].copy()

        # 更新关键元数据
        metadata["0020|000e"] = series_instance_uid  # Series Instance UID
        metadata["0020|000d"] = study_instance_uid  # Study Instance UID

        # 更新几何信息
        position = image.TransformIndexToPhysicalPoint((0, 0, z))
        metadata["0020|0032"] = "\\".join(f"{p:.6f}" for p in position)  # Image Position
        metadata["0028|0030"] = f"{spacing[0]:.6f}\\{spacing[1]:.6f}"  # Pixel Spacing
        metadata["0018|0050"] = f"{spacing[2]:.6f}"  # Slice Thickness

        # 应用元数据到图像
        for key, value in metadata.items():
            slice_img.SetMetaData(key, value)

        # 设置新的SOP Instance UID
        slice_img.SetMetaData("0008|0018", generate_uid())

        # 写入文件
        writer = sitk.ImageFileWriter()
        writer.SetFileName(os.path.join(output_dir, f"slice_{z:03d}.dcm"))
        writer.Execute(slice_img)

if __name__ == "__main__":
    ct_dicom_dir = r"E:/2023.6.1-2024.12.31RAW/hutaiming/1.3.46.670589.33.1.63848095126244764200004.4920574690626924124"
    output_dicom_dir = r"E:/2023.6.1-2024.12.31RAW/hutaiming/processed_CT"

    # 加载数据
    ct_image, metadata_list = load_dicom_series(ct_dicom_dir)

    # 处理图像
    processed_ct = process_and_resample_ct(ct_image)

    # 保存结果
    if processed_ct and metadata_list:
        save_dicom_image(processed_ct, output_dicom_dir, metadata_list)
