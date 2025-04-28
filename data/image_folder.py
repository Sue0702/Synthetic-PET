import os

def is_image_file(filename):
    """判断文件是否是 NIfTI 文件"""
    return filename.endswith(".nii") or filename.endswith(".nii.gz")

def make_dataset(dir):
    """获取目录下所有 NIfTI 文件路径"""
    images = []
    assert os.path.isdir(dir), f"{dir} 不是一个有效的文件夹路径！"

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images
