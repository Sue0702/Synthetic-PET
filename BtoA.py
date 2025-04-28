import os


def rename_files(folder_path):
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在。")
        return

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件名是否以 B 开头
        if filename.startswith('B'):
            # 构建新的文件名，将 B 替换为 A
            new_filename = 'A' + filename[1:]
            # 构建旧文件的完整路径
            old_file_path = os.path.join(folder_path, filename)
            # 构建新文件的完整路径
            new_file_path = os.path.join(folder_path, new_filename)
            try:
                # 重命名文件
                os.rename(old_file_path, new_file_path)
                print(f"已将 {old_file_path} 重命名为 {new_file_path}")
            except Exception as e:
                print(f"重命名 {old_file_path} 时出错: {e}")


# 请将此路径替换为你实际的文件夹路径
folder_path = r'D:\Synthetic-PET-from-CT-main\data_7CHL\pix2pix_7Ch7\trainB'
rename_files(folder_path)
