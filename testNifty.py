import os
import numpy as np
import nibabel as nib
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images, save_npy
from models.test_model import TestModel


checkpoints_dir = r"D:/Synthetic-PET-from-CT-main/checkpoints"
os.makedirs(checkpoints_dir, exist_ok=True)

opt = TestOptions().parse()  # 解析命令行参数
model = TestModel(opt)  # 传递 opt 给模型

checkpoint_path = r"D:/Synthetic-PET-from-CT-main/checkpoints/G_latest.pth"
# 动态选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint_path, map_location=device)
print(checkpoint.keys())
model.netG.load_state_dict(checkpoint, strict=False)  # ✅ 正确


class Nifty():
    def __init__(self, opt):
        self.preprocess_gamma = opt.preprocess_gamma
        self.temp_fl = 'temp_folder'  # temp folder name

    def path_to_nifty_files(self, test_path):
        nifty_path_list = []
        for nifty_file in os.listdir(test_path):
            if nifty_file.endswith('.nii') or nifty_file.endswith('.nii.gz'):
                nifty_path_list.append(os.path.join(test_path, nifty_file))
        return nifty_path_list

    def create_npy_from_Nifty(self, path_to_nifty, slide=1, rotation=True, transpose=True,
                              temp_folder_name='temp_folder'):
        try:
            whole_img = nib.load(path_to_nifty)
            im = whole_img.get_fdata()
            im = im.astype(np.int32)
            if rotation:
                im = np.rot90(im, -1)
                im = np.fliplr(im)
            if transpose:
                im = im.transpose((2, 0, 1))
            n_slide = im.shape[0]
            count = 0
            nifty_name = path_to_nifty.split('/')[-1].split('.')[0]
            print('1.  convert CT Nifty to npy ...    ', nifty_name, im.shape)

            for k in range(3, n_slide - 3, slide):
                im_k = np.array(im[k - 3:k + 4, :, :])
                new_folder = os.path.join(os.path.dirname(path_to_nifty), temp_folder_name)
                os.makedirs(new_folder, exist_ok=True)
                dst_npy_name = nifty_name + '_' + str(count).zfill(6) + ".npy"
                dst_img_path = os.path.join(new_folder, dst_npy_name)
                np.save(dst_img_path, im_k)
                count += 1
        except Exception as e:
            print(f"Error in create_npy_from_Nifty: {e}")

    def nonlinear_PET(self, img, middle=2.5, y_axis=0.8, minn=0.0, maxx=10.0):
        print('    nonlinear_conversion of PET was selected! ')
        img = np.clip(img, minn, 1.0)
        img_L_y_axis = (img / y_axis) * middle
        m = (maxx - middle) / (1 - y_axis)
        img_G_y_axis = img * m - m + maxx
        img = (img >= y_axis) * img_G_y_axis + (img < y_axis) * img_L_y_axis
        return img

    def post_gamma_PET(self, img, gamma=1 / 2, maxx=10.0):
        print('    gamma of {} was selected! '.format(gamma))
        img = np.clip(img, 0.0, 1.0)
        img = np.power(img, 1 / gamma)
        img = img * maxx
        return img

    def npy_to_nifti(self, nifty_path_dir, nifty_path):
        # 加载 .npy 文件
        try:
            loaded_data = np.load(nifty_path, allow_pickle=True)
        except Exception as e:
            print(f"Error loading npy file: {e}")
            return

        # 打印数组形状，用于调试
        print("Loaded data shape:", loaded_data.shape)

        # 根据实际情况调整索引
        if len(loaded_data.shape) == 3:
            loaded_data = loaded_data[0, :, :]
        elif len(loaded_data.shape) == 4:
            loaded_data = loaded_data[0, 0, :, :]
        else:
            print(f"Unexpected data shape: {loaded_data.shape}")
            return

        # 加载原始的 Nifti 文件
        try:
            whole_img = nib.load(nifty_path)
        except Exception as e:
            print(f"Error loading original Nifti file: {e}")
            return

        # 初始化 PET 数组
        PET = np.zeros((loaded_data.shape[0], loaded_data.shape[1], 1))  # 假设形状

        # 假设 file_list 和 pred_path 已正确定义
        file_list = []  # 示例：['file1.npy', 'file2.npy', ...]
        pred_path = ''  # 示例：'path/to/predicted/npy'

        for i, file_path in enumerate(file_list):
            try:
                loaded_data = np.load(os.path.join(pred_path, file_path), allow_pickle=True)
            except Exception as e:
                print(f"Error loading npy file: {e}")
                continue

            # 调整数据形状
            if loaded_data.shape != (512, 512):
                if len(loaded_data.shape) == 4:
                    loaded_data = loaded_data[0, 0, :, :]
                else:
                    print(f"Unexpected data shape for {file_path}: {loaded_data.shape}")
                    continue

            PET[:, :, i + 3] = np.rot90(loaded_data, +1)
            PET[:, :, i + 3] = np.flipud(PET[:, :, i + 3])

        # 应用非线性或 Gamma 变换
        if self.preprocess_gamma:
            PET = self.post_gamma_PET(PET, maxx=7.0)
        else:
            PET = self.nonlinear_PET(PET)

        # 保存为 Nifti 文件
        try:
            img_nifti = nib.Nifti1Image(PET, whole_img.affine)
            nifty_file = os.path.basename(nifty_path)
            out_files = f"{os.path.splitext(nifty_file)[0]}_pred.nii.gz"
            nifty_folder = os.path.join(os.path.dirname(pred_path), 'nifty_pred')
            os.makedirs(nifty_folder, exist_ok=True)
            img_nifti.to_filename(os.path.join(nifty_folder, out_files))
            print('3.  converted to Nifty!', out_files)
        except Exception as e:
            print(f"Error saving Nifti file: {e}")

    def remove_npy(self, path):
        all_files = os.listdir(path) if os.path.exists(path) else []
        for file in all_files:
            if file.endswith('.npy'):
                os.remove(os.path.join(path, file))


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.checkpoints_dir = r"D:/Synthetic-PET-from-CT-main/checkpoints"
    print("opt.checkpoints_dir:", opt.checkpoints_dir)
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()
    nifty_obj = Nifty(opt)
    niftys_path_list = nifty_obj.path_to_nifty_files(opt.dataroot)
    print('############################################################################################')
    for nifty_path in niftys_path_list:
        only_nifty_name = os.path.basename(nifty_path).split('.')[0]
        # delete if there is some remaining npy files in case.
        nifty_obj.remove_npy(os.path.join(os.path.split(nifty_path)[0], nifty_obj.temp_fl))
        nifty_obj.remove_npy(os.path.join(opt.results_dir, opt.name, opt.npy_save_name))
        # create npy files from Nifty files and store them in 'temp_folder'
        nifty_obj.create_npy_from_Nifty(nifty_path, slide=1, temp_folder_name=nifty_obj.temp_fl)
        dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options (opt.dataset_mode='cttopet')
        for i, data in enumerate(dataset):  ## data: dict_keys(['A', 'B', 'A_paths', 'B_paths'])
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            # 检查输入数据的通道数
            input_data = data['A']
            print("Input data shape before adjustment:", input_data.shape)
            if input_data.shape[1] != 7:
                print("Warning: Input data does not have 7 channels.")
                # 根据输入数据的实际形状调整 repeat 参数
                if input_data.dim() == 5:
                    # 假设输入数据形状为 (batch_size, 1, depth, height, width)
                    # 这里 depth 可能是 7 层数据，需要调整维度
                    input_data = input_data.squeeze(1)  # 移除通道维度
                    input_data = input_data.permute(0, 3, 1, 2)  # 调整维度顺序为 (batch_size, height, width, depth)
                    input_data = input_data.unsqueeze(1)  # 重新添加通道维度
                    input_data = input_data.repeat(1, 7, 1, 1, 1)  # 重复通道维度
                else:
                    print("Unsupported input data shape.")
                    continue
            data['A'] = input_data
            print("Input data shape after adjustment:", input_data.shape)

            # 沿着深度维度循环处理数据
            depth = input_data.shape[2]
            fake_outputs = []
            for d in range(depth):
                slice_data = input_data[:, :, d, :, :]  # 取出一个切片
                print("Slice data shape:", slice_data.shape)
                # 添加 'A_paths' 键到新的输入字典中
                new_input = {'A': slice_data, 'A_paths': data['A_paths']}
                model.set_input(new_input)  # unpack data from data loader
                model.test()  # run inference
                fake_output = model.get_current_visuals()['fake_B']
                # 确保每个 fake_output 是四维张量
                if fake_output.dim() != 4:
                    fake_output = fake_output.unsqueeze(0)  # 添加批次维度
                fake_outputs.append(fake_output)

            # 拼接并调整维度
            fake_outputs = torch.cat(fake_outputs, dim=0)  # 形状 (84, 1, 3, 512, 512)
            print("After concatenation, fake_outputs shape:", fake_outputs.shape)

            # 移除多余的 batch 维度（如果存在）
            if fake_outputs.shape[1] == 1:
                fake_outputs = fake_outputs.squeeze(1)  # 形状 (84, 3, 512, 512)

            # 调整维度顺序为 (channels, height, width, depth)
            fake_outputs = fake_outputs.permute(1, 2, 3, 0)  # 形状 (3, 512, 512, 84)

            # 取通道 1 并保持四维
            fake_outputs = fake_outputs[1:2, :, :, :]  # 形状 (1, 512, 512, 84)

            # 调整维度顺序为 (height, width, depth, channels)
            fake_outputs = fake_outputs.permute(1, 2, 3, 0)  # 形状 (512, 512, 84, 1)

            fake_outputs_np = fake_outputs.cpu().numpy()  # 转换为 numpy 数组

        print('2.  npy prediction is done!', os.path.split(nifty_path)[1])

        nifty_path_dir = os.path.split(nifty_path)[0]
        nifty_obj.npy_to_nifti(nifty_path_dir, nifty_path)

