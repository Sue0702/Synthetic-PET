import os
import torch
from models.base_model import BaseModel  # 假设 BaseModel 已正确继承 nn.Module
from. import networks
import torch.nn as nn
from datetime import datetime


class Pix2PixModel(BaseModel):
    def __init__(self, opt):
        print("Pix2PixModel __init__ is called.")  # 添加调试信息
        super().__init__(opt)  # 假设 BaseModel 已正确初始化 nn.Module
        print(f"At the start of __init__ in Pix2PixModel, opt.norm = {opt.norm}")

        print(f"Before calling define_G, opt.norm = {opt.norm}")
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        print(f"After calling define_G, opt.norm = {opt.norm}")

        print(f"Before calling define_D, opt.norm = {opt.norm}")
        self.netD = networks.define_D(opt.input_nc, opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                                      opt.init_type, opt.init_gain, self.gpu_ids)
        print(f"After calling define_D, opt.norm = {opt.norm}")

        self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
        self.criterionL1 = torch.nn.L1Loss()

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.optimizers = [self.optimizer_G, self.optimizer_D]

        self.visual_names = ['real_A','real_B', 'fake_B']  # 明确设置 visual_names，确保包含'real_A'

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)

        print(f"self.real_B shape in set_input: {self.real_B.shape}")
        print(f"self.real_A shape in set_input: {self.real_A.shape}")  # 添加调试信息，打印 real_A 的形状
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        print(f"self.real_A channels: {self.real_A.shape[1]}")
        print(f"self.real_B channels: {self.real_B.shape[1]}")

    def forward(self):
        self.fake_B = self.netG(self.real_A)
        return self.fake_B

    def backward_D(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_losses(self):
        losses = {
            'G_GAN': self.loss_G_GAN.item() if hasattr(self, 'loss_G_GAN') else None,
            'G_L1': self.loss_G_L1.item() if hasattr(self, 'loss_G_L1') else None,
            'D_real': self.loss_D_real.item() if hasattr(self, 'loss_D_real') else None,
            'D_fake': self.loss_D_fake.item() if hasattr(self, 'loss_D_fake') else None
        }
        return {k: v for k, v in losses.items() if v is not None}

    def save_networks(self, epoch):
        save_dir = self.opt.checkpoints_dir
        os.makedirs(save_dir, exist_ok=True)

        save_path_G = os.path.join(save_dir, f'G_{epoch}.pth')
        save_path_D = os.path.join(save_dir, f'D_{epoch}.pth')

        print(f"Saving Generator to {save_path_G}")
        print(f"Saving Discriminator to {save_path_D}")

        torch.save(self.netG.state_dict(), save_path_G)
        torch.save(self.netD.state_dict(), save_path_D)

        print(f"Model saved: Generator -> {save_path_G}, Discriminator -> {save_path_D}")

    def load_networks(self, epoch):
        save_dir = self.opt.checkpoints_dir
        load_path_G = os.path.join(save_dir, f'G_{epoch}.pth')
        load_path_D = os.path.join(save_dir, f'D_{epoch}.pth')

        if os.path.exists(load_path_G):
            self.netG.load_state_dict(torch.load(load_path_G, map_location=self.device))
            print(f"Loaded Generator weights from {load_path_G}")
        else:
            print(f"Warning: {load_path_G} not found")

        if os.path.exists(load_path_D):
            self.netD.load_state_dict(torch.load(load_path_D, map_location=self.device))
            print(f"Loaded Discriminator weights from {load_path_D}")
        else:
            print(f"Warning: {load_path_D} not found")

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """添加新的特定于模型的选项，并重写现有选项的默认值。

        参数:
            parser (argparse.ArgumentParser) -- 原始的选项解析器
            is_train (bool) -- 是否处于训练阶段。可以用此标志添加训练或测试特定的选项。

        返回:
            修改后的解析器。
        """
        # 检查参数是否已存在
        existing_actions = [action.dest for action in parser._actions]

        # 添加特定于 Pix2PixModel 的选项
        if 'lambda_L1' not in existing_actions:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        # 如果是训练阶段，可以添加更多训练相关的选项
        if is_train:
            if 'beta1' not in existing_actions:
                parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')

        return parser

    def compute_visuals(self):
        # 这里可以添加计算可视化的逻辑
        # 例如，返回一些图像数据用于可视化
        visuals = {
           'real_A': self.real_A,
           'real_B': self.real_B,
            'fake_B': self.fake_B
        }
        print(f"visuals keys in compute_visuals: {visuals.keys()}")  # 添加调试信息，打印 visuals 字典的键名
        return visuals

    def train_epochs(self, dataloader, epochs):
        print(f"================ Training Loss ({datetime.now().strftime('%a %b %d %H:%M:%S %Y')}) ================")
        for epoch in range(epochs):
            iter_count = 0
            total_time = 0
            total_data_time = 0
            total_G_GAN_loss = 0
            total_G_L1_loss = 0
            total_D_real_loss = 0
            total_D_fake_loss = 0

            for i, data in enumerate(dataloader):
                iter_count += len(data['A'])
                self.set_input(data)
                print(f"Input data keys in train_epochs: {data.keys()}")  # 添加调试信息，打印输入数据的键名
                self.optimize_parameters()

                losses = self.get_current_losses()
                total_G_GAN_loss += losses['G_GAN']
                total_G_L1_loss += losses['G_L1']
                total_D_real_loss += losses['D_real']
                total_D_fake_loss += losses['D_fake']

            avg_G_GAN_loss = total_G_GAN_loss / len(dataloader)
            avg_G_L1_loss = total_G_L1_loss / len(dataloader)
            avg_D_real_loss = total_D_real_loss / len(dataloader)
            avg_D_fake_loss = total_D_fake_loss / len(dataloader)

            print(f"(epoch: {epoch + 1}, iters: {iter_count}, time: {total_time:.3f}, data: {total_data_time:.3f}) "
                  f"G_GAN: {avg_G_GAN_loss:.3f} G_L1: {avg_G_L1_loss:.3f} D_real: {avg_D_real_loss:.3f} D_fake: {avg_D_fake_loss:.3f}")