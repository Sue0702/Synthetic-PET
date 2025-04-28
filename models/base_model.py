import os
import torch
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """抽象基类，所有模型必须继承此类

    子类必须实现以下方法：
    -- <__init__>         : 初始化方法，需调用 BaseModel.__init__(self, opt)
    -- <set_input>        : 解包数据并预处理
    -- <forward>          : 前向传播
    -- <optimize_parameters> : 优化参数（训练模式）
    """

    def __init__(self, opt):
        """模型初始化

        参数:
            opt (Option类) : 包含所有训练参数的配置对象
        """
        nn.Module.__init__(self)  # 调用 nn.Module 初始化
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain

        # 设备管理
        self.device = self.get_device(opt)
        self.save_dir = opt.checkpoints_dir  # 直接使用 checkpoints_dir

        # 初始化参数
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # 用于学习率调整策略

    @staticmethod
    def get_device(opt):
        """获取设备（自动选择GPU或CPU）"""
        if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
            return torch.device(f'cuda:{opt.gpu_ids[0]}')
        else:
            return torch.device('cpu')

    def setup(self, opt):
        """训练模式设置（加载模型、设置调度器等）"""
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.epoch)

    def eval(self):
        """设置模型为评估模式"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        """设置模型为训练模式"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def set_requires_grad(self, nets, requires_grad=False):
        """设置网络参数是否需要梯度"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def update_learning_rate(self):
        """更新所有优化器的学习率"""
        for scheduler in self.schedulers:
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]['lr']
        print(f'Learning rate = {lr}')

    def save_networks(self, epoch):
        """保存所有网络到检查点目录

        参数:
            epoch (str/int) : 当前epoch，用于文件名
        """
        os.makedirs(self.save_dir, exist_ok=True)

        for name in self.model_names:
            if isinstance(name, str):
                save_filename = f'{epoch}_net_{name}.pth'
                save_path = os.path.join(self.save_dir, save_filename)

                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module

                torch.save(net.state_dict(), save_path)
                print(f'Saved {name} model to {save_path}')

    def load_networks(self, epoch):
        """从检查点目录加载所有网络

        参数:
            epoch (str/int) : 要加载的epoch，'latest' 表示最新模型
        """
        if epoch == 'latest':
            epoch = self.find_latest_checkpoint()

        for name in self.model_names:
            if isinstance(name, str):
                load_filename = f'{epoch}_net_{name}.pth'
                load_path = os.path.join(self.save_dir, load_filename)

                if not os.path.exists(load_path):
                    print(f'Warning: {load_path} not found')
                    continue

                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module

                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # 修复InstanceNorm的状态字典
                for key in list(state_dict.keys()):
                    self.patch_instance_norm_state_dict(state_dict, net, key.split('.'))

                net.load_state_dict(state_dict)
                print(f'Loaded {name} model from {load_path}')

    @staticmethod
    def patch_instance_norm_state_dict(state_dict, module, keys, i=0):
        """修复InstanceNorm的状态字典（旧版兼容）"""
        key = keys[i]
        if i + 1 == len(keys):  # 最后一层
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
        else:
            BaseModel.patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def find_latest_checkpoint(self):
        """查找最新的检查点文件"""
        save_dir = self.save_dir
        files = os.listdir(save_dir)
        epochs = []
        for file in files:
            if file.startswith('epoch_') and file.endswith('.pth'):
                epoch = int(file.split('_')[1])
                epochs.append(epoch)
        return max(epochs) if epochs else 'latest'

    @abstractmethod
    def set_input(self, input):
        """解包数据并预处理"""
        pass

    @abstractmethod
    def forward(self):
        """前向传播"""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """优化参数（训练模式）"""
        pass

    def get_current_visuals(self):
        """获取当前可视化结果"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """获取当前损失值"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))
        return errors_ret
