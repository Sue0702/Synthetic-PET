import time
import signal
import logging
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from visdom import Visdom
import sys
import os

# 添加当前脚本所在目录到模块搜索路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 清除模块缓存
if 'models.pix2pix_model' in sys.modules:
    del sys.modules['models.pix2pix_model']

# 其他代码保持不变
from options.train_options import TrainOptions

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('D:/Synthetic-PET-from-CT-main/checkpoints', 'training.log')),
        logging.StreamHandler()
    ]
)

opt = TrainOptions().parse()

# 修改 Visdom 初始化，禁用传入套接字
vis = Visdom(port=8097, use_incoming_socket=False)

# 根据 CUDA 可用性选择设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logging.info(f"Using device: {device}")

# 中断信号处理
interrupted = False


def signal_handler(sig, frame):
    global interrupted
    logging.info('You pressed Ctrl+C! Saving the model and exiting...')
    interrupted = True


signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    tensorboard_save = True
    opt = TrainOptions().parse()

    # 创建检查点目录
    import os

    # 设置检查点保存的绝对路径，你可以根据实际情况修改
    opt.checkpoints_dir = 'D:/Synthetic-PET-from-CT-main/checkpoints'
    os.makedirs(opt.checkpoints_dir, exist_ok=True)
    logging.info(f"Checkpoint directory created: {opt.checkpoints_dir}")

    dataset = create_dataset(opt)
    dataset_size = len(dataset)
    logging.info('The number of training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    model.to(device)  # 将模型移动到指定设备
    logging.info("Model moved to the specified device.")

    total_iters = 0

    if tensorboard_save:
        writer = SummaryWriter(comment=f'LR_{opt.lr}_lambda_L1_{opt.lambda_L1}')

    try:
        for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):
            if interrupted:
                break
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0
            model.update_learning_rate()
            loss_list = []
            logging.info(f"Starting epoch {epoch}...")

            for i, data in enumerate(dataset):
                if interrupted:
                    break
                iter_start_time = time.time()

                # 加速数据迁移，将数据移动到指定设备
                data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size

                model.set_input(data)
                model.optimize_parameters()

                if total_iters % opt.display_freq == 0:
                    model.compute_visuals()

                if total_iters % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    logging.info(f"Epoch {epoch}, Iter {total_iters}: Losses - {losses}, Comp Time: {t_comp}")

                if total_iters % opt.save_latest_freq == 0:
                    logging.info('Saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                    save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                    model.save_networks(save_suffix)
                    logging.info(f"Model saved with suffix: {save_suffix}")

                losses = model.get_current_losses()
                loss_list.append([losses['G_GAN'], losses['G_L1'], losses['D_real'], losses['D_fake']])
                iter_data_time = time.time()

            if interrupted:
                logging.info('Saving interrupted model...')
                model.save_networks('latest')
                logging.info("Model saved as 'latest' due to interruption.")
                model.save_networks(epoch)
                logging.info(f"Model saved for epoch {epoch} due to interruption.")
                break

            epoch_end_time = time.time()
            epoch_time = epoch_end_time - epoch_start_time
            logging.info(f'Epoch {epoch} completed in {epoch_time:.2f} seconds.')

            if tensorboard_save:
                model.compute_visuals()
                visuals = model.get_current_visuals()

                # 检查 'real_A' 键是否存在
                if 'real_A' not in visuals:
                    logging.error(f"Error: 'real_A' key not found in visuals. Available keys: {list(visuals.keys())}")
                else:
                    mid_rA_idx = (visuals['real_A'].shape[1] - 1) // 2
                    mid_rB_idx = (visuals['real_B'].shape[1] - 1) // 2
                    mid_fB_idx = (visuals['fake_B'].shape[1] - 1) // 2

                    writer.add_images('CT', visuals['real_A'][:, mid_rA_idx:mid_rA_idx + 1], epoch)
                    writer.add_images('PET_Real', visuals['real_B'][:, mid_rB_idx:mid_rB_idx + 1], epoch)
                    writer.add_images('PET_Fake',
                                      torch.clamp(visuals['fake_B'][:, mid_fB_idx:mid_fB_idx + 1], 0.0, 1.0),
                                      epoch)

                loss_list_mean = np.mean(loss_list, axis=0)
                writer.add_scalar('Loss/Train/G_GAN', loss_list_mean[0], epoch)
                writer.add_scalar('Loss/Train/G_L1', loss_list_mean[1], epoch)
                writer.add_scalar('Loss/Train/D_real', loss_list_mean[2], epoch)
                writer.add_scalar('Loss/Train/D_fake', loss_list_mean[3], epoch)
                writer.add_scalar('LR', model.optimizer_G.param_groups[0]['lr'], epoch)

            # 每一轮结束都保存模型
            if epoch % 10 == 0:
                logging.info('Saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                logging.info("Model saved as 'latest' at the end of epoch.")
                model.save_networks(epoch)
                logging.info(f"Model saved for epoch {epoch} at the end of epoch.")

            logging.info('End of epoch %d / %d \t Time Taken: %d sec' % (
                epoch, opt.n_epochs + opt.n_epochs_decay, epoch_time))

    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            logging.error("ERROR: CUDA out of memory. Try reducing batch size.")
        else:
            logging.error(f"Runtime Error: {e}")
        exit(1)

    if tensorboard_save:
        writer.close()
