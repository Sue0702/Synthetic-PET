from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # 继承 BaseOptions 中的选项
        parser.add_argument('--results_dir', type=str, default='D:/Synthetic-PET-from-CT-main/checkpoints', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='', help='train, val, test, etc')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=2500, help='how many test images to run')
        parser.add_argument('--npy_save_name', type=str, default='npy_results',
                            help='folder name to save the results as .npy')
        # rewrite default values
        parser.set_defaults(model='test')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))

        # 继承 BaseOptions 中的其它选项，不需要再定义
        self.isTrain = False
        return parser
