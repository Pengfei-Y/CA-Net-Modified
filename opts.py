import argparse
from distutils.version import LooseVersion
import torch


def get_parser():
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), 'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(description='U-net add Attention mechanism for biomedical Dataset')
    # Model related arguments
    parser.add_argument('--id', default='Comp_Atten_Unet',
                        help='a name for identitying the model. Choose from the following options: Unet_fetus')
    # Path related arguments
    parser.add_argument('--root_path', default='./data/ISIC2018_Task1_npy_all',
                        help='root directory of data')
    parser.add_argument('--ckpt', default='./saved_models',
                        help='folder to output checkpoints')
    parser.add_argument('--save', default='./result',
                        help='folder to outoput result')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--num_input', default=3, type=int,
                        help='number of input image for each patient')
    parser.add_argument('--epoch', type=int, default=300, metavar='N',
                        help='choose the specific epoch checkpoints')

    # other arguments
    parser.add_argument('--data', default='ISIC2018', help='choose the dataset')
    parser.add_argument('--out_size', default=(224, 300), help='the output image size')
    parser.add_argument('--att_pos', default='dec', type=str,
                        help='where attention to plug in (enc, dec, enc\&dec)')
    parser.add_argument('--view', default='axial', type=str,
                        help='use what views data to test (for fetal MRI)')
    parser.add_argument('--val_folder', default='folder0', type=str,
                        help='which cross validation folder')

    args = parser.parse_args()
    return args