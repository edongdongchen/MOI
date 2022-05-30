import torch
import argparse

from moi.moi import MOI

from datasets.mridb import multi_op_mri_dataloader
from datasets.mnist import multi_op_mnist_dataloader
from datasets.celeba import multi_op_celeba_dataloader

from physics.mri import get_group_mri_ops
from physics.inpainting import get_group_inpainting_ops, get_group_inpainting_mnist_ops
from physics.cs import get_group_cs_ops


"""
PyTorch implementation of the below paper:

@article{tachella2022sampling,
title={Unsupervised Learning From Incomplete Measurements for Inverse Problems},
author={Tachella, Juli{\'a}n and Chen, Dongdong and Davies, Mike},
journal={arXiv preprint arXiv:2201.12151},
year={2022}}
}

26/May/2022: by Dongdong Chen (d.chen@ed.ac.uk)
"""

parser = argparse.ArgumentParser(description='MOI')

parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
parser.add_argument('--schedule', nargs='+', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')
parser.add_argument('--epochs', default=500, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-1, type=float,
                    metavar='LR', help='initial learning rate',
                    dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-8, type=float,
                    metavar='W', help='weight decay (default: 1e-8)',
                    dest='weight_decay')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--ckp-interval', default=100, type=int,
                    help='save checkpoints interval epochs')

# inverse problem task configs:
parser.add_argument('--task', default='mnist-cs', type=str,
                    help="inverse problems=['mnist-cs', 'mnist-inpainting', "
                         "'celeba-inpainting', 'fastmri-mri'] (default: 'mnist-cs')")
parser.add_argument('-G', default=40, type=int,
                    help='number of operators (40)')
parser.add_argument('--mri-acceleration', default=4, type=int,
                    help='acceleration ratio for MRI task (default: 4)')
parser.add_argument('-m', '--dim-y',default=100, type=int,
                    help='dim of meas. for CS and Inpainting tasks on MNIST(default: 100)')
parser.add_argument('--mask-rate', default=0.5, type=float,
                    help='mask rate for Inpainting-CelebA task (default: 0.5)')

if __name__ == '__main__':
    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.gpu}")
    args.dtype = torch.float
    args.net_name = f'moi_{args.task}'

    if args.task == 'mnist-cs':
        args.arch = 'ae'
        args.dim_input = 28 * 28
        args.dim_hid = 1000
        args.residual = True

        args.lr = 1e-4
        args.wd = 1e-8

        args.cos = False
        args.epochs = 1000
        args.batch_size = 128
        args.schedule = [800]
        args.ckp_interval = 100

        # args.G = 40 # 1,10,20,30,40
        # args.m = 100 # 50,100,200,300,400
        args.n = 1 * 28 * 28

        train_loader_group = multi_op_mnist_dataloader('train', args.batch_size, True, G=args.G)
        physics_group = get_group_cs_ops(args.m, args.n, args.image_shape, G=args.G, dtype=args.dtype, device=args.device)

    if args.task == 'mnist-inpainting':
        args.arch = 'ae'
        args.dim_input = 28 * 28
        args.dim_hid = 1000
        args.residual = True

        args.lr = 5e-4
        args.wd = 1e-8

        args.cos = False
        args.epochs = 500
        args.batch_size = 128
        args.schedule = [300]
        args.ckp_interval = 10

        # args.G = 40 # 1,10,20,30,40
        # args.m = 100 # 50,100,200,300,400
        args.n = 1 * 28 * 28

        train_loader_group = multi_op_mnist_dataloader('train', args.batch_size, True, G=args.G)
        physics_group = get_group_inpainting_mnist_ops(args.d, args.D, args.image_shape, G=args.G, device=args.device)

    if args.task == 'celeba-inpainting':
        args.G = 40
        args.img_size = 128
        args.mask_rate = 0.5

        args.arch = 'unet'
        args.cat = True
        args.residual = True
        args.circular_padding = True
        args.in_channels = 3
        args.out_channels = 3

        args.lr = 5e-4
        args.wd = 1e-8

        args.cos = False
        args.epochs = 300
        args.batch_size = 20
        args.schedule = [200]
        args.ckp_interval = 100

        train_loader_group = multi_op_celeba_dataloader('train', args.batch_size, True, crop_size=(128,128), G=args.G)
        physics_group = get_group_inpainting_ops(args.mask_rate, args.img_heighth, args.img_width, args.device, args.G)

    if args.task == 'fastmri-mri':
        args.G = 40
        args.img_size = 320
        args.mri_acceleration = '4'

        args.arch = 'unet'
        args.cat = True
        args.residual = True
        args.circular_padding = True
        args.in_channels = 2
        args.out_channels = 2

        args.lr = 5e-4
        args.wd = 0
        args.cos = False

        args.epochs = 500
        args.batch_size = 4
        args.schedule = [300]
        args.ckp_interval = 100

        train_loader_group = multi_op_mri_dataloader('train', args.batch_size, True, tag=args.mri_tag, G=args.G)
        physics_group = get_group_mri_ops(acceleration=args.mri_acceleration, device=args.device, G=args.G)

    moi = MOI(args)
    moi.train(train_loader_group, physics_group)
