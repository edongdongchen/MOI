import torch
from models.ae import AE
from models.unet import UNet

from datasets.mridb import multi_op_mri_dataloader
from datasets.mnist import multi_op_mnist_dataloader
from datasets.celeba import multi_op_celeba_dataloader

from physics.mri import get_group_mri_ops
from physics.inpainting import get_group_inpainting_ops, get_group_inpainting_mnist_ops
from physics.cs import get_group_cs_ops
from utils.metric import cal_psnr

import numpy as np

def test(ckp_path, cuda=0):
    device = torch.device(f"cuda:{cuda}")
    dtype = torch.float

    checkpoint = torch.load(ckp_path, map_location=device)
    args = checkpoint['args']

    if args.task == 'mnist-cs':
        test_loader_group = multi_op_mnist_dataloader('test', batch_size=1, shuffle=True, G=args.G)
        physics_group = get_group_cs_ops(args.m, args.n, args.image_shape, G=args.G, dtype=args.dtype, device=args.device)

    if args.task == 'mnist-inpainting':
        test_loader_group = multi_op_mnist_dataloader('test', batch_size=1, shuffle=True, G=args.G)
        physics_group = get_group_inpainting_mnist_ops(args.d, args.D, args.image_shape, G=args.G, device=args.device)

    if args.task == 'celeba-inpainting':
        test_loader_group = multi_op_celeba_dataloader('test', batch_size=1, shuffle=True, crop_size=(128,128), G=args.G)
        physics_group = get_group_inpainting_ops(args.mask_rate, args.img_heighth, args.img_width, args.device, args.G)

    if args.task == 'fastmri-mri':
        test_loader_group = multi_op_mri_dataloader('test', batch_size=1, shuffle=True, tag=args.mri_tag, G=args.G)
        physics_group = get_group_mri_ops(acceleration=args.mri_acceleration, device=args.device, G=args.G)

    if args.arch == 'ae':
        model = AE(residual=args.residual,
                   dim_input=args.dim_input,
                   dim_hid=args.dim_hid).to(args.device)
    if args.arch == 'unet':
        model = UNet(in_channels=args.in_channels,
                     out_channels=args.out_channels,
                     residual=args.residual,
                     circular_padding=args.circular_padding,
                     cat=args.cat).to(args.device)

    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()

    psnr_net_seq, psnr_fbp_seq = [], []
    for g in range(len(test_loader_group)):
        test_loader = test_loader_group[g]
        physics = physics_group[g]
        for i, x in enumerate(test_loader):
            x_gt = x[0] if isinstance(x, list) else x
            x_gt = x_gt.unsqueeze(1) if len(x_gt.shape) == 3 else x_gt
            x_gt = x_gt.type(dtype).to(device)

            y0 = physics.A(x_gt.type(dtype).to(device))
            fbp = physics.A_dagger(y0)

            x_net = model(fbp)

            psnr_net_seq.append(cal_psnr(x_net, x_gt, complex=args.task in ['fastmri-mri']))
            psnr_fbp_seq.append(cal_psnr(fbp, x_gt, complex=args.task in ['fastmri-mri']))

    print('{} || \tA^+y: psnr={:.2f}\t std.={:.2f} || \tNet: psnr={:.2f}\tstd.={:.2f}'
        .format(args.net_name, np.mean(psnr_fbp_seq), np.std(psnr_fbp_seq),
                np.mean(psnr_net_seq), np.std(psnr_net_seq)))


if __name__ == '__main__':
    cuda=0
    test(f'trained_cs_mnist_ckp.pth.tar', cuda=cuda)
    test(f'trained_inpainting_mnist_ckp.pth.tar', cuda=cuda)
    test(f'trained_inpainting_celeba_ckp.pth.tar', cuda=cuda)
    test(f'trained_mri_fastmri_ckp.pth.tar', cuda=cuda)