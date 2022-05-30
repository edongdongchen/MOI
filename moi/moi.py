import os
import torch
import numpy as np

from models.ae import AE
from models.unet import UNet

from utils.metric import cal_psnr
from utils.nn import adjust_learning_rate
from utils.logger import AverageMeter, ProgressMeter, get_timestamp

class MOI(object):
    def __init__(self, args):
        super(MOI, self).__init__()

        args.save_path = './ckp/{}'.format('_'.join([get_timestamp(), args.net_name]))
        os.makedirs(args.save_path, exist_ok=True)
        self.args = args

    def train(self, train_loader_group, physics_group):
        # define model
        if self.args.arch == 'ae':
            model = AE(residual=self.args.residual,
                       dim_input=self.args.dim_input,
                       dim_hid=self.args.dim_hid).to(self.args.device)
        if self.args.arch == 'unet':
            model = UNet(in_channels=self.args.in_channels,
                         out_channels=self.args.out_channels,
                         residual=self.args.residual,
                         circular_padding=self.args.circular_padding,
                         cat=self.args.cat).to(self.args.device)

        criterion = torch.nn.MSELoss().to(self.args.device)
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=self.args.lr,
                                     weight_decay=self.args.wd)

        # start training
        for epoch in range(self.args.epochs):
            adjust_learning_rate(optimizer, epoch, self.args.lr, self.args.cos,
                                 self.args.epochs, self.args.schedule)
            closure_moi_epoch(epoch, train_loader_group, model, physics_group,
                              criterion, optimizer, self.args)
            save_model(epoch, model, optimizer, self.args)

def closure_moi_epoch(epoch, train_loader_group, model, physics_group, criterion, optimizer, args):
    losses = AverageMeter('loss', ':.2e')
    losses_mc = AverageMeter('loss_mc', ':.4e')
    losses_coc = AverageMeter('loss_coc', ':.4e')
    psnr = AverageMeter('psnr', ':2.2f')
    meters = [losses, losses_mc, losses_coc, psnr]
    progress = ProgressMeter(args.epochs, meters)

    G = np.random.permutation(args.G)
    for g in G:
        physics_g = physics_group[g]
        train_loader_g = train_loader_group[g]

        physics_rest = [physics_group[t] for t in G if t != g]

        for i, x in enumerate(train_loader_g):
            x = x[0] if isinstance(x, list) else x
            x = x.type(args.dtype).to(args.device)

            y0 = physics_g.A(x)  # generate input measurement y
            x0 = physics_g.A_dagger(y0)  # range input A^+y

            x1 = model(x0)
            y1 = physics_g.A(x1)
            loss_mc = criterion(y1, y0)

            # cross-operator consistency
            physics_j = physics_rest[np.random.permutation(args.G - 1)[0]]
            y2 = physics_j.A(x1)
            x2 = model(physics_j.A_dagger(y2))
            loss_coc = criterion(x2, x1)

            # totall losses
            loss = loss_mc + loss_coc

            # record losses
            losses.update(loss.item())
            losses_mc.update(loss_mc.item())
            losses_coc.update(loss_coc.item())
            psnr.update(cal_psnr(x1, x, complex=args.complex))

            # gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    progress.display(epoch + 1)

def save_model(epoch, model, optimizer, args):
    if (epoch > 0 and epoch % args.ckp_interval == 0) or epoch + 1 == args.epochs:
        state = {'epoch': epoch,
                 'state_dict': model.state_dict(),
                 'optimizer': optimizer.state_dict(),
                 'args':args}
        torch.save(state, os.path.join(args.save_path, 'ckp_{}.pth.tar'.format(epoch)))