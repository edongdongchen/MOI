import os
import torch

import fastmri
from fastmri.data import transforms as T
from fastmri.data.subsample import RandomMaskFunc

def get_group_mri_ops(acceleration, device, G=1):
    if G > 1 and G<100:
        mri_group=[]
        for g in range(G):
            mask_path = f'../physics/{acceleration}x_group/mask_{acceleration}x_G{100}_g{g}.pth.tar'
            if not os.path.exists(mask_path):
                mask_func = RandomMaskFunc(center_fractions=[0.04],
                                           accelerations=[acceleration])

                masked_kspace, mask = T.apply_mask(torch.randn(1, 320, 320), mask_func)
                torch.save({'mask': mask}, mask_path)
            mri_group.append(MRI(acceleration, device, mask_path))
        return mri_group
    else:
        return MRI(acceleration, device)


class MRI():
    def __init__(self, acceleration=2, device='cpu', mask_path='filepath'):
        self.name = 'mri'
        self.acceleration = acceleration
        mask = torch.load(mask_path)['mask']

        self.mask = mask.to(device)
        self.mask_func = lambda shape, seed: self.mask
        self.device = device

    def dim_y(self):
        return torch.count_nonzero(self.mask)

    def apply_mask(self, y):
        y, _ = T.apply_mask(y, self.mask_func)
        return y

    def A(self, x, permute=True):
        y = fastmri.fft2c(x.permute(0, 2, 3, 1) if permute else x)
        y, _ = T.apply_mask(y, self.mask_func)
        return y.permute(0,3,1,2) if permute else y

    def A_dagger(self, y, permute=True):
        x = fastmri.ifft2c(y.permute(0,2,3,1) if permute else y)
        return x.permute(0,3,1,2) if permute else x

    def A_adjoint(self, y, permute=True):
        return self.A_dagger(y, permute=permute)