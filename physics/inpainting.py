import os
import torch

def get_group_inpainting_ops(mask_rate, img_heigth, img_width, device, G=40):
    if G > 1 and G<=100: # maximum G=100
        ipt_group=[]
        for g in range(G):
            mask_path = f'../physics/mask{img_heigth}x{img_width}_random{mask_rate}_G{100}_g{g}.pt'
            if not os.path.exists(mask_path):
                mask = torch.ones(img_heigth, img_width, device=device)
                mask[torch.rand_like(mask) > 1 - mask_rate] = 0
                torch.save(mask, mask_path)
            ipt_group.append(Inpainting(img_heigth, img_width, mask_rate, device, mask_path))
        return ipt_group
    else:
        return Inpainting(img_heigth, img_width, mask_rate, device)

class Inpainting():
    def __init__(self, img_heigth=256, img_width=256, mask_rate=0.3, device='cuda:0', mask_path='mask_filepath'):
        self.name = 'inpainting'

        if os.path.exists(mask_path):
            mask = torch.load(mask_path)
            self.mask = mask.to(device)
        else:
            self.mask = torch.ones(img_heigth, img_width, device=device)
            self.mask[torch.rand_like(self.mask) > 1 - mask_rate] = 0
            torch.save(self.mask, mask_path)

    def A(self, x):
        return torch.einsum('kl,ijkl->ijkl', self.mask, x)

    def A_dagger(self, x):
        return torch.einsum('kl,ijkl->ijkl', self.mask, x)

    def A_adjoint(self, x):
        return self.A_dagger(x)


def get_group_inpainting_mnist_ops(d, D, img_shape, G, device='cuda:0'):
    A_group = []
    for g in range(G):
        A_group.append(Inpainting_MNIST(d, D, img_shape, G=G, g=g, device=device))
    return A_group


class Inpainting_MNIST():
    def __init__(self, d, D, img_shape, G=None, g=None, device='cuda:0'):
        self.name = 'inpainting'
        self.img_shape = img_shape

        C,H,W = img_shape
        mask_rate = 1 - d/D # mask_rate: number(%) of pixels will be deleted

        fname = '../physics/forw_ipt_{}x{}_G{}_g{}.pt'.format(d, D, G, g)

        if os.path.exists(fname):
            self.mask= torch.load(fname, map_location=device)
            self.mask = self.mask.to(device)
        else:
            self.mask = torch.ones(C*H*W, 1, device=device)
            self.mask[torch.rand_like(self.mask) > 1 - mask_rate] = 0
            torch.save(self.mask, fname)
            print('Inpainting MASK matrix has been CREATED & SAVED at {}'.format(fname))


    def A(self, x):
        N,C,H,W = x.shape
        x = x.reshape(N, -1) #N, C*H*W
        y = torch.einsum('kl,nk->nk', self.mask, x)
        return y

    def A_dagger(self, y):
        N = y.shape[0]
        C, H, W = self.img_shape
        x = torch.einsum('kl,nk->nk', self.mask, y)
        x = x.reshape(N, C, H, W)
        return x

    def A_T(self, y):
        return self.A_dagger(y)