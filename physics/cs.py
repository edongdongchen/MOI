import torch
import os
import numpy as np

def get_group_cs_ops(d, D, img_shape, G, dtype=torch.float, device='cuda:0'):
    A_group = []
    for g in range(G):
        A_group.append(CS(d, D, img_shape, G, g, dtype, device))
    return A_group


class CS():
    def __init__(self, d, D, img_shape, G=None, g=None, dtype=torch.float, device='cuda:0'):
        self.img_shape = img_shape

        fname = '../physics/forw_cs_{}x{}_G{}_g{}.pt'.format(d, D, G, g)

        if os.path.exists(fname):
            A, A_dagger = torch.load(fname)
        else:
            A = np.random.randn(d, D) / np.sqrt(D)
            A_dagger = np.linalg.pinv(A)
            torch.save([A, A_dagger], fname)
            print('CS matrix has been CREATED & SAVED at {}'.format(fname))

        self._A = torch.from_numpy(A).type(dtype).to(device)
        self._A_dagger = torch.from_numpy(A_dagger).type(dtype).to(device)
        self._A_adjoint = self._A.t().type(dtype).to(device)


    def A(self, x):
        N,C,H,W = x.shape
        x = x.reshape(N, -1)
        y = torch.einsum('in, mn->im', x, self._A)
        return y

    def A_dagger(self, y):
        N = y.shape[0]
        C, H, W = self.img_shape[0], self.img_shape[1], self.img_shape[2]
        x = torch.einsum('im, nm->in', y, self._A_dagger)
        x = x.reshape(N, C, H, W)
        return x

    def A_adjoint(self, y):
        N = y.shape[0]
        C, H, W = self.img_shape[0], self.img_shape[1], self.img_shape[2]
        x = torch.einsum('im, nm->in', y, self._A_adjoint)  # x:(N, n, 1)
        x = x.reshape(N, C, H, W)
        return x