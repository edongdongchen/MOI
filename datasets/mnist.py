import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms, datasets

import numpy as np


def  multi_op_mnist_dataloader(mode='train', batch_size=20, shuffle=True, G=24):
    def get_group_datalaer(train=True, shuffle=True):
        data_set = datasets.MNIST(root='../datasets/',
                                  train=train,
                                  download=True,
                                  transform=transforms.Compose([transforms.ToTensor()]))

        group_size = int(np.floor(len(data_set) / G))

        indices = torch.randperm(len(data_set)) if shuffle \
            else torch.arange(0, len(data_set))

        data_loader_group = []

        for g in range(G):
            if g == G-1:
                subset = torch.utils.data.Subset(data_set, indices[g * group_size:])
            else:
                subset = torch.utils.data.Subset(data_set, indices[g * group_size: (g + 1) * group_size])

            data_loader_group.append(torch.utils.data.DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=4))

        return data_loader_group

    # return get_dataloader_group(train=True if mode=='train' else False, shuffle=shuffle)

    if mode=='train':
        return get_group_datalaer(train=True, shuffle=shuffle)
    if mode=='test':
        return get_group_datalaer(train=False, shuffle=shuffle)
