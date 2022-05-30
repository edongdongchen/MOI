import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms, datasets

def multi_op_celeba_dataloader(mode='train', batch_size=1,
                           shuffle=True,
                           num_workers=1, crop_size=(512, 512),
                           resize=False, G=40):
    def celeba_dataset(dataset_name='CelebA', mode='train', crop_size=(512, 512), resize=False):
        transform = [transforms.CenterCrop(crop_size)]
        transform.append(transforms.Resize(int(crop_size[0] / 2))) if resize else None
        transform.append(transforms.ToTensor())
        transform = transforms.Compose(transform)
        return datasets.ImageFolder(f'/local_filepath/{dataset_name}/{mode}/',
                                    transform=transform)

    data_set = celeba_dataset(mode=mode, crop_size=crop_size,resize=resize)

    group_size = int(np.floor(len(data_set) / G))
    indices = torch.randperm(len(data_set)) if shuffle else torch.arange(0, len(data_set))

    data_loader_group = []
    for g in range(G):
        if g == G - 1:
            subset = torch.utils.data.Subset(data_set, indices[g * group_size:])
        else:
            subset = torch.utils.data.Subset(data_set, indices[g * group_size: (g + 1) * group_size])
        data_loader_group.append(torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers))
    return data_loader_group