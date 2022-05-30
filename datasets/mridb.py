import numpy as np
import torch
from torch.utils.data.dataset import Dataset

def multi_op_mri_dataloader(mode='train', batch_size=1, shuffle=True, num_workers=4, tag=900, G=24):
    data_set = MRIData(mode=mode, tag=tag)
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


class MRIData(Dataset):
    """MRI dataset."""
    def __init__(self, mode='train', root_dir='your local filepath', tag=900):
        x = torch.load(root_dir).squeeze()
        if mode == 'train':
            self.x = x[:tag] # first 900 for training
        if mode == 'test':
            self.x = x[tag:] # last 73 for testing
        # stack a imaginary dimension
        self.x = torch.stack([self.x, torch.zeros_like(self.x)], dim=1)

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return len(self.x)