import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


# 根据路径，读取npy返回dataloader
def get_dataloader(path, batch_size, shuffle=True, device='cpu'):
    data = np.load(path)
    data = torch.from_numpy(data).float().to(device)
    # dataset = TensorDataset(data)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return dataloader