import torch
from torchvision import transforms
import random
from torch.utils.data import Dataset
import os
import numpy as np


class makeDataset(Dataset):
    def __init__(self, transform=None, mode='train', alignB=False, sameTransformB=False):
        self.transform = transform
        self.mode = mode
        self.alignB = alignB
        self.sameTransformB = sameTransformB
        self.random_idxs = np.arange(1, self.__len__() + 1)
        np.random.shuffle(self.random_idxs)
        self.random_idxs = list(self.random_idxs)

    def __len__(self):
        if self.mode == 'train':
            return 3839
        else:
            return 421

    def __getitem__(self, idx):
        root = os.path.join('.', 'data', self.mode)
        img_fd = torch.unsqueeze(torch.tensor(np.load(os.path.join(root, 'fd', '{}.npy'.format(idx + 1)))), dim=0)
        if self.alignB:
            img_qd = torch.unsqueeze(torch.tensor(np.load(os.path.join(root, 'qd', '{}.npy'.format(idx + 1)))), dim=0)
        else:
            if len(self.random_idxs) == 0:
                self.random_idxs = np.arange(1, self.__len__() + 1)
                np.random.shuffle(self.random_idxs)
                self.random_idxs = list(self.random_idxs)
            random_idx = self.random_idxs.pop(0)
            img_qd = torch.unsqueeze(torch.tensor(np.load(os.path.join(root, 'qd', '{}.npy'.format(random_idx)))),
                                     dim=0)

        if self.transform is not None:
            if self.sameTransformB:
                seed = np.random.randint(2 ^ 31)
                torch.manual_seed(seed)
                img_fd = self.transform(img_fd)
                torch.manual_seed(seed)
                img_qd = self.transform(img_qd)
            else:
                seed = np.random.randint(2 ^ 31)
                seed2 = np.random.randint(2 ^ 31)
                torch.manual_seed(seed)
                img_fd = self.transform(img_fd)

                torch.manual_seed(seed2)
                img_qd = self.transform(img_qd)

        return img_fd, img_qd


class makeDataset_for_denoising_unet(Dataset):
    def __init__(self, transform=None, mode='train'):
        self.transform = transform
        self.mode = mode

    def __len__(self):
        if self.mode == 'train':
            return 3839
        else:
            return 421

    def __getitem__(self, idx):
        root = os.path.join('.', 'data', self.mode)
        img_qd = torch.unsqueeze(torch.tensor(np.load(os.path.join(root, 'qd', '{}.npy'.format(idx + 1)))), dim=0)
        noise_target = torch.unsqueeze(torch.tensor(
            np.load(os.path.join(root, 'noise_target', '{}.npy'.format(idx + 1)))), dim=0)
        if self.transform is not None:
            seed = np.random.randint(2 ^ 31)
            torch.manual_seed(seed)
            img_qd = self.transform(img_qd)
            torch.manual_seed(seed)
            noise_target = self.transform(noise_target)
        ret = (img_qd, noise_target)
        if self.mode == 'test':
            img_fd = torch.unsqueeze(torch.tensor(np.load(os.path.join(root, 'fd', '{}.npy'.format(idx + 1)))), dim=0)
            ret = (img_qd, img_fd, noise_target)
        return ret
