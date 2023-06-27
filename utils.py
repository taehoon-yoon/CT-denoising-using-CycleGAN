import torch
import random


class Buffer:
    def __init__(self, buffer_size=50):
        self.buffer = []
        self.buffer_size = buffer_size
        self.cur_size = 0

    def push_pop(self, image):
        return_img = []
        for img in image.data:
            img = torch.unsqueeze(img, 0)
            if self.cur_size < self.buffer_size:
                self.cur_size += 1
                return_img.append(img)
                self.buffer.append(img)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    idx = random.randint(0, self.buffer_size - 1)
                    old_img = self.buffer[idx].clone()
                    return_img.append(old_img)
                    self.buffer[idx] = img
                else:
                    return_img.append(img)
        return torch.cat(return_img, dim=0)


class LambdaLR():
    def __init__(self, n_epochs, decay_start_epoch=None, offset=0):

        self.n_epochs = n_epochs
        self.offset = offset
        if decay_start_epoch is None:
            self.decay_start_epoch = int(n_epochs / 2)
        else:
            self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
