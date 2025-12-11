# This script defines all the necessary functions and classes for running different architectures
# mainly: Class for loading the CIFAR dataset
# SGD, SGD with momemtum and Adam Optimizers following the specifications given in the assignment
# custom Glogot / Xavier initialization function

# References:
# https://mmuratarat.github.io/2020-05-13/rgb_to_grayscale_formulas
# https://medium.com/the-ml-practitioner/how-to-implement-an-adam-optimizer-from-scratch-76e7b217f1cc
# 

import os
import pickle
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

"""
NOTE - we can also directly load the data like this -
I am following what is given in the assignment instructions

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=512,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=512,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)
"""

# load CIFAR-10 batch
def load_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    data = batch['data'].reshape(-1, 3, 32, 32).astype(np.float32)
    labels = np.array(batch['labels'], dtype=np.int64)
    return data, labels



# custom dataset with manual grayscale
class CIFAR10GrayscaleDataset(Dataset):
    def __init__(self, root, train=True):
        if train:
            batch_files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            batch_files = ["test_batch"]

        data_list, label_list = [], []
        for file in batch_files:
            data, labels = load_batch(os.path.join(root, file))
            data_list.append(data)
            label_list.append(labels)

        self.data = np.concatenate(data_list)
        self.labels = np.concatenate(label_list)

        # convert to grayscale
        self.data = 0.299 * self.data[:,0] + 0.587 * self.data[:,1] + 0.114 * self.data[:,2]
        self.data = self.data / 255.0  # scale to [0,1]
        self.data = self.data[:, np.newaxis, :, :]


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        img = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label



# dataloaders
def get_cifar10_loaders(batch_size=512, num_workers=2, root='./cifar-10-batches-py'):
    train_dataset = CIFAR10GrayscaleDataset(root=root, train=True)
    test_dataset = CIFAR10GrayscaleDataset(root=root, train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True, persistent_workers=True)
    
    return train_loader, test_loader



# =========== define the optimizers similar to what is asked in the assignment ============
class SGD:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr


    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


    def step(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    g_t = p.grad
                    p -= self.lr * g_t



class SGDMomentum:
    def __init__(self, params, lr, beta):
        self.params = list(params)
        self.lr = lr
        self.beta = beta
        self.velocity = [torch.zeros_like(p) for p in self.params]


    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


    def step(self):
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is not None:
                    g_t = p.grad
                    v_t = (1 - self.beta) * self.velocity[i] + self.beta * g_t
                    self.velocity[i] = v_t
                    p -= self.lr * v_t


"""
Note: for Adam, I use the same momentum rule for SGD momentum.

Initially, I just used simplest Adam (as in lecture notes) just simple update rules from
SGD momentum (given in the assignment) and RMS Prop

eg.
def step(self):
    self.t += 1
    with torch.no_grad():
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad

            # update biased first and second moment estimates
            self.mean_g[i] = (1 - self.beta1) * self.mean_g[i] + self.beta1 * g
            self.mean_g2[i] = self.beta2 * self.mean_g2[i] + (1 - self.beta2) * (g * g)

            # parameter update
            p -= self.lr * self.mean_g[i] / torch.sqrt(self.mean_g2[i])


But for certain lrs, that just did not update the loss function at all (was constant at 90%)
So to make it more stable I have also added a bias term and using epsilon.
"""
class Adam:
    def __init__(self, params, lr, beta1, beta2, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mean_g = [torch.zeros_like(p) for p in self.params]
        self.mean_g2 = [torch.zeros_like(p) for p in self.params]
        self.t = 0
        

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()


    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue

                g = p.grad

                # assignment specific
                self.mean_g[i] = (1 - self.beta1) * self.mean_g[i] + self.beta1 * g
                self.mean_g2[i] = self.beta2 * self.mean_g2[i] + (1 - self.beta2) * (g * g)

                # adding bias to denominators
                denom_m = 1.0 - (1.0 - self.beta1) ** self.t
                denom_v = 1.0 - (self.beta2) ** self.t

                # avoid division by zero
                if denom_m == 0.0:
                    mean_g_hat = self.mean_g[i]
                else:
                    mean_g_hat = self.mean_g[i] / denom_m

                if denom_v == 0.0:
                    mean_g2_hat = self.mean_g2[i]
                else:
                    mean_g2_hat = self.mean_g2[i] / denom_v

                p -= self.lr * mean_g_hat / (torch.sqrt(mean_g2_hat) + self.eps)



# helps with Glogot / Xavier initializations
def custom_glogot_normal(tensor: torch.Tensor):
    # make sure dim is more than 2
    # < 2 like for bias set to 0
    if tensor is None:
        return
    if tensor.dim() < 2:
        with torch.no_grad():
            tensor.zero_()
        return

    with torch.no_grad():
        shape = tensor.shape
        # for linear layers
        if tensor.dim() == 2:
            n_in, n_out = shape[1], shape[0]
        else:
            # for conv layers
            cur_size = 1
            for s in shape[2:]:
                cur_size = cur_size * s
            n_in = shape[1] * cur_size
            n_out = shape[0] * cur_size

        # get normal distribution according to the assignment
        std = 1.0 / math.sqrt(n_in + n_out)
        tensor.normal_(0.0, std)