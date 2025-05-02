import numpy as np
import torch
import torch.nn.functional as F
import utils as ut
from torch import autograd, nn, optim
from torch.nn import functional as F



class Reshape(torch.nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
    def forward(self, x):
        return x.reshape(x.size(0), *self.shape)


class ReLU(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.maximum(x, torch.zeros_like(x))



class Generator(torch.nn.Module):
    def __init__(self, z_dim=64, num_channels=1):
        super().__init__()
        self.z_dim = z_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(z_dim, 512),
            torch.nn.BatchNorm1d(512),
            ReLU(),

            torch.nn.Linear(512, 64 * 7 * 7),
            torch.nn.BatchNorm1d(64 * 7 * 7),
            ReLU(),
            Reshape(64, 7, 7),

            torch.nn.PixelShuffle(2),
            torch.nn.Conv2d(64 // 4, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            ReLU(),

            torch.nn.PixelShuffle(2),
            torch.nn.Conv2d(32 // 4, num_channels, kernel_size=3, padding=1),
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(torch.nn.Module):
    def __init__(self, num_channels=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            ReLU(),

            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            ReLU(),

            Reshape(64 * 7 * 7),
            torch.nn.Linear(64 * 7 * 7, 512),
            ReLU(),

            torch.nn.Linear(512, 1),
            Reshape()
        )

    def forward(self, x):
        return self.net(x)



class Encoder(nn.Module):
    def __init__(self, z_dim, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(784 + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 2 * z_dim),
        )

    def encode(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        h = self.net(xy)
        m, v = ut.gaussian_parameters(h, dim=1)
        return m, v

class Decoder(nn.Module):
    def __init__(self, z_dim, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 784)
        )

    def decode(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(zy)

class Classifier(nn.Module):
    def __init__(self, y_dim):
        super().__init__()
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(784, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, y_dim)
        )

    def classify(self, x):
        return self.net(x)
