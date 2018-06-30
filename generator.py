import os
import numpy as np

import torch
import torch.nn as nn

class vanilla_canG(nn.Module):
    def __init__(self, batch_size=64, z_size=100, slope=0.2):
        super(vanilla_canG, self).__init__()
        self.batch_size = batch_size
        self.linear = nn.Linear(z_size, 1024*4*4)
        self.main = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            # current (nBathch, 1024, 4, 4)
            nn.ConvTranspose2d(1024, 1024, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            # current(nBathch, 1024, 8, 8)
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            # current(nBathch, 512, 16, 16)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            # current(nBathch, 256, 32, 32)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            # current(nBathch, 128, 64, 64)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=slope, inplace=True),
            # current(nBathch, 64, 128, 128)
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.Tanh()
            # current(nBatch, 3, 256, 256)
        )

    def forward(self, z):
        out = self.linear(z)
        out = out.view(-1, 1024, 4, 4)
        return self.main(out)



# G = vanilla_canG()
# zzz = torch.randn((64, 100))
# out = G(zzz)
# out.size()
#
#
# criterion = nn.BCELoss()
# criterion(o)
