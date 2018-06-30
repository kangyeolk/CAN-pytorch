import os
import numpy as np
import argparse

import torch
import torch.nn as nn
from torchvision import transforms as transforms

class vanilla_canD(nn.Module):
    def __init__(self, batch_size=64, D_dim=32, n_class=2, slope=0.2, img_size=256):
        super(vanilla_canD, self).__init__()
        self.batch_size=batch_size
        self.img_size=img_size
        self.D_dim=D_dim
        self.conv6 = nn.Sequential(
            # image input size 3X256X256 by default
            nn.Conv2d(3, D_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(D_dim),
            nn.LeakyReLU(negative_slope=slope),

            nn.Conv2d(D_dim, D_dim*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(D_dim*2),
            nn.LeakyReLU(negative_slope=slope),

            nn.Conv2d(D_dim*2, D_dim*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(D_dim*4),
            nn.LeakyReLU(negative_slope=slope),

            nn.Conv2d(D_dim*4, D_dim*8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(D_dim*8),
            nn.LeakyReLU(negative_slope=slope),

            nn.Conv2d(D_dim*8, D_dim*16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(D_dim*16),
            nn.LeakyReLU(negative_slope=slope),

            nn.Conv2d(D_dim*16, D_dim*16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(D_dim*16),
            nn.LeakyReLU(negative_slope=slope))

        # Linear layer for determining real/fake
        self.disc = nn.Sequential(
            nn.Linear((D_dim*16)*(img_size/64)*(img_size/64), 1),
            nn.Sigmoid())

        # Linear layers for classify the image
        self.classify = nn.Sequential(
            nn.Linear((D_dim*16)*(img_size/64)*(img_size/64), 1024),
            nn.LeakyReLU(negative_slope=slope),

            nn.Linear(1024, 512),
            nn.LeakyReLU(negative_slope=slope),

            nn.Linear(512, n_class)
            # nn.Softmax(dim=1))
            )
    def forward(self, x):
        x = self.conv6(x)
        con_x = x.view(-1, (self.D_dim*16)*(self.img_size/64)*(self.img_size/64))
        r_out = self.disc(con_x)
        c_out = self.classify(con_x)
        return r_out, c_out

#
#
#
# ############################################################################################
