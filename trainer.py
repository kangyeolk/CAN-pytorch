import os
import torch
import datetime
import time

import torch.nn as nn
from torchvision.utils import save_image

from generator import *
from discriminator import *
from utils import *



class Trainer(object):
    def __init__(self, data_loader, config):

        ## config setting
        self.data_loader = data_loader

        # Model hyper-get_parameters
        self.model = config.model
        self.img_size = config.img_size
        self.z_size = config.z_size
        self.n_class = config.n_class

        # Training setting
        self.n_steps = config.n_steps
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.beta0 = config.beta0
        self.beta1 = config.beta1
        self.slope = config.slope

        # Path
        self.img_rootpath = config.img_rootpath
        self.log_path = config.log_path
        self.model_save_path = config.log_path
        self.sample_path = config.sample_path

        # Step size
        self.log_step = config.log_step
        self.sample_step = config.sample_step
        self.model_save_step = config.model_save_step

        # Tensor-board?
        self.use_tensorboard = config.use_tensorboard

        # generator & discriminator losses
        self.D_loss = 0
        self.G_loss = 0

        self.build_model()

        # if self.use_tensorboard:
        #     self.build_tensorboard()


    def train(self):

        # Data iterator setting
        data_iter = iter(self.data_loader)
        step_per_epoch = len(self.data_loader)
        model_save_step = int(self.model_save_step * step_per_epoch)
        start_time = time.time()

        for step in range(self.n_steps):
            print('step:{}'.format(step+1))
            # self.D.train() #
            # self.G.train() #

            # Real images
            try:
                real_imgs, labels = next(data_iter)
            except:
                data_iter = iter(self.data_loader)
                real_imgs, labels = next(data_iter)
            labels = self.label2onehot(labels)
            labels = labels.long()

            # Fake images / lantent vector from N(0, I)
            zs = torch.randn((self.batch_size, self.z_size))
            fake_imgs = self.G(zs)

            real_r_out, real_c_out = self.D(real_imgs)
            fake_r_out, fake_c_out = self.D(fake_imgs)

            # ======== Train D ======== #
            # minimize [log(D_r(x)) + log(D_c(c_hat=c|x)) + log(1 - D_r(G(z)))]
            D_loss = self.b_loss(real_r_out, torch.ones((self.batch_size,1))) + self.c_loss(real_c_out, labels)
            D_loss = D_loss + self.b_loss(fake_r_out, torch.zeros((self.batch_size,1)))

            self.reset_grad()
            D_loss.backward(retain_graph=True)
            self.d_optimizer.step()
            # ======== Train G ======== #
            # maximize [log(D_r(G(z))) + sum(BCE of each class)]
            G_loss = self.b_loss(fake_r_out, torch.ones((self.batch_size, 1)))
            G_loss = G_loss + self.CrossEntropy_uniform(fake_c_out)

            self.reset_grad()
            G_loss.backward()
            self.g_optimizer.step()


            # Print out log info
            if (step + 1) % self.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], G_step [{}/{}], D_step[{}/{}], D_loss: {:.4f}, G_loss: {:.4f}".
                      format(elapsed, step + 1, self.n_steps, (step + 1), self.n_steps,
                             D_loss.data[0], G_loss.data[0]))

            # Sample images
            if (step + 1) % self.sample_step == 0:
                fake_images = self.G(zs)
                save_image(denorm(fake_images.data),
                           os.path.join(self.sample_path, '{}_fake.png'.format(step + 1)))

            # Saving model / .pth format(Pytorch own serialization mechanism)
            if (step+1) % model_save_step==0:
                torch.save(self.G.state_dict(),
                           os.path.join(self.model_save_path, '{}_G.pth'.format(step + 1)))
                torch.save(self.D.state_dict(),
                           os.path.join(self.model_save_path, '{}_D.pth'.format(step + 1)))




    def build_model(self):

        if self.model == 'can':
            self.G = vanilla_canG(batch_size=self.batch_size, z_size=self.z_size, slope=self.slope)
            self.D = vanilla_canD(batch_size=self.batch_size, n_class=self.n_class, slope=self.slope, img_size=self.img_size)

        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), self.lr, [self.beta0, self.beta1])
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), self.lr, [self.beta0, self.beta1])

        self.b_loss = nn.BCELoss()
        self.c_loss = nn.CrossEntropyLoss()


        ## TODO: g_lr / d_lr + loss define....!!

        print(self.G)
        print(self.D)


    # def build_tensorboard(self):
    #     from logger import Logger
    #     self.logger = Logger(self.log_path)

    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()


    def label2onehot(self, labels):
        uni_labels = labels.unique(sorted=True)
        k = 0
        dic = {}
        for l in uni_labels:
            dic[str(l.item())] = k
            k += 1
        for (i, l) in enumerate(labels):
            labels[i] = dic[str(l.item())]
        return labels


    def CrossEntropy_uniform(self, pred):
        logsoftmax = nn.LogSoftmax(dim=1)
        unif = torch.full((self.batch_size, self.n_class), 1/self.n_class)
        return torch.mean(-torch.sum(unif * logsoftmax(pred), 1))
