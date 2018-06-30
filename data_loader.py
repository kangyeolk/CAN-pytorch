import os
import torch
from torchvision import transforms
import torchvision.datasets

class dataloader():
    def __init__(self, dataset, img_rootpath, img_size, batch_size):
        self.dataset = dataset
        self.img_rootpath = img_rootpath
        self.img_size = img_size
        self.batch_size = batch_size

    def transform(self, centercrop, totensor, normalize, resize):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(160)) #
        if resize:
            options.append(transforms.Resize((self.img_size,self.img_size)))
        if totensor:
            options.append(transforms.ToTensor())
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def loader(self):
        transforms = self.transform(False, True, True, True) # TODO: centercrop vs Resize?
        trainset = torchvision.datasets.ImageFolder(root=self.img_rootpath, transform=transforms)
        # cls = self.dataset.split()
        # trainset.classes = [ cl for cl in trainset.classes if any(c in cl for c in cls) ]
        # trainset.imgs = [ img for img in trainset.imgs if any(c in img[0] for c in cls) ]
        loader = torch.utils.data.DataLoader(dataset=trainset,
                                             batch_size=self.batch_size,
                                             shuffle=True,
                                             drop_last=True,
                                             num_workers=2)
        return loader
