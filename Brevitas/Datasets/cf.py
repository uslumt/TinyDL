"""
Created on Wed Mar  3 08:51:57 2021

This module imports Cifar-10 datasets and make them usable for train and also test

@author: uslumt

"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class CF_dataset:
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5, ))])
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    @classmethod
    def get_trainloader(cls):
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=cls.transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
                                              shuffle=True, num_workers=2)
        return trainloader
    
    @classmethod
    def get_testloader(cls):
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=cls.transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=500,
                                             shuffle=False, num_workers=2)
        return testloader
    
    @classmethod
    def imshow(cls, img):
        
        img = img / 2 + 0.5     
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        
# Prove the import was successful   
if __name__ == "__main__":
     
    dataiter = iter(CF_dataset.get_testloader())
    images, labels = dataiter.next()
    CF_dataset.imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % CF_dataset.classes[labels[j]] for j in range(4)))
