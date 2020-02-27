import torch
from torchvision.datasets.cifar import CIFAR10
import torchvision
import torchvision.transforms as transforms
import os


class cifar():
    def __init__(self, root='./data', transform=None ):
        self.root = os.path.join(root, 'cifar')
        self.classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        if transform == None:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomGrayscale(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
        test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
        self.cifar_train = CIFAR10(self.root, train=True, transform=transform, download=True)
        self.cifar_test = CIFAR10(self.root,transform=test_transform, download=True)
