import torch
from torchvision.datasets.cifar import CIFAR10
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
from sklearn.decomposition import PCA
from skimage import feature as ft
from sklearn.externals import joblib

class cifar():
    def __init__(self, root='./data', transform=None, test_transform=None,feature_channel=256):
        self.root = os.path.join(root, 'cifar')
        self.classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


        self.hog_norm = 'L1-sqrt'
        self.feature_channel = feature_channel
        #load_path_pca = os.path.join(self.root, "cifar_pca_{}_{}.model".format(self.hog_norm, self.pca_channel))

        if transform == None:
            # load pca
            #pca_model = self.load_pca(load_path_pca)
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomGrayscale(),
                    transforms.ToTensor(),
                    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    Flatten(),
                    #HOG(self.hog_norm),
                    #PCA_transform(pca_model),
                ]
            )
            test_transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                    Flatten(),
                    #HOG(self.hog_norm),
                   #PCA_transform(pca_model),
                ]
            )
        self.cifar_train = CIFAR10(self.root, train=True, transform=transform, download=True)
        self.cifar_test = CIFAR10(self.root,train=False,transform=test_transform, download=True)






    def load_pca(self, load_path):
        if os.path.exists(load_path):
            pca = joblib.load(load_path)
            return pca
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ])
        self.cifar_train = CIFAR10(self.root, train=True, transform=transform, download=True)
        self.cifar_test = CIFAR10(self.root, transform=transform, download=True)

        train_img = []
        train_label = []
        test_img = []
        test_label = []
        for i in range(len(self.cifar_train)):
            #print(self.cifar_train[i][0].shape)
            # train_img.append(np.array(self.cifar_train[i][0]))
            features = ft.hog(np.array(self.cifar_train[i][0]).transpose(1, 2, 0),  # input image
                              orientations=16,  # number of bins
                              pixels_per_cell=(8, 8),  # pixel per cell
                              cells_per_block=(4, 4),  # cells per blcok
                              block_norm=self.hog_norm,  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                              transform_sqrt=True,  # power law compression (also known as gamma correction)
                              feature_vector=True,  # flatten the final vectors
                              visualize=False)  # return HOG map
            train_img.append(features)
            break
            #train_label.append(self.cifar_test[i][1])

        for i in range(len(self.cifar_train)):
            features = ft.hog(np.array(self.cifar_test[i][0]).transpose(1, 2, 0),  # input image
                              orientations=16,  # number of bins
                              pixels_per_cell=(8, 8),  # pixel per cell
                              cells_per_block=(4, 4),  # cells per blcok
                              block_norm=self.hog_norm,  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                              transform_sqrt=True,  # power law compression (also known as gamma correction)
                              feature_vector=True,  # flatten the final vectors
                              visualize=False)  # return HOG map
            test_img.append(features)
            #test_label.append(self.cifar_test[i][1])

        pca = PCA(self.pca_channel)
        pca.fit(train_img + test_img)
        joblib.dump(pca, load_path)
        return pca

class HOG(object):
    def __init__(self,block_norm):
        self.block_norm = block_norm

    def __call__(self, img):
        features = ft.hog(np.array(img),  # input image
                         orientations=16,  # number of bins
                         pixels_per_cell=(8, 8),  # pixel per cell
                         cells_per_block=(4, 4),  # cells per blcok
                         block_norm=self.block_norm,  # block norm : str {‘L1’, ‘L1-sqrt’, ‘L2’, ‘L2-Hys’}
                         transform_sqrt=True,  # power law compression (also known as gamma correction)
                         feature_vector=True,  # flatten the final vectors
                         visualize=False)  # return HOG map

        return torch.from_numpy(features).float()

class PCA_transform(object):
    def __init__(self, pca_model):
        self.model = pca_model
    def __call__(self, features):
        return self.model.transform(features)


class Flatten(object):
    def __call__(self, features):
        features = features-255/2
        return features.flatten()




