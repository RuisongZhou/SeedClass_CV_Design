import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class preceptron(nn.Module):
    def __init__(self, feature_size, num_classes, hidden_size):
        super(preceptron, self).__init__()

        self.classes = num_classes

        self.layer = nn.Sequential(
            nn.Linear(feature_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        #self.layer = nn.Linear(feature_size, num_classes)



    def forward(self, x):

        y = self.layer(x)
        y = F.softmax(y,dim=1)
        return y


if __name__ == '__main__':
    net = preceptron(10,10,10)
    print(net.parameters)