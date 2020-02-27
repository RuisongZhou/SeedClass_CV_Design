import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class preceptron(nn.Module):
    def __init__(self, feature_size, num_classes, hidden_size, device):
        super(preceptron, self).__init__()

        self.classes = num_classes
        # self.w = np.ones([len(feature_size), len(classes)], dtype=np.float32)
        # self.b = 0
        # self.l_rate = 0.1
        # self.data = data
        self.W1 = torch.tensor(np.random.normal(0, 1, (feature_size, hidden_size)), dtype=torch.float32,
                                requires_grad=True, device=device)
        self.b1 = torch.zeros(hidden_size, dtype=torch.float32,requires_grad=True, device=device)
        self.W2 = torch.tensor(np.random.normal(0, 1, (hidden_size, num_classes)), dtype=torch.float32,
                               requires_grad=True, device=device)
        self.b2 = torch.zeros(num_classes, dtype=torch.float32,requires_grad=True, device=device)

        self.params = [self.W1, self.b1, self.W2, self.b2]

        # self.layer = nn.Sequential(
        #     nn.Linear(feature_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, num_classes),
        # )
    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = F.relu(torch.matmul(x, self.W1) + self.b1)
        y = torch.matmul(y, self.W2) + self.b2
        y = F.softmax(y,dim=1)
        return y


    # 随机梯度下降法
    # def fit(self, X_train, y_train):
    #     is_wrong = False
    #     while not is_wrong:
    #         wrong_count = 0
    #         for d in range(len(X_train)):
    #             X = X_train[d]
    #             y = y_train[d]
    #             if y * self.sign(X, self.w, self.b) <= 0:
    #                 self.w = self.w + self.l_rate * np.dot(y, X)
    #                 self.b = self.b + self.l_rate * y
    #                 wrong_count += 1
    #         if wrong_count == 0:
    #             is_wrong = True
    #
    # def score(self):
    #     pass

if __name__ == '__main__':
    net = preceptron(10,10,10)
    print(net.parameters())