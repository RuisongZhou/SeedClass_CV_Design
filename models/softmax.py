import numpy as np
from models.Linear import LinearClassifier

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #循环计算计算loss
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
        scores = X[i] @ W
        scores -= np.max(scores)
        sum_scores = np.sum(np.exp(scores))
        loss -= scores[y[i]]
        loss += np.log(sum_scores)
        for j in range(num_classes):
            dW[:, j] += X[i] * np.exp(scores[j]) / sum_scores
            if j == y[i]:
                dW[:, j] -= X[i]

    dW /= num_train
    dW += reg * W
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    #前向传播
    scores = X @ W
    #指数修正
    scores -= np.max(scores, axis=1,keepdims=True)
    #cross entropy损失函数
    sum_scores = np.sum(np.exp(scores), 1)
    loss -= np.sum(scores[np.arange(num_train), y])
    loss += np.sum(np.log(sum_scores))
    #求softmax概率
    ret = np.zeros(scores.shape)
    ret += np.exp(scores) / sum_scores.reshape(-1, 1)
    # 每一行中只有对应的那个正确类别 = 1，其他都是0
    ret[range(num_train), y] -= 1
    #求导
    dW += X.T @ ret

    dW /= num_train

    dW += reg * W        #加上正则化参数
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    return loss, dW


def multi_loss(W, W1, b, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    dW1 = np.zeros_like(W1)
    db = np.zeros_like(b)
    #矩阵计算loss和梯度

    num_classes = W.shape[1]
    num_train = X.shape[0]
    #前向传播
    hid = X@W1 + b
    avg_hid = np.sum(hid, axis=0)
    hid[hid<0] = 0
    scores = hid @ W
    #指数修正
    scores -= np.max(scores, axis=1,keepdims=True)
    #cross entropy损失函数
    sum_scores = np.sum(np.exp(scores), 1)
    loss -= np.sum(scores[np.arange(num_train), y])
    loss += np.sum(np.log(sum_scores))
    #求softmax概率
    ret = np.zeros(scores.shape)
    ret += np.exp(scores) / sum_scores.reshape(-1, 1)
    # 每一行中只有对应的那个正确类别 = 1，其他都是0
    ret[range(num_train), y] -= 1
    #求导
    dW += hid.T @ ret

    dW /= num_train
    dW += reg * W        #加上正则化参数
    import time

    tmp = ret @ W.T     #num * hid
    tmp[:, avg_hid < 0]= 0

    dW1 += X.T @ tmp
    dW1 /= num_train
    dW1 += reg * W1

    db = np.average(tmp, axis=0)
    db += reg * b

    loss /= num_train
    #loss += 0.5 * reg * np.sum(W * W)
    loss += 0.5 * reg * np.sum(W * W) + 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(b * b)
    return loss, dW, dW1, db


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return multi_loss(self.W, self.W2, self.b, X_batch, y_batch, reg)