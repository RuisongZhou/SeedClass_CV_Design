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

    #矩阵计算loss和梯度

    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = X @ W
    scores -= np.max(scores, axis=1,keepdims=True)
    sum_scores = np.sum(np.exp(scores), 1)
    loss -= np.sum(scores[np.arange(num_train), y])
    loss += np.sum(np.log(sum_scores))

    ret = np.zeros(scores.shape)
    ret += np.exp(scores) / sum_scores.reshape(-1, 1)
    ret[range(num_train), y] -= 1

    dW += X.T @ ret

    dW /= num_train
    dW += reg * W
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)
    return loss, dW

class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)