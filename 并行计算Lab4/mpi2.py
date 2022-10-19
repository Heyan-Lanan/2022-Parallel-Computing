import sys

sys.path.append(r'C:\Users\凝雨\AppData\Local\Programs\Python\Python39\Lib\site-packages')

from mpi4py import MPI
import gzip
import pickle
import random
from urllib import request

import numpy as np
import matplotlib.pyplot as plt

filename = [
    ["training_images", "train-images-idx3-ubyte.gz"],
    ["test_images", "t10k-images-idx3-ubyte.gz"],
    ["training_labels", "train-labels-idx1-ubyte.gz"],
    ["test_labels", "t10k-labels-idx1-ubyte.gz"]
]


def download_mnist():
    base_url = "http://yann.lecun.com/exdb/mnist/"
    for name in filename:
        print("Downloading " + name[1] + "...")
        request.urlretrieve(base_url + name[1], name[1])
    print("Download complete.")


def save_mnist():
    mnist = {}
    for name in filename[:2]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28 * 28)
    for name in filename[-2:]:
        with gzip.open(name[1], 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)
    with open("mnist.pkl", 'wb') as f:
        pickle.dump(mnist, f)
    print("Save complete.")


def init():
    download_mnist()
    save_mnist()


def load():
    with open("mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


def MakeOneHot(Y, D_out):
    N = Y.shape[0]
    Z = np.zeros((N, D_out))
    Z[np.arange(N), Y] = 1
    return Z


def draw_losses(losses):
    t = np.arange(len(losses))
    plt.plot(t, losses)
    plt.show()


def get_batch(X, Y, batch_size):
    N = len(X)
    i = random.randint(1, N - batch_size)
    return X[i:i + batch_size], Y[i:i + batch_size]


class FC:

    def __init__(self, D_in, D_out):
        self.cache = None
        self.W = {'val': np.random.normal(0.0, np.sqrt(2 / D_in), (D_in, D_out)), 'grad': 0}
        self.b = {'val': np.random.randn(D_out), 'grad': 0}

    def _forward(self, X):
        out = np.dot(X, self.W['val']) + self.b['val']
        self.cache = X
        return out

    def _backward(self, dout):
        X = self.cache
        dX = np.dot(dout, self.W['val'].T).reshape(X.shape)  # 64*D_out D_out*D_in
        self.W['grad'] = np.dot(X.T, dout)  # D_in*64 64*D_out
        self.b['grad'] = np.sum(dout, axis=0)
        self._update_params()
        return dX

    def _update_params(self, lr=0.001):
        self.W['val'] -= lr * self.W['grad']
        self.b['val'] -= lr * self.b['grad']


class ReLU:

    def __init__(self):
        self.cache = None

    def _forward(self, X):
        out = np.maximum(0, X)
        self.cache = X
        return out

    def _backward(self, dout):
        X = self.cache
        dX = np.array(dout, copy=True)
        dX[X <= 0] = 0
        return dX


class Softmax:

    def __init__(self):
        self.cache = None

    def _forward(self, X):
        maxes = np.amax(X, axis=1)
        maxes = maxes.reshape(maxes.shape[0], 1)
        Y = np.exp(X - maxes)
        Z = Y / np.sum(Y, axis=1).reshape(Y.shape[0], 1)
        self.cache = (X, Y, Z)
        return Z

    def _backward(self, dout):
        X, Y, Z = self.cache
        dZ = np.zeros(X.shape)
        dY = np.zeros(X.shape)
        dX = np.zeros(X.shape)
        N = X.shape[0]
        for n in range(N):
            i = np.argmax(Z[n])
            dZ[n, :] = np.diag(Z[n]) - np.outer(Z[n], Z[n])
            M = np.zeros((N, N))
            M[:, i] = 1
            dY[n, :] = np.eye(N) - M
        dX = np.dot(dout, dZ)
        dX = np.dot(dX, dY)
        return dX


def NLLLoss(Y_pred, Y_true):
    loss = 0.0
    N = Y_pred.shape[0]
    M = np.sum(Y_pred * Y_true, axis=1)
    for e in M:
        if e == 0:
            loss += 500
        else:
            loss += -np.log(e)
    return loss / N


class CrossEntropyLoss:

    def __init__(self):
        pass

    def get(self, Y_pred, Y_true):
        N = Y_pred.shape[0]
        softmax = Softmax()
        prob = softmax._forward(Y_pred)
        loss = NLLLoss(prob, Y_true)
        Y_serial = np.argmax(Y_true, axis=1)
        dout = prob.copy()
        dout[np.arange(N), Y_serial] -= 1
        return loss, dout


class ThreeLayerNet:

    def __init__(self, N, D_in, H1, H2, D_out, weights=''):
        self.FC1 = FC(D_in, H1)
        self.ReLU1 = ReLU()
        self.FC2 = FC(H1, H2)
        self.ReLU2 = ReLU()
        self.FC3 = FC(H2, D_out)

        if weights == '':
            pass
        else:
            with open(weights, 'rb') as f:
                params = pickle.load(f)
                self.set_params(params)

    def forward(self, X):
        h1 = self.FC1._forward(X)
        a1 = self.ReLU1._forward(h1)
        h2 = self.FC2._forward(a1)
        a2 = self.ReLU2._forward(h2)
        h3 = self.FC3._forward(a2)
        return h3

    def backward(self, dout):
        dout = self.FC3._backward(dout)
        dout = self.ReLU2._backward(dout)
        dout = self.FC2._backward(dout)
        dout = self.ReLU1._backward(dout)
        dout = self.FC1._backward(dout)

    def get_params(self):
        return [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b]

    def set_params(self, params):
        [self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b, self.FC3.W, self.FC3.b] = params


# init()
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:

    X_train, Y_train, X_test, Y_test = load()
    X_train, X_test = X_train / float(255), X_test / float(255)
    X_train -= np.mean(X_train)
    X_test -= np.mean(X_test)

    batch_size = 64
    D_in = 784
    D_out = 10
    print("batch_size: " + str(batch_size) + ", D_in: " + str(D_in) + ", D_out: " + str(D_out))

    H1 = 300
    H2 = 100
    model = ThreeLayerNet(batch_size, D_in, H1, H2, D_out)

    ITER = 200

    for i in range(ITER):

        X_batch, Y_batch = get_batch(X_train, Y_train, batch_size)
        Y_batch = MakeOneHot(Y_batch, D_out)
        Y_pred = model.forward(X_batch)
        # print(Y_batch.shape)

        for j in range(1, size):
            # print(Y_pred[(j - 1) * 32:j * 32])
            Y1 = Y_pred[(j - 1) * 32:j * 32]
            Y2 = Y_batch[(j - 1) * 32:j * 32]

            # print(Y1.shape)
            #
            # print(Y2)
            comm.Send(Y1, dest=j)
            comm.Send(Y2, dest=j)

        loss_sum = 0.0
        dout_list = np.empty((64, 10))
        # dout_np = np.empty((64, 10))
        for j in range(1, size):
            dout = np.empty((32, 10))

            loss = comm.recv()
            # comm.Recv(dout, source=j)

            # print(dout.shape)
            # print(loss_[0])
            loss_sum += loss
            for item in dout:
                dout_list += item
        # print(dout_list)
        print(loss_sum)
        model.backward(dout_list)

        if i % 100 == 0:
            print("%s%% iter: %s, loss: %s" % (100 * i / ITER, i, loss_sum))

    Y_pred = model.forward(X_train)
    result = np.argmax(Y_pred, axis=1) - Y_train
    result = list(result)
    print("TRAIN--> Correct: " + str(result.count(0)) + " out of " + str(X_train.shape[0]) + ", acc=" + str(
        result.count(0) / X_train.shape[0]))

    Y_pred = model.forward(X_test)
    result = np.argmax(Y_pred, axis=1) - Y_test
    result = list(result)
    print("TEST--> Correct: " + str(result.count(0)) + " out of " + str(X_test.shape[0]) + ", acc=" + str(
        result.count(0) / X_test.shape[0]))

else:

    Y1 = np.empty((32, 10))
    Y2 = np.empty((32, 10))
    comm.Recv(Y1, source=0)
    comm.Recv(Y2, source=0)
    # print(Y1)
    # print(Y2)

    loss, dout = CrossEntropyLoss().get(Y1, Y2)

    comm.send(loss, dest=0)
    comm.Send(dout, dest=0)

