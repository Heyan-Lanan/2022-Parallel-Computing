import gzip
import pickle
import random
from urllib import request

import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod

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
        # urllib.request.urlopen(url).read()
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
    with open("./test1/mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


def MakeOneHot(Y, D_out):  # 热独码，返回64*10，每行只有y相应的位是1，其余是0
    N = Y.shape[0]
    # print(Y)
    Z = np.zeros((N, D_out))
    # print(Z)
    Z[np.arange(N), Y] = 1
    # print(Z)
    return Z


def draw_losses(losses):
    t = np.arange(len(losses))
    plt.plot(t, losses)
    plt.show()


def get_batch(X, Y, batch_size):
    N = len(X)
    i = random.randint(1, N - batch_size)
    return X[i:i + batch_size], Y[i:i + batch_size]


class FC():
    """
    Fully connected layer，全连接层
    """

    def __init__(self, D_in, D_out):
        # print("Build FC")
        self.cache = None
        # self.W = {'val': np.random.randn(D_in, D_out), 'grad': 0}
        self.W = {'val': np.random.normal(0.0, np.sqrt(2 / D_in), (D_in, D_out)), 'grad': 0}  # D_in*D_out 正态分布，均值标准差，范围
        # print(self.W['val'].shape)
        self.b = {'val': np.random.randn(D_out), 'grad': 0}  # D_out,返回D_out个标准正态分布数据

    def _forward(self, X):  # X:64*(784)
        # print("FC: _forward")
        out = np.dot(X, self.W['val']) + self.b['val']  # 样本数*D_out,b加到每一行
        # print(out.shape)
        self.cache = X
        return out

    def _backward(self, dout):
        # print("FC: _backward")
        X = self.cache
        # print(np.dot(dout, self.W['val'].T).shape[0])
        # print(np.dot(dout, self.W['val'].T).shape[1])
        # print(X.shape[0])
        # print(X.shape[1])
        dX = np.dot(dout, self.W['val'].T).reshape(X.shape)  # 64*D_out D_out*D_in
        # print(X.shape[1])
        # print(np.prod(X.shape[1:]))
        self.W['grad'] = np.dot(X.T, dout)  # D_in*64 64*D_out
        self.b['grad'] = np.sum(dout, axis=0)
        # print(dout.shape[0])
        # print(dout.shape[1])
        self._update_params()
        return dX

    def _update_params(self, lr=0.001):
        # Update the parameters
        self.W['val'] -= lr * self.W['grad']
        self.b['val'] -= lr * self.b['grad']


class ReLU():
    """
    ReLU activation layer 整流线性单元激活层
    """

    def __init__(self):
        # print("Build ReLU")
        self.cache = None

    def _forward(self, X):
        # print("ReLU: _forward")
        out = np.maximum(0, X)  # 逐元素比较两个array的大小，返回大的那个，取绝对值
        self.cache = X
        return out

    def _backward(self, dout):
        # print("ReLU: _backward")
        X = self.cache
        # print(X.shape[0])
        # print(X.shape[1])
        # print(dout.shape[0])
        # print(dout.shape[1])
        dX = np.array(dout, copy=True)
        dX[X <= 0] = 0
        return dX


class Softmax():
    """
    Softmax activation layer,Softmax 活化层
    """

    def __init__(self):
        # print("Build Softmax")
        self.cache = None

    def _forward(self, X):
        # print("Softmax: _forward")

        maxes = np.amax(X, axis=1)  # 每个样本的最大值, 64
        maxes = maxes.reshape(maxes.shape[0], 1)  # 64*1
        Y = np.exp(X - maxes)  # 64*10
        Z = Y / np.sum(Y, axis=1).reshape(Y.shape[0], 1)  # 64*10
        self.cache = (X, Y, Z)
        return Z  # distribution

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
    """
    Negative log likelihood loss
    """
    loss = 0.0
    N = Y_pred.shape[0]
    M = np.sum(Y_pred * Y_true, axis=1)  # 64*10 64*10
    # print(M)
    for e in M:
        # print(e)
        if e == 0:
            loss += 500
        else:
            loss += -np.log(e)
    return loss / N


class CrossEntropyLoss():
    def __init__(self):
        pass

    def get(self, Y_pred, Y_true):
        N = Y_pred.shape[0]
        # print(N)
        softmax = Softmax()
        prob = softmax._forward(Y_pred)  # 64*10
        loss = NLLLoss(prob, Y_true)
        Y_serial = np.argmax(Y_true, axis=1)  # 最大值索引
        # print(Y_true)
        dout = prob.copy()
        # print(dout)
        dout[np.arange(N), Y_serial] -= 1  # 64*10
        # print(dout)
        return loss, dout


class Net(metaclass=ABCMeta):
    # Neural network super class

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, dout):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass


class ThreeLayerNet(Net):

    # Simple 3 layer NN，3层神经网络

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


# class SGDMomentum():
#     def __init__(self, params, lr=0.001, momentum=0.99, reg=0.0):
#         self.l = len(params)
#         self.parameters = params
#         self.velocities = []
#         for param in self.parameters:
#             self.velocities.append(np.zeros(param['val'].shape))
#         self.lr = lr
#         self.rho = momentum
#         self.reg = reg
#
#     def step(self):
#         for i in range(self.l):
#             self.velocities[i] = self.rho * self.velocities[i] + (1 - self.rho) * self.parameters[i]['grad']
#             self.parameters[i]['val'] -= (self.lr * self.velocities[i] + self.reg * self.parameters[i]['val'])


"""
(1) Prepare Data: Load, Shuffle, Normalization, Batching, Preprocessing
"""

# init()
X_train, Y_train, X_test, Y_test = load()
X_train, X_test = X_train / float(255), X_test / float(255)
X_train -= np.mean(X_train)
X_test -= np.mean(X_test)

batch_size = 64
D_in = 784
D_out = 10

print("batch_size: " + str(batch_size) + ", D_in: " + str(D_in) + ", D_out: " + str(D_out))

### TWO LAYER NET FORWARD TEST ###
# H=400
# model = nn.TwoLayerNet(batch_size, D_in, H, D_out)
H1 = 300
H2 = 100
model = ThreeLayerNet(batch_size, D_in, H1, H2, D_out)
# print(model.get_params()[0]['grad'])
# print(model.get_params()[0]['val'].shape)
# print(model.get_params()[1]['grad'])
# print(model.get_params()[1]['val'].shape)
# print(model.get_params()[2]['grad'])
# print(model.get_params()[2]['val'].shape)
# print(model.get_params()[3]['grad'])
# print(model.get_params()[3]['val'].shape)
# print(model.get_params()[4]['grad'])
# print(model.get_params()[4]['val'].shape)
# print(model.get_params()[5]['grad'])
# print(model.get_params()[5]['val'].shape)

losses = []
# optim = optimizer.SGD(model.get_params(), lr=0.0001, reg=0)
# optim = SGDMomentum(model.get_params(), lr=0.0001, momentum=0.80, reg=0.00003)
criterion = CrossEntropyLoss()  # 交叉熵损失

# TRAIN
ITER = 1000
for i in range(ITER):
    # get batch, make onehot
    X_batch, Y_batch = get_batch(X_train, Y_train, batch_size)  # 抽样，X_batch: 64*784，Y_batch: 64
    # print(X_batch.shape)
    # print(Y_batch.shape)

    Y_batch = MakeOneHot(Y_batch, D_out)  # Y_batch: 64*10
    # print(Y_batch.shape)

    # forward, loss, backward, step
    Y_pred = model.forward(X_batch)  # X_batch: 64*(28*28)

    loss, dout = criterion.get(Y_pred, Y_batch)  # Y_pred: 64*10
    # print(dout)
    model.backward(dout)  # dout:64*10

    # optim = SGDMomentum(model.get_params(), lr=0.0001, momentum=0.80, reg=0.00003)
    # optim.step()
    # model.set_params(optim.parameters)


    if i % 100 == 0:
        # print(X_batch)
        print("%s%% iter: %s, loss: %s" % (100 * i / ITER, i, loss))
        losses.append(loss)

# save params
weights = model.get_params()
with open("weights.pkl", "wb") as f:
    pickle.dump(weights, f)

draw_losses(losses)

# TRAIN SET ACC
Y_pred = model.forward(X_train)
result = np.argmax(Y_pred, axis=1) - Y_train
result = list(result)
print("TRAIN--> Correct: " + str(result.count(0)) + " out of " + str(X_train.shape[0]) + ", acc=" + str(
    result.count(0) / X_train.shape[0]))

# TEST SET ACC
Y_pred = model.forward(X_test)
result = np.argmax(Y_pred, axis=1) - Y_test
result = list(result)
print("TEST--> Correct: " + str(result.count(0)) + " out of " + str(X_test.shape[0]) + ", acc=" + str(
    result.count(0) / X_test.shape[0]))
