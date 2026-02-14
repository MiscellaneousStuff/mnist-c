import torch
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def sigmoid_deriv(x):
    nexpx = torch.exp(-x)
    return nexpx / (1 + nexpx) ** 2

def softmax(logits):
    log_logits = torch.exp(logits)
    s = torch.sum(log_logits)
    o = log_logits / s
    # print("log_logits:", log_logits, s, o)
    return o

def cross_entropy_loss(pred, target):
    return -torch.sum(target * torch.log(pred))

def softmax_deriv(pred, target):
    return pred - target # [0.1, 0.9] - [0, 1] := [-0.1, +0.1]

# or gate

XY = [
    ([0, 0], [1, 0]), # (binary flag for two inputs), (one hot encoded truth value)
    ([0, 1], [0, 1]),
    ([1, 0], [0, 1]),
    ([1, 1], [1, 0])
]

X = [pair[0] for pair in XY]
Y = [pair[1] for pair in XY]

X = torch.tensor(X, dtype=torch.float)
Y = torch.tensor(Y, dtype=torch.float)

# weights = torch.rand(2, 2, dtype=torch.float) # (row, col) +> (outputs, inputs)

class Layer: pass

class Linear(Layer):
    def __init__(self, in_dim, out_dim):
        self.w = torch.rand(out_dim, in_dim, dtype=torch.float)
        self.b = torch.rand(out_dim, dtype=torch.float)
        self.in_dim = in_dim
        self.out_dim = out_dim
        # self.b = torch.rand(out_dim, dtype=torch.float)
    def forward(self, x):
        return self.w @ x + self.b
    def __call__(self, x):
        return self.forward(x)
    def backward(self, x, grads, lr): # grads := (class_0, ..., class_n)
        w_grads = torch.stack([
            (grad * x).detach().clone()
        for grad in grads])
        self.w -= w_grads * lr
        self.b -= grads.detach().clone() * lr  # dL/db = dL/dz directly

class LinearModel:
    def __init__(self):
        self.fc = Linear(in_dim=2, out_dim=2)
    def forward(self, x):
        x = self.fc(x)
        x = softmax(x)
        return x
    def __call__(self, x):
        return self.forward(x)
    def backward(self, x, pred, target):
        assert pred.shape == target.shape
        grads = softmax_deriv(pred, target) # (class_0, class_1)
        self.fc.backward(x, grads, lr=1e-2)

class XORModel:
    def __init__(self):
        self.fc1 = Linear(in_dim=2, out_dim=2)
        self.fc2 = Linear(in_dim=2, out_dim=2)
        self.activations = {"sigmoid": []}
    def forward(self, x):
        # print("x:", x)
        x = self.fc1(x)
        # print("fc1:", x)
        x = sigmoid(x)
        # print("sigmoid:", x)
        self.activations["sigmoid"].append(x)
        x = self.fc2(x)
        x = softmax(x)
        return x
    def __call__(self, x):
        return self.forward(x)
    def backward(self, x, pred, target, lr=1e-2):
        """
        d_loss / d_z := (pred - target)
        d_z / d_fc2_i := 
        """
        assert pred.shape == target.shape

        # sigmoid activations
        sig_actives = self.activations["sigmoid"].pop(-1)
        # print("sig:", sig_actives)

        # update fc2
        softmax_grads = softmax_deriv(pred, target) # (class_0, class_1) dL/dz2
        
        fc2 = self.fc2
        grad_loss_act_0 = fc2.w[0][0] * softmax_grads[0] + fc2.w[1][0] * softmax_grads[1]
        grad_loss_act_1 = fc2.w[0][1] * softmax_grads[0] + fc2.w[1][1] * softmax_grads[1]
        grad_loss_act = torch.tensor([grad_loss_act_0, grad_loss_act_1])

        self.fc2.backward(sig_actives, softmax_grads, lr=lr)

        grad_loss_z1 = grad_loss_act * sig_actives * (1 - sig_actives)
        self.fc1.backward(x, grad_loss_z1, lr=lr)

model = XORModel()
lr = 1e-2
epochs = 10000
mean_losses = []

print(">>> training")
for e in range(epochs):
    losses = []
    for i in range(4):
        x = X[i]
        y = Y[i]
        # pred
        pred = model(x)
        target = y
        
        # loss val
        loss = cross_entropy_loss(pred, Y[i])
        losses.append(loss)
        model.backward(x, pred, y, lr=lr)
    mean_loss = np.mean(losses)
    mean_losses.append(mean_loss)
    # print(mean_loss)

print(">>> testing")
for i in range(4):
    x = X[i]
    y = Y[i]
    
    pred = model(x)
    target = y

    print(
        x,
        torch.argmax(target),
        torch.argmax(pred),
    )

plt.plot(mean_losses)
plt.show()