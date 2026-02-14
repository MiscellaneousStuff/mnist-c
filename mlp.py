import torch
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

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
    ([1, 1], [0, 1])
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
        self.in_dim = in_dim
        self.out_dim = out_dim
        # self.b = torch.rand(out_dim, dtype=torch.float)
    def forward(self, x):
        return self.w @ x # + self.b
    def __call__(self, x):
        return self.forward(x)
    def backward(self, x, grads, lr): # grads := (class_0, ..., class_n)
        grads = torch.stack([
            (grad * x).detach().clone()
        for grad in grads])
        self.update(grads, lr)

    def update(self, grads, lr):
        self.w -= grads * lr

class Model:
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

class Optim:
    pass

model = Model()
lr = 1e-2
epochs = 100
for e in range(epochs):
    # pred
    pred = model(X[1])
    target = Y[1]
    
    # loss val
    loss = cross_entropy_loss(pred, Y[1])

    model.backward(
        x=X[1],
        pred=pred,
        target=target
    )

    print(e, pred, loss)