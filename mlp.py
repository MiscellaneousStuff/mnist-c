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
    ([0, 0], [1, 0]),
    ([0, 1], [0, 1]),
    ([1, 0], [0, 1]),
    ([1, 1], [0, 0])
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
        # self.b = torch.rand(out_dim, dtype=torch.float)
    def forward(self, x):
        return self.w.T @ x # + self.b
    def __call__(self, x):
        return self.forward(x)
    def update(self):
        pass

class Model:
    def __init__(self):
        self.fc = Linear(in_dim=2, out_dim=2)
    def forward(self, x):
        x = self.fc(x)
        x = softmax(x)
        return x
    def __call__(self, x):
        return self.forward(x)
    def backward(self):
        # update
        # [0.1, 0.9] - [0, 1] := [-0.1, +0.1]
        d_loss_over_d_z = softmax_deriv(pred, Y[1]) # (class_0, class_1)
        c0_w0_gradient = d_loss_over_d_z[0] * X[1][0]
        c0_w1_gradient = d_loss_over_d_z[0] * X[1][1]
        c1_w0_gradient = d_loss_over_d_z[1] * X[1][0]
        c1_w1_gradient = d_loss_over_d_z[1] * X[1][1]

        print(
            "gradients:\n",
            c0_w0_gradient,
            c0_w1_gradient,
            c1_w0_gradient,
            c1_w1_gradient,
        )

        # class 0 weight updates
        self.fc.w[0][0] -= (c0_w0_gradient * lr) # class 0, input 0 weight updates
        self.fc.w[0][1] -= (c0_w1_gradient * lr) # class 0, input 1 weight updates

        # class 1 weight updates
        self.fc.w[1][0] -= (c1_w0_gradient * lr) # class 1, input 0 weight updates
        self.fc.w[1][1] -= (c1_w1_gradient * lr) # class 1, input 1 weight updates

model = Model()
lr = 1e-2
epochs = 100
for e in range(epochs):
    # pred
    z = model(X[1])
    pred = softmax(z)
    
    # loss val
    loss = cross_entropy_loss(pred, Y[1])

    model.backward()
    # print(X[1], Y[1], weights, loss, pred, w0_gradient, w1_gradient)

    print(e, pred, loss) # , weights)