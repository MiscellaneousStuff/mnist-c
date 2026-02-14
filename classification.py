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
    ([1, 1], [0, 1])
]

X = [pair[0] for pair in XY]
Y = [pair[1] for pair in XY]

X = torch.tensor(X, dtype=torch.float)
Y = torch.tensor(Y, dtype=torch.float)

weights = torch.rand(2, 2, dtype=torch.float) # (row, col) +> (outputs, inputs)

lr = 1e-2
epochs = 100
for e in range(epochs):
    # pred
    z = X[1] @ weights.T
    # print(z)
    pred = softmax(z)
    # print(pred, Y[1])
    
    # loss val
    loss = cross_entropy_loss(pred, Y[1])

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
    weights[0][0] -= (c0_w0_gradient * lr) # class 0, input 0 weight updates
    weights[0][1] -= (c0_w1_gradient * lr) # class 0, input 1 weight updates

    # class 1 weight updates
    weights[1][0] -= (c1_w0_gradient * lr) # class 1, input 0 weight updates
    weights[1][1] -= (c1_w1_gradient * lr) # class 1, input 1 weight updates

    # print(X[1], Y[1], weights, loss, pred, w0_gradient, w1_gradient)

    print(e, pred, loss) # , weights)