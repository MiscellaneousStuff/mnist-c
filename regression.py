import torch
import random
import numpy as np

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def mse(pred, target): return (pred - target) ** 2

XY = [
    ([1, 2], [3]),
    ([4, 5], [9]),
    ([123, 221], [344])
]

X = [pair[0] for pair in XY]
Y = [pair[1] for pair in XY]

X = torch.tensor(X, dtype=torch.float)
Y = torch.tensor(Y, dtype=torch.float)

weights = torch.rand(1, 2, dtype=torch.float) # (row, col)

lr = 1e-2
epochs = 100
for _ in range(epochs):
    # pred
    pred = X[0] @ weights.T
    
    # loss val
    loss = mse(pred, Y[0])

    # update
    w0_gradient = (2 * (pred - Y[0])) * weights[0][0]
    w1_gradient = (2 * (pred - Y[0])) * weights[0][1]
    weights[0][0] -= (w0_gradient[0] * lr)
    weights[0][1] -= (w1_gradient[0] * lr)

    print(X[0], Y[0], weights, loss, pred, w0_gradient, w1_gradient)

# d_loss / d_weight := d_loss / d_pred * d_pred / d_weight

# d_loss / d_pred := 2 (d_loss) * (pred - target) (d_pred)
# d_pred / d_w1 := x1
# d_pred / d_w2 := x2

# pred = 2
# target = 1.9
# w1_delta = (2 * (pred - target)) * (0.5)
# w2_delta = (2 * (pred - target)) * (0.5)