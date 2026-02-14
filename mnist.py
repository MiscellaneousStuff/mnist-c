import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from mlp import MLP, cross_entropy_loss

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # (batch, 1, 28, 28) → (batch, 784)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(x, dim=1)
        return x


def train(model, loader, lr=1e-3):
    # losses = []
    for batch_idx, (data, target) in enumerate(loader):
        batch_count = data.shape[0]
        losses = []
        for b in range(batch_count):
            x = data[b]
            x = x.view(x.size(0), -1)[0]  # (batch, 1, 28, 28) → (batch, 784)
            # print("inp.shape:", x.shape)
            pred = model(x)
            y = F.one_hot(target[b], num_classes=10).float()  # 7 → [0,0,0,0,0,0,0,1,0,0]
            
            # loss val
            loss = cross_entropy_loss(pred, y)
            
            losses.append(loss)
            # print("pred.shape, y.shape:", pred.shape, y.shape, target.shape, target)
            model.backward(x, pred, y, lr=lr)
        
        print(batch_idx, np.mean(losses))
    plt.plot(losses)

    # mean_loss = torch.mean(losses)
    # mean_losses.append(mean_loss)

def test(model, loader):
    correct, total = 0, 0
    for data, target in loader:
        for b in range(data.shape[0]):
            x = data[b].view(-1)
            pred = model(x)
            if torch.argmax(pred) == target[b]:
                correct += 1
            total += 1
    print(f"Accuracy: {correct}/{total} ({100.0 * correct / total:.2f}%)")


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    model = MLP(784, 128, 10)

    for epoch in range(1):
        train(model, train_loader)
        test(model, test_loader)

if __name__ == "__main__":
    main()