import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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


def train(model, device, loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = F.nll_loss(model(data), target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(loader.dataset)}] Loss: {loss.item():.6f}")


def test(model, device, loader):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            correct += output.argmax(dim=1).eq(target).sum().item()
    test_loss /= len(loader.dataset)
    acc = 100.0 * correct / len(loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({acc:.2f}%)\n")


def main():
    device = "mps"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

if __name__ == "__main__":
    main()