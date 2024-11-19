#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch import optim
import fire


class ChessDataset(Dataset):
    def __init__(self, path="processed/1301_1M.npz"):
        self.data = np.load(path)
        self.X, self.y = self.data["arr_0"], self.data["arr_1"]
        print("loaded", self.X.shape, self.y.shape)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection to match dimensions if needed
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.skip(x)  # Identity or projection
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)  # Add skip connection


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(5, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Residual blocks
        self.resblock1 = ResidualBlock(32, 64)
        self.resblock2 = ResidualBlock(64, 128)
        self.resblock3 = ResidualBlock(128, 128)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def nparam(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.global_pool(x).squeeze()  # Squeeze spatial dimensions
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))  # Output in [-1, 1]
        return x


def train(
    train_set="processed/1301_1M.npz",
    device="cpu",
    bsize=512,
    epochs=10,
    out_name="model",
):
    """
    Train the neural network model
    :param train_set: path to training set
    :param device: device to train on (cpu or cuda or mps)
    :param bsize: batch size
    :param epochs: number of epochs
    :param out_name: output model name
    :saves: model to models/{out_name}.pth

    Example:
    ./train.py --train_set dataset/train_data.npz --device cuda --bsize 512 --epochs 10 --out_name model
    """
    device = torch.device(device)
    dataset = ChessDataset(path=train_set)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bsize, shuffle=True)

    net = Net().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(net.parameters(), lr=1e-4)

    net.train()
    for epoch in range(epochs):
        run_loss = 0
        for i, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            y_pred = net(X.float())
            loss = criterion(y_pred.squeeze(-1), y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            run_loss += loss.item()
            print(f"Epoch {epoch:02} | Loss {run_loss/(i+1)}", end="\r")
        else:
            print(f"Epoch {epoch:02} | Loss {run_loss/(i+1)}")
            
    torch.save(net.state_dict(), f"models/{out_name}.pth")
    print(f"\nModel saved to models/{out_name}.pth")


if __name__ == "__main__":
    fire.Fire(train)
