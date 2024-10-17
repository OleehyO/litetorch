import sys

sys.path.append("../python")
import litetorch as ltt
import litetorch.nn as nn
import numpy as np
import time
import os
from typing import Optional

np.random.seed(0)
# MY_DEVICE = ltt.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim)
            )
        ),
        nn.ReLU()
    )

def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    return nn.Sequential(
        nn.Linear(dim, hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks)],
        nn.Linear(hidden_dim, num_classes)
    )

def epoch(dataloader, model: nn.Module, opt: Optional[ltt.optim.Optimizer]=None):
    np.random.seed(4)
    model.train() if opt else model.eval()
    loss_acc, err_acc = [], 0.0
    softmax_fn = nn.SoftmaxLoss()
    for X, y in dataloader:
        y_hat = model(X)
        loss: nn.Tensor = softmax_fn(y_hat, y)
        if opt:
            opt.reset_grad()
            loss.backward()
            opt.step()
        loss_acc.append(loss.numpy())
        err_acc += np.sum(y_hat.numpy().argmax(axis=1) != y.numpy())
    return err_acc / len(dataloader.dataset), np.mean(loss_acc)

def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ltt.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    train_set = ltt.data.datasets.MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz", 
                             f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_set = ltt.data.datasets.MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz",
                            f"{data_dir}/t10k-labels-idx1-ubyte.gz")

    train_loader = ltt.data.DataLoader(train_set, batch_size, True)
    test_loader = ltt.data.DataLoader(test_set, batch_size)
    model = MLPResNet(dim=28*28, hidden_dim=hidden_dim, num_classes=10)
    opt = optimizer(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    for _ in range(epochs):
        train_err, train_loss = epoch(train_loader, model, opt)
    test_err, test_loss = epoch(test_loader, model)

    return train_err, train_loss, test_err, test_loss


if __name__ == "__main__":
    train_mnist(data_dir="../data")
