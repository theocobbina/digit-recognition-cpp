"""Train the same 2-layer network in PyTorch, as a check on the from-scratch C++ version.

Same architecture (784 -> 128 -> ReLU -> 10), loss, optimizer, batch size, learning rate
and epoch count as src/nn/train.cpp. Writes results/pytorch.csv.
Run after scripts/download_mnist.py.
"""

import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

root = Path(__file__).resolve().parent.parent
data = root / "data"


def images(name):
    raw = np.fromfile(data / name, dtype=np.uint8, offset=16)  # skip the 16-byte IDX header
    return torch.tensor(raw.reshape(-1, 784) / 255.0, dtype=torch.float32)


def labels(name):
    raw = np.fromfile(data / name, dtype=np.uint8, offset=8)  # skip the 8-byte IDX header
    return torch.tensor(raw.astype(np.int64))


X_train, y_train = images("train-images-idx3-ubyte"), labels("train-labels-idx1-ubyte")
X_test, y_test = images("t10k-images-idx3-ubyte"), labels("t10k-labels-idx1-ubyte")

torch.manual_seed(42)
model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()
batch, epochs = 64, 5


def accuracy(X, y):
    with torch.no_grad():
        return 100.0 * (model(X).argmax(dim=1) == y).float().mean().item()


rows = ["epoch,loss,train_acc,test_acc,seconds"]
print(f"{'epoch':>5} {'loss':>8} {'train acc':>10} {'test acc':>10} {'time(s)':>8}")
for epoch in range(1, epochs + 1):
    start = time.time()
    order = torch.randperm(len(X_train))
    batches = len(X_train) // batch  # drop the last partial batch, as train.cpp does
    total = 0.0
    for b in range(batches):
        idx = order[b * batch : (b + 1) * batch]
        loss = loss_fn(model(X_train[idx]), y_train[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
    secs = time.time() - start
    train_acc, test_acc = accuracy(X_train, y_train), accuracy(X_test, y_test)
    print(f"{epoch:5d} {total / batches:8.4f} {train_acc:9.2f}% {test_acc:9.2f}% {secs:8.1f}")
    rows.append(f"{epoch},{total / batches:.4f},{train_acc:.2f},{test_acc:.2f},{secs:.2f}")

(root / "results" / "pytorch.csv").write_text("\n".join(rows) + "\n")
print("\nWrote results/pytorch.csv")
