"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import litetorch as ltt


def parse_mnist(image_filename, label_filename):
    import numpy as np
    import gzip

    def read_labels(filename):
        with gzip.open(filename, 'rb') as f:
            # Skip the first 8 bytes (magic number and number of labels)
            f.read(8)
            # Read the labels
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    def read_images(filename):
        with gzip.open(filename, 'rb') as f:
            # Skip the first 16 bytes (magic number, number of images, rows, cols)
            f.read(16)
            # Read the images
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
            images = images.astype(np.float32) / 255.0
            images = images.reshape(-1, 28 * 28)
        return images
    return read_images(image_filename), read_labels(label_filename)


def softmax_loss(Z: ltt.Tensor, y_one_hot: ltt.Tensor) -> ltt.Tensor:
    o = ltt.summation(Z * y_one_hot, axes=(1,))
    log_sum = ltt.log(ltt.summation(ltt.exp(Z), axes=(1,)))
    assert log_sum.shape == o.shape
    return ltt.summation(log_sum - o) / o.shape[0]


def nn_epoch(X, y, W1: ltt.Tensor, W2: ltt.Tensor, lr=0.1, batch=100):
    import numpy as np
    num_examples, input_dim = X.shape
    _, hidden_dim = W1.shape
    _, num_classes = W2.shape
    assert W1.shape == (input_dim, hidden_dim)
    assert W2.shape == (hidden_dim, num_classes)

    for idx in range(0, num_examples, batch):
        X_batch = ltt.Tensor(X[idx:idx + batch])  # (batch, input_dim)
        y_batch = y[idx:idx + batch]  # (batch,)
        y_batch_onehot = np.zeros((batch, num_classes))  # (batch, num_classes)
        y_batch_onehot[np.arange(batch), y_batch.astype(int)] = 1
        y_batch_onehot = ltt.Tensor(y_batch_onehot)

        z = ltt.relu(X_batch @ W1) @ W2  # (batch, num_classes)
        loss = softmax_loss(z, y_batch_onehot)
        loss.backward()

        W1_grad, W2_grad = W1.grad.detach(), W2.grad.detach()
        W1.data -= lr * W1_grad.data
        W2.data -= lr * W2_grad.data

        assert z.shape == y_batch_onehot.shape == (batch, num_classes)
        assert X_batch.shape == (batch, input_dim)
    return (W1.data, W2.data)


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ltt.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
