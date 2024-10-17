# litetorch

litetorch is a lightweight implementation of PyTorch,
designed with an interface that mirrors `torch.Tensor` exactly.
This makes it not only an ideal tool for those looking to gain a deeper understanding of PyTorch’s internal workings,
but also a flexible framework for prototyping neural network frameworks.

## Key Features

* **Lightweight Design**: litetorch is streamlined in its architecture, allowing for more accessible exploration of PyTorch's core functionalities. Whether you're a researcher aiming to demystify PyTorch’s internals or a developer working on experimental neural network frameworks, litetorch offers a simple and intuitive environment for testing ideas quickly.

* **Second-Generation Autodiff Principle**: litetorch follows the second-generation autodiff design principles, which extend the computational graph during backpropagation. This not only increases the optimization potential of the computational graph but also supports the computation of higher-order gradients, such as the gradient of a gradient.

## Build

Prerequisites:

  * Python 3.10+
  * Pybind
  * CMake
  * CUDA 11.7+ (for GPU support)

Build the code: please refer to CMakeLists.txt and compile the code in the `src/` directory and place the compiled `.so` file in the `backend_ndarray/` directory.

## Usage

case1: tensor autodiff

  ```python
  import litetorch as ltt

  a = ltt.tensor(2, dtype="float32", requires_grad=True)
  b = ltt.tensor(8, dtype="float32", requires_grad=True)
  c = ltt.tensor(3, dtype="float32", requires_grad=True)
  loss = b**2 - 4*a*c
  loss.backward()

  print(loss)
  print(a.grad, b.grad, c.grad)
  # >>> 40.0
  # >>> -12.0 16.0 -8.0
  ```

case2: a simple mlp resnet

  ```python
  import litetorch as ltt
  from litetorch import nn

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

  mlp_resnet = MLPResNet(10, 100, 3, 10)
  data = ltt.init.randn(10, 10)

  print(mlp_resnet(data))
  # >>> litetorch.Tensor([[-2.6002522   1.8013879  -1.9132146   2.9064531   2.5027244   0.1097846 ...]])
  ```

## Core Components

litetorch reimplements most of PyTorch's essential modules, while reusing parts of the scaffold and test code from cmu10-714. By providing these key components, litetorch allows users to experiment with neural network architectures, learning algorithms, and gradient computation methods in a lightweight and customizable environment. The major components include:

  * `NDArray`: The core computational module of litetorch, responsible for underlying operator calculations.

  * `Tensor`: Full implementation of PyTorch-like tensors for numerical operations and autograd functionality.

  * `Module`: Base class for neural network layers, following PyTorch's modular design.

  * `Initializer`: Tools for weight initialization in neural networks.

  * `DataSet` & `DataLoader`: Supports efficient data loading for model training.

  * `Optimizer`: Provides gradient-based optimization algorithms, similar to those in PyTorch.

### NDArray

Supports both CPU and CUDA backends, with the backend code located in `backend_ndarray/` and the frontend code in `src/`. Most array computation interfaces have been implemented.

### Tensor

The code is located in the ops directory. The following operators are supported (currently only "float32" type is supported):

* add
* negate
* multiply
* divide
* pow
* log
* exp
* relu
* tanh
* summation
* matmul
* transpose
* reshape
* broadcast
* get_item
* permute
* stack
* split
* flip
* dilate
* undilate
* conv (loop-free)

### Module

The code is located in the `nn/` directory, and the following modules are currently supported:

* Identity
* Linear
* Flatten
* ReLU
* Sequential
* SoftmaxLoss
* BatchNorm1d
* BatchNorm2d
* LayerNorm1d
* Dropout
* Residual
* Convolution

### Initializer

The code is located in the `init/` directory, and the following initialization methods are currently supported:

* xavier_uniform
* xavier_normal
* kaiming_uniform
* kaiming_normal

### Data

The code is located in the `data/` directory, where the `Dataset` and `Dataloader` classes are implemented to support loading and iterating over training data. The following data augmentations are supported:

* RandomFlipHorizontal
* RandomCrop

### Optimizer

The code is located in `optim.py`, and the following optimizers are currently supported:

* SGD
* AdamW
