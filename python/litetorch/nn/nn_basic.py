"""The module.
"""
from typing import List
from litetorch.autograd import Tensor
from litetorch import ops
import litetorch.init as init


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []

def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.bias = None
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, nonlinearity="relu", device=device, dtype=dtype))
        if bias:
            self.bias = init.kaiming_uniform(out_features, 1, nonlinearity="relu", device=device, dtype=dtype)
            self.bias.data = self.bias.data.reshape((1, out_features))
            self.bias = Parameter(self.bias)

    def forward(self, X: Tensor) -> Tensor:
        res = X @ self.weight
        return res + ops.broadcast_to(self.bias, res.shape) if self.bias else res


class Flatten(Module):
    def forward(self, X):
        return ops.reshape(X, (X.shape[0], -1))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        res = x
        for m in self.modules:
            res = m(res)
        return res


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        logsumexp = ops.logsumexp(logits, axes=-1)

        nclass = logits.shape[-1]
        y_onehot = init.one_hot(nclass, y)
        assert y_onehot.shape == logits.shape
        logits_label = ops.summation(y_onehot * logits, axes=(-1,))

        assert logits_label.shape == logsumexp.shape
        return ops.summation(logsumexp - logits_label) / (logits.size / nclass)


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        # Suppose x is a 2D tensor with shape (batch_size, dim)
        assert x.ndim == 2
        weight = ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape)
        bias = ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)

        if self.training:
            nsample = x.shape[0]
            mean = ops.summation(x, axes=(0,)) / nsample
            mean_broad = ops.broadcast_to(mean.reshape((1, self.dim)), x.shape)
            var = ops.summation((x - mean_broad) ** 2, axes=(0,)) / nsample
            var_broad = ops.broadcast_to(var.reshape((1, self.dim)), x.shape)

            x_hat = (x - mean_broad) / (var_broad + self.eps) ** 0.5
            res = x_hat * weight + bias
            
            self.running_mean.data = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var.data = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            rmean = ops.broadcast_to(ops.reshape(self.running_mean, (1, self.dim)), x.shape)
            rvar = ops.broadcast_to(ops.reshape(self.running_var, (1, self.dim)), x.shape)
            x_hat = (x - rmean) / (rvar + self.eps) ** 0.5
            res = x_hat * weight + bias
        
        assert res.shape == x_hat.shape == x.shape
        return res

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        assert x.ndim == 2
        mean = ops.summation(x, axes=(1,)) / self.dim
        mean = ops.broadcast_to(mean.reshape((x.shape[0], 1)), x.shape)
        var = ops.summation((x - mean) ** 2, axes=(1,)) / self.dim
        var = ops.broadcast_to(var.reshape((x.shape[0], 1)), x.shape)
        weight = ops.broadcast_to(self.weight.reshape((1, self.dim)), x.shape)
        bias = ops.broadcast_to(self.bias.reshape((1, self.dim)), x.shape)

        x_hat = (x - mean) / (var + self.eps) ** 0.5
        assert mean.shape == var.shape == weight.shape == bias.shape == x_hat.shape
        return weight * x_hat + bias


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            reset_tensor = init.randb(*x.shape, p=1.0-self.p, dtype=x.dtype) / (1.0 - self.p)
            return x * reset_tensor
        else:
            return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fn(x)
