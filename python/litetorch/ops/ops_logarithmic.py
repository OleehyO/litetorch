from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 


class LogSoftmax(TensorOp):
    def compute(self, Z):
        raise NotImplementedError()

    def gradient(self, out_grad, node):
        raise NotImplementedError()

def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z: NDArray): # type: ignore
        maxz = array_api.max(Z, axis=self.axes, keepdims=True)
        self.expz = array_api.exp(Z - maxz)
        self.sumexpz = array_api.sum(self.expz, axis=self.axes, keepdims=True)
        self.origin_outshape = self.sumexpz.shape
        logsumexpz = array_api.log(self.sumexpz)
        return array_api.squeeze(logsumexpz + maxz, axis=self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor):
        assert len(node.inputs) == 1
        assert out_grad.shape == node.shape

        inp = node.inputs[0]
        local_grad = self.expz / self.sumexpz
        assert inp.shape == local_grad.shape
        out_grad_broadcast = broadcast_to(reshape(out_grad, self.origin_outshape), inp.shape)
        return Tensor(local_grad, dtype=node.dtype) * out_grad_broadcast

def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

