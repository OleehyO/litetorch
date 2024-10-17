"""Operator implementations."""

from numbers import Number
from typing import Optional, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray: # type: ignore
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        return out_grad, out_grad

def add(a: Tensor, b: Tensor) -> Tensor:
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray: # type: ignore
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        return (out_grad,)

def add_scalar(a: Tensor, scalar: Number) -> Tensor:
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray: # type: ignore
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs

def multiply(a: Tensor, b: Tensor) -> Tensor:
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray: # type: ignore
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        return (out_grad * self.scalar,)

def mul_scalar(a: Tensor, scalar: Number) -> Tensor:
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray: # type: ignore
        return a**b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a: Tensor, b: Tensor) -> Tensor:
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar if scalar >= 0 else scalar * 1.0

    def compute(self, a: NDArray) -> NDArray: # type: ignore
        return a ** self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        assert len(node.inputs) == 1
        local_grad = self.scalar * (node.inputs[0] ** (self.scalar - 1))
        assert out_grad.shape == local_grad.shape == node.inputs[0].shape
        return (out_grad * local_grad,)

def power_scalar(a: Tensor, scalar: Number) -> Tensor:
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray: # type: ignore
        assert a.shape == b.shape
        return a / b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        assert len(node.inputs) == 2
        numer, denom = node.inputs
        numer_local_grad = denom ** -1
        denom_local_grad = -numer * denom**-2
        numer_grad = out_grad * numer_local_grad
        denom_grad = out_grad * denom_local_grad

        assert numer_grad.shape == numer.shape and denom_grad.shape == denom.shape
        return (numer_grad, denom_grad)

def divide(a: Tensor, b: Tensor) -> Tensor:
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray: # type: ignore
        return a / self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        assert len(node.inputs) == 1
        numer: Tensor = node.inputs[0]
        denom = broadcast_to(
            reshape(Tensor(self.scalar, requires_grad=False), tuple([1]*(numer.ndim))),
            numer.shape
        )
        local_grad = denom ** -1
        assert out_grad.shape == local_grad.shape == numer.shape
        return out_grad * local_grad

def divide_scalar(a: Tensor, scalar: Number) -> Tensor:
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a: NDArray) -> NDArray: # type: ignore
        assert a.ndim > 1
        if self.axes is None and a.ndim > 1:
            self.axes = tuple([a.ndim-2, a.ndim-1])
        return array_api.swapaxes(a, *self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        assert len(node.inputs) == 1
        res = transpose(out_grad, axes=self.axes)

        assert res.shape == node.inputs[0].shape
        return (res,)

def transpose(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray: # type: ignore
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        assert len(node.inputs) == 1
        res = reshape(out_grad, node.inputs[0].shape)
        assert res.shape == node.inputs[0].shape
        return (res,)

def reshape(a: Tensor, shape: tuple) -> Tensor:
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a: NDArray) -> NDArray: # type: ignore
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        assert len(node.inputs) == 1
        inp = node.inputs[0]
        new_inp_shape = list(inp.shape)

        if inp.ndim != node.ndim:
            temp = 0
            if inp.ndim != 0 and inp.shape[0] != 1:
                while temp < len(node.shape) and node.shape[temp] != inp.shape[0]:
                    temp += 1
                assert temp < len(node.shape)

            new_inp_shape = [1] * temp + new_inp_shape
            new_inp_shape = new_inp_shape + [1] * (node.ndim - len(new_inp_shape))

        assert len(new_inp_shape) == node.ndim
        new_inp_shape = array_api.array(new_inp_shape)
        indexes = tuple(array_api.where(new_inp_shape == 1)[0])

        res = out_grad
        res = summation(res, axes=indexes)
        res.data = res.data.reshape(inp.shape)

        assert res.shape == node.inputs[0].shape
        return (res, )

def broadcast_to(a: Tensor, shape: tuple) -> Tensor:
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        if axes is not None and not isinstance(axes, tuple):
            self.axes = (axes,)

    def compute(self, a: NDArray) -> NDArray: # type: ignore
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        assert len(node.inputs) == 1
        if self.axes is None:
            self.axes = tuple(range(node.inputs[0].ndim))
        inp = node.inputs[0]
        out_new_shape = list(inp.shape)
        for i in self.axes:
            out_new_shape[i] = 1

        res = reshape(out_grad, tuple(out_new_shape))
        res = broadcast_to(res, inp.shape)

        assert res.shape == node.inputs[0].shape
        return res

def summation(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray) -> NDArray: # type: ignore
        # Support broadcasting
        return a @ b

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        assert len(node.inputs) == 2
        X, Y = node.inputs
        X_T, Y_T = transpose(X), transpose(Y)
        X_grad = out_grad @ Y_T
        Y_grad = X_T @ out_grad
        if X_grad.ndim != X.ndim:
            assert X_grad.ndim > X.ndim
            assert X_grad.shape[-1] == X.shape[-1] and X_grad.shape[-2] == X.shape[-2]
            X_grad = summation(X_grad, tuple(range(X_grad.ndim - X.ndim)))
        if Y_grad.ndim != Y.ndim:
            assert Y_grad.ndim > Y.ndim
            assert Y_grad.shape[-1] == Y.shape[-1] and Y_grad.shape[-2] == Y.shape[-2]
            Y_grad = summation(Y_grad, tuple(range(Y_grad.ndim - Y.ndim)))

        assert X_grad.shape == X.shape and Y_grad.shape == Y.shape
        return (X_grad, Y_grad)

def matmul(a: Tensor, b: Tensor) -> Tensor:
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a: NDArray) -> NDArray: # type: ignore
        return -a

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        assert out_grad.shape == node.inputs[0].shape
        return (-out_grad,)

def negate(a: Tensor) -> Tensor:
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a: NDArray) -> NDArray: # type: ignore
        return array_api.log(a)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        assert len(node.inputs) == 1
        res = out_grad * (node.inputs[0] ** -1)

        assert res.shape == node.inputs[0].shape
        return (res,)

def log(a: Tensor) -> Tensor:
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a: NDArray) -> NDArray: # type: ignore
        return array_api.exp(a)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        assert len(node.inputs) == 1
        assert out_grad.shape == node.shape
        return (out_grad * node, )

def exp(a: Tensor) -> Tensor:
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a: NDArray) -> NDArray: # type: ignore
        return array_api.maximum(a, 0)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        outv = node.realize_cached_data()
        assert array_api.sum(outv<0) == 0
        grad_data = (outv>0).astype(float)
        local_grad = Tensor(grad_data, requires_grad=True)
        assert local_grad.shape == out_grad.shape == node.inputs[0].shape
        return (local_grad * out_grad,)

def relu(a: Tensor) -> Tensor:
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a: NDArray) -> NDArray: # type: ignore
        return array_api.tanh(a)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        assert len(node.inputs) == 1
        assert out_grad.shape == node.shape
        return (out_grad * (1 - (node ** 2)), )

def tanh(a: Tensor) -> Tensor:
    return Tanh()(a)


class GetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def compute(self, a: NDArray) -> NDArray: # type: ignore
        return a[self.index]

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        assert len(node.inputs) == 1
        inp = node.inputs[0]
        ret = array_api.zeros(inp.shape, device=inp.device)
        ret[self.index] = out_grad
        return (Tensor(ret),)

def get_item(a: Tensor, index):
    return GetItem(index)(a)


def find_positions(sequence):
    result = [0] * len(sequence)
    for index, value in enumerate(sequence):
        result[value] = index
    return result

class Permute(TensorOp):
    def __init__(self, order):
        self.order = order
        self.invorder = find_positions(order)

    def compute(self, a: NDArray) -> NDArray: # type: ignore
        return array_api.transpose(a, self.order)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        return (permute(out_grad, self.invorder),)


def permute(a: Tensor, order: tuple) -> Tensor:
    return Permute(order)(a)

class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        assert isinstance(axis, int)
        self.axis = axis

    def compute(self, *args: Tuple[NDArray]) -> NDArray: # type: ignore
        ori_shape = list(args[0].shape)
        for s in args:
            assert list(s.shape) == ori_shape
        new_shape = [len(args)] + ori_shape

        new_array = array_api.empty(new_shape, device=args[0].device)
        for i in range(len(args)):
            new_array[i] = args[i]
        
        order = list(range(1, len(new_shape)))
        order.insert(self.axis, 0)
        return array_api.transpose(new_array, order)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        assert out_grad.shape == node.shape
        res = split(out_grad, self.axis)
        assert res.shape[0] == len(node.inputs)
        ret = []
        for i in range(res.shape[0]):
            ret.append(res[i])
        return tuple(ret)

def stack(args: Union[tuple, list], axis: int) -> Tensor:
    assert isinstance(axis, int)
    if not (isinstance(args, tuple) or isinstance(args, list)):
        raise ValueError("args must be a tuple or list")
    return Stack(axis)(*args)

# class Split(TensorTupleOp):
class Split(TensorOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        assert isinstance(axis, int)
        self.axis = axis

    def compute(self, A: NDArray) -> NDArray: # type: ignore
        # (4,5,6,7) -> (6,4,5,7) when axis = 2
        nsplit = A.shape[self.axis]
        new_order = list(range(len(A.shape)))
        new_order = [new_order[self.axis]] + new_order[:self.axis] + new_order[self.axis+1:]
        self.new_order = new_order

        tmpA = array_api.transpose(A, new_order)
        res = []
        for i in range(nsplit):
            res.append((tmpA[i] + 0).numpy())  # copy a new array
        return array_api.array(res, device=A.device)


    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        # node are Tensor that has been split
        return (permute(out_grad, find_positions(self.new_order)) ,)

def split(a: Tensor, axis: int) -> Tensor:
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, list):
            self.axes = tuple(axes)
        elif not isinstance(axes, tuple):
            self.axes = (axes,)
        else:
            self.axes = axes

    def compute(self, a: NDArray) -> NDArray: # type: ignore
        return array_api.flip(a, self.axes)

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        return (flip(out_grad, self.axes),)

def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        assert axes is not None
        if isinstance(axes, list):
            self.axes = tuple(axes)
        elif not isinstance(axes, tuple):
            self.axes = (axes,)
        else:
            self.axes = axes
        self.dilation = dilation

    def compute(self, a: NDArray) -> NDArray: # type: ignore
        new_shape = list(a.shape)
        for i in self.axes:
            new_shape[i] *= (self.dilation + 1)
        dilate_array = array_api.zeros(new_shape, device=a.device)
        slices = [slice(None) for _ in range(a.ndim)]
        for i in self.axes:
            slices[i] = slice(None, None, self.dilation + 1)
        assert dilate_array[slices].shape == a.shape
        dilate_array[slices] = a
        return dilate_array


    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        return (undilate(out_grad, self.axes, self.dilation),)

def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        if isinstance(axes, list):
            self.axes = tuple(axes)
        elif not isinstance(axes, tuple):
            self.axes = (axes,)
        else:
            self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        new_shape = list(a.shape)
        for i in self.axes:
            new_shape[i] //= (self.dilation + 1)
        undilate_array = array_api.zeros(new_shape, device=a.device)
        slices = [slice(None) for _ in range(a.ndim)]
        for i in self.axes:
            slices[i] = slice(None, None, self.dilation + 1)
        assert undilate_array.shape == a[slices].shape
        undilate_array = a[slices]
        return undilate_array
        
    def gradient(self, out_grad, node):
        return (dilate(out_grad, self.axes, self.dilation),)

def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding
    
    def compute(self, A: NDArray, B: NDArray) -> NDArray: # type: ignore
        '''
        A: activation (N, H, W, c_in)
        B: kernel weight (k_h, k_w, c_in, c_out)
        '''
        assert BACKEND == "nd", f"Conv op only support ltt.NDArray, but got BACKEND: {BACKEND}"
        assert A.ndim == 4 and B.ndim == 4
        N = A.shape[0]
        in_h, in_w = A.shape[1] + 2*self.padding, A.shape[2] + 2*self.padding
        c_in, c_out = A.shape[-1], B.shape[-1]
        k_h, k_w = B.shape[0], B.shape[1]

        uniformed_A = array_api.pad(A, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        uniformed_A = uniformed_A[:, :in_h-(in_h-k_h)%self.stride, :in_w-(in_w-k_w)%self.stride, :].compact()
        uniformed_B = B
        in_h, in_w = uniformed_A.shape[1], uniformed_A.shape[2]
        out_h, out_w = (in_h-k_h)//self.stride + 1, (in_w-k_w)//self.stride + 1

        self.out_h, self.out_w = out_h, out_w
        self.N, self.c_in, self.c_out = N, c_in, c_out

        assert((in_h - k_h) % self.stride == 0 and 
               (in_w - k_w) % self.stride == 0), \
               f"Error in conv compute: {A.shape}, B.shape: {B.shape}, stride: {self.stride}, padding: {self.padding}"

        out_shape      = (N, out_h, out_w, c_out)
        im2col_shape   = (N, out_h, out_w, c_in, k_h, k_w)
        im2col_strides = (in_w * in_h * c_in, self.stride * in_w * c_in, self.stride * c_in, 1, in_w * c_in, c_in)

        W = array_api.transpose(uniformed_B, (2, 0, 1, 3)).compact()
        assert W.shape[0]*W.shape[1]*W.shape[2]*W.shape[3] == (c_in * k_h * k_w * c_out), \
            f"Some error in shape of W.shape: {W.shape}, uniformed_B.shape: {uniformed_B.shape}, c_in: {c_in}, k_h: {k_h}, k_w: {k_w}, c_out: {c_out}"
        W = array_api.reshape(W, (c_in * k_h * k_w, c_out)).compact()

        im2col = NDArray.make(
            shape=im2col_shape,
            strides=im2col_strides,
            device=uniformed_A.device,
            handle=uniformed_A._handle,
            offset=0
        ).compact().reshape(
            (N * out_h * out_w, c_in * k_h * k_w)
        ).compact()
        self.im2col = im2col

        out = (im2col @ W).reshape(out_shape).compact()
        return out

    def gradient(self, out_grad: Tensor, node: Tensor) -> Tuple[Tensor]:
        ### out_grad: (N, out_h, out_w, c_out)
        assert len(node.inputs) == 2
        A, B = node.inputs  # A: (N, H, W, c_in), B: (k_h, k_w, c_in, c_out)
        assert A.ndim == 4 and B.ndim == 4, "A.ndim != 4 or B.ndim != 4"
        assert B.shape[0] == B.shape[1], "B.shape[0] != B.shape[1]"

        im2col_T = Tensor(
            self.im2col.permute((1, 0)).compact(),
            device=out_grad.device,
            dtype=out_grad.dtype
        )  # (c_in * k_h * k_w, N * out_h * out_w)

        out_grad_reshaped = out_grad.reshape(
            (self.N * self.out_h * self.out_w, self.c_out)
        )
        B_grad = (im2col_T @ out_grad_reshaped)\
                 .reshape((B.shape[2], B.shape[0], B.shape[1], B.shape[3]))\
                 .permute((1, 2, 0, 3))
        assert B_grad.shape == B.shape, "B_grad.shape != B.shape"

        out_grad_dialated = dilate(out_grad, (1, 2), self.stride-1)

        B_filped = flip(B, axes=(0, 1)).transpose((2, 3))

        A_grad = conv(out_grad_dialated, B_filped, stride=1, padding=B.shape[0]-1)
        A_grad = A_grad[:, self.padding:self.padding+A.shape[1], self.padding:self.padding+A.shape[2], :]
        assert A_grad.shape == A.shape, "A_grad.shape != A.shape"
        return (A_grad, B_grad)

def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
