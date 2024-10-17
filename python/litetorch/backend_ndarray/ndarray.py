import operator
import math
from functools import reduce
from typing import Optional
import numpy as np
from . import ndarray_backend_numpy
from . import ndarray_backend_cpu


def prod(x):
    return reduce(operator.mul, x, 1)


class BackendDevice:
    """A backend device, wrapps the implementation module."""

    def __init__(self, name, mod):
        self.name = name
        self.mod = mod

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name + "()"

    def __getattr__(self, name):
        return getattr(self.mod, name)

    def enabled(self):
        return self.mod is not None

    def randn(self, *shape, dtype="float32"):
        return NDArray(np.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape, dtype="float32"):
        return NDArray(np.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n, i, dtype="float32"):
        return NDArray(np.eye(n, dtype=dtype)[i], device=self)

    def empty(self, shape, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return NDArray.make(shape, device=self)

    def full(self, shape, fill_value, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr


def cuda():
    """Return cuda device"""
    try:
        from . import ndarray_backend_cuda

        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return BackendDevice("cuda", None)


def cpu_numpy():
    """Return numpy device"""
    return BackendDevice("cpu_numpy", ndarray_backend_numpy)


def cpu():
    """Return cpu device"""
    return BackendDevice("cpu", ndarray_backend_cpu)


def default_device():
    return cpu_numpy()


def all_devices():
    """return a list of all available devices"""
    return [cpu(), cuda(), cpu_numpy()]


class NDArray:
    def __init__(self, other, device=None):
        """Create by copying another NDArray, or from numpy"""
        if isinstance(other, NDArray):
            # create a copy of existing NDArray
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)  # this creates a copy
        elif isinstance(other, np.ndarray):
            # create copy from numpy array
            device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            array = NDArray(np.array(other), device=device)
            self._init(array)

    def _init(self, other):
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle

    @staticmethod
    def compact_strides(shape):
        """Utility function to compute compact strides"""
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])

    @staticmethod
    def make(shape, strides=None, device=None, handle=None, offset=0):
        """Create a new NDArray with the given properties.  This will allocation the
        memory if handle=None, otherwise it will use the handle of an existing
        array."""
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = NDArray.compact_strides(shape) if strides is None else strides
        array._offset = offset
        array._device = device if device is not None else default_device()
        if handle is None:
            array._handle = array.device.Array(prod(shape))
        else:
            array._handle = handle
        return array

    ### Properies and string representations
    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        return self._strides

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        # only support float32 for now
        return "float32"

    @property
    def ndim(self):
        """Return number of dimensions."""
        return len(self._shape)

    @property
    def size(self):
        return prod(self._shape)

    def __repr__(self):
        return "NDArray(" + self.numpy().__str__() + f", device={self.device})"

    def __str__(self):
        return self.numpy().__str__()

    ### Basic array manipulation
    def fill(self, value):
        """Fill (in place) with a constant value."""
        self._device.fill(self._handle, value)

    def to(self, device):
        """Convert between devices, using to/from numpy calls as the unifying bridge."""
        if device == self.device:
            return self
        else:
            return NDArray(self.numpy(), device=device)

    def numpy(self):
        """convert to a numpy array"""
        return self.device.to_numpy(
            self._handle, self.shape, self.strides, self._offset
        )

    def is_compact(self):
        """Return true if array is compact in memory and internal size equals product
        of the shape dimensions"""
        return (
            self._strides == self.compact_strides(self._shape)
            and prod(self.shape) == self._handle.size
        )

    def compact(self):
        """Convert a matrix to be compact"""
        if self.is_compact():
            return self
        else:
            out = NDArray.make(self.shape, device=self.device)
            self.device.compact(
                self._handle, out._handle, self.shape, self.strides, self._offset
            )
            return out

    def as_strided(self, shape, strides):
        """Restride the matrix without copying memory."""
        assert len(shape) == len(strides)
        return NDArray.make(
            shape, strides=strides, device=self.device, handle=self._handle
        )

    @property
    def flat(self):
        return self.reshape((self.size,))

    def reshape(self, new_shape):
        """
        Reshape the matrix without copying memory.  This will return a matrix
        that corresponds to a reshaped array but points to the same memory as
        the original array.

        Raises:
            ValueError if product of current shape is not equal to the product
            of the new shape, or if the matrix is not compact.

        Args:
            new_shape (tuple): new shape of the array

        Returns:
            NDArray : reshaped array; this will point to thep
        """

        if prod(self.shape) != prod(new_shape):
            raise ValueError(
                "Product of current shape is not equal to the product of the new shape"
            )
        if not self.is_compact():
            raise ValueError("Matrix is not compact")
        return NDArray.make(
            new_shape,
            self.compact_strides(new_shape),
            self.device,
            self._handle,
            self._offset,
        )

    def permute(self, new_axes):
        new_strides = tuple(self._strides[i] for i in new_axes)
        new_shape = tuple(self._shape[i] for i in new_axes)
        return NDArray.make(
            new_shape, new_strides, self.device, self._handle, self._offset
        )

    def broadcast_to(self, new_shape):
        """
        Broadcast an array to a new shape.  new_shape's elements must be the
        same as the original shape, except for dimensions in the self where
        the size = 1 (which can then be broadcast to any size).  As with the
        previous calls, this will not copy memory, and just achieves
        broadcasting by manipulating the strides.

        Raises:
            assertion error if new_shape[i] != shape[i] for all i where
            shape[i] != 1

        Args:
            new_shape (tuple): shape to broadcast to

        Returns:
            NDArray: the new NDArray object with the new broadcast shape; should
            point to the same memory as the original array.
        """

        assert len(new_shape) >= self.ndim
        expand_shape = (1,) * (len(new_shape) - self.ndim) + self.shape
        expand_strides = (0,) * (len(new_shape) - self.ndim) + self.strides
        for l, r in zip(expand_shape, new_shape):
            assert l == r or l == 1, "Cannot broadcast from %s to %s" % (
                expand_shape,
                new_shape,
            )
        new_strides = [
            expand_strides[i] if expand_shape[i] != 1 else 0
            for i in range(len(expand_strides))
        ]
        return NDArray.make(
            new_shape, tuple(new_strides), self.device, self._handle, self._offset
        )

    ### Get and set elements

    def process_slice(self, sl, dim):
        """Convert a slice to an explicit start/stop/step"""
        start, stop, step = sl.start, sl.stop, sl.step
        if start == None:
            start = 0
        if start < 0:
            start = self.shape[dim]
        if stop == None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step == None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)

    def __getitem__(self, idxs):

        # handle singleton as tuple, everything as slices
        raw_idxs = idxs
        if isinstance(idxs, list):
            idxs = tuple(idxs)
        elif not isinstance(idxs, tuple):
            idxs = (idxs,)
        if len(idxs) < self.ndim:
            idxs = idxs + (slice(None),) * (self.ndim - len(idxs))
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(idxs) == self.ndim, "Need indexes equal to number of dimensions"

        new_shape = list(self.shape)
        new_offset = self._offset
        new_strides = list(self._strides)
        for i in range(len(idxs)):
            new_shape[i] = 1 + (idxs[i].stop - idxs[i].start - 1) // idxs[i].step
            new_offset += idxs[i].start * self._strides[i]
            new_strides[i] *= idxs[i].step
        if isinstance(raw_idxs, int):
            new_shape = new_shape[1:]
            new_strides = new_strides[1:]
        return NDArray.make(
            tuple(new_shape), tuple(new_strides), self.device, self._handle, new_offset
        )

    def __setitem__(self, idxs, other):
        """Set the values of a view into an array, using the same semantics
        as __getitem__()."""
        view = self.__getitem__(idxs)
        if isinstance(other, NDArray):
            assert prod(view.shape) == prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )

    ### Collection of elementwise and scalar function: add, multiply, boolean, etc

    def ewise_or_scalar(self, other, ewise_func, scalar_func):
        """Run either an elementwise or scalar version of a function,
        depending on whether "other" is an NDArray or scalar
        """
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            if self.shape != other.shape:
                other = other.broadcast_to(self.shape)
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __neg__(self):
        return self * (-1)

    def __pow__(self, other):
        out = NDArray.make(self.shape, device=self.device)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    ### Binary operators all return (0.0, 1.0) floating point values, could of course be optimized
    def __eq__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)

    ### Elementwise functions

    def log(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    ### Matrix multiplication
    def __matmul__(self, other):

        assert self.ndim == 2 and other.ndim == 2

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        # if the matrix is aligned, use tiled matrix multiplication
        if hasattr(self.device, "matmul_tiled") and all(
            d % self.device.__tile_size__ == 0 for d in (m, n, p)
        ):

            if self.device.name == "cuda":
                def tile(a, tile):
                    return a.as_strided(
                        (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                        (a.shape[1] * tile, tile, a.shape[1], 1),
                    )

                t = self.device.__tile_size__
                a = tile(self.compact(), t).compact()
                b = tile(other.compact(), t).permute((0, 2, 3, 1)).compact()
                out = NDArray.make((a.shape[0], t, t, b.shape[-1]), device=self.device)
                self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

                return (
                    out.permute((0, 1, 3, 2)).compact()
                    .reshape((self.shape[0], other.shape[1]))
                )
            else:
                def tile(a, tile):
                    return a.as_strided(
                        (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                        (a.shape[1] * tile, tile, self.shape[1], 1),
                    )

                t = self.device.__tile_size__
                a = tile(self.compact(), t).compact()
                b = tile(other.compact(), t).compact()
                out = NDArray.make((a.shape[0], b.shape[1], t, t), device=self.device)
                self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

                return (
                    out.permute((0, 2, 1, 3))
                    .compact()
                    .reshape((self.shape[0], other.shape[1]))
                )
        else:
            out = NDArray.make((m, p), device=self.device)
            self.device.matmul(
                self.compact()._handle, other.compact()._handle, out._handle, m, n, p
            )
            return out

    ### Reductions, i.e., sum/max over all element or over given axis
    def reduce_view_out(self, axis, keepdims=False):
        """ Return a view to the array set up for reduction functions and output array. """
        if isinstance(axis, tuple) and not axis:
            raise ValueError("Empty axis in reduce")

        if axis is None:
            view = self.compact().reshape((1,) * (self.ndim - 1) + (prod(self.shape),))
            #out = NDArray.make((1,) * self.ndim, device=self.device)
            out = NDArray.make((1,), device=self.device)

        else:
            if isinstance(axis, (tuple, list)):
                assert len(axis) == 1, "Only support reduction over a single axis"
                axis = axis[0]

            view = self.permute(
                tuple([a for a in range(self.ndim) if a != axis]) + (axis,)
            )
            out = NDArray.make(
                tuple([1 if i == axis else s for i, s in enumerate(self.shape)])
                if keepdims else
                tuple([s for i, s in enumerate(self.shape) if i != axis]),
                device=self.device,
            )
        return view, out
    
    def squeeze(self, axis:Optional[tuple[int]]=None):
        if axis is None:
            axis = tuple(i for i in range(self.ndim) if self.shape[i] == 1)
        elif not isinstance(axis, tuple):
            axis = (axis,)
        for a in axis:
            assert self.shape[a] == 1, "Cannot squeeze dimension with size not 1"

        # remove the squeezed dimensions
        new_shape = tuple(s for i, s in enumerate(self.shape) if i not in axis)
        new_strides = tuple(st for i, st in enumerate(self.strides) if i not in axis)
        return NDArray.make(new_shape, new_strides, self.device, self._handle, self._offset)

    def sum(self, axis=None, keepdims=False):
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def max(self, axis=None, keepdims=False):
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def flip(self, axes=None):
        """
        Flip this ndarray along the specified axes.
        Note: compact() before returning.
        """
        assert self._offset == 0, "Offset must be 0"
        if axes is None:
            axes = tuple(range(self.ndim))
        elif not isinstance(axes, tuple):
            axes = (axes,)

        new_strides = list(self.strides)
        new_offset = self._offset
        for axis in axes:
            new_strides[axis] = -self.strides[axis]
            new_offset += (self.shape[axis] - 1) * self.strides[axis]
        res = NDArray.make(self.shape, new_strides, self.device, self._handle, new_offset)
        return res.compact()

    def pad(self, axes):
        """
        Pad this ndarray by zeros by the specified amount in `axes`,
        which lists for _all_ axes the left and right padding amount, e.g.,
        axes = ( (0, 0), (1, 1), (0, 0)) pads the middle axis with a 0 on the left and right side.
        """
        assert len(axes) == self.ndim
        expanded_shape = tuple(self.shape[i] + axes[i][0] + axes[i][1] for i in range(self.ndim))
        padded_array = zeros(expanded_shape, device=self.device)
        for i in range(self.ndim):
            padded_array[tuple(slice(axes[i][0], axes[i][0] + self.shape[i]) for i in range(self.ndim))] = self
        return padded_array


def array(a, dtype="float32", device=None):
    """Convenience methods to match numpy a bit more closely."""
    dtype = "float32" if dtype is None else dtype
    assert dtype == "float32"
    return NDArray(a, device=device)

def empty(shape, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.empty(shape, dtype)

def full(shape, fill_value, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.full(shape, fill_value, dtype)

def zeros(shape, dtype="float32", device=None):
    return full(shape, 0, dtype, device)

def ones(shape, dtype="float32", device=None):
    return full(shape, 1, dtype, device)

def broadcast_to(array, new_shape):
    return array.broadcast_to(new_shape)

def reshape(array, new_shape):
    return array.reshape(new_shape)

def transpose(array, new_order):
    return array.permute(new_order)

def swapaxes(array, axis0, axis1):
    order = list(range(len(array.shape)))
    order[axis0], order[axis1] = order[axis1], order[axis0]
    return transpose(array, order)

def maximum(a, b):
    return a.maximum(b)

def max(a, axis=None, keepdims=False):
    return a.max(axis=axis, keepdims=keepdims)

def log(a):
    return a.log()

def exp(a):
    return a.exp()

def tanh(a):
    return a.tanh()

def sum(a, axis=None, keepdims=False):
    return a.sum(axis=axis, keepdims=keepdims)

def flip(a, axes):
    return a.flip(axes)

def squeeze(a, axis=None):
    return a.squeeze(axis)

def pad(a, axes):
    return a.pad(axes)
