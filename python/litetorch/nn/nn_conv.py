"""The module.
"""
from litetorch.autograd import Tensor
from litetorch import ops
import litetorch.init as init
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.padding = self.kernel_size // 2
        # kernel
        fan_in = self.in_channels * self.kernel_size ** 2
        fan_out = self.out_channels * self.kernel_size ** 2
        shape = (self.kernel_size, self.kernel_size, self.in_channels, self.out_channels)
        weight = init.kaiming_uniform(fan_in, fan_out, shape=shape)
        self.weight = Parameter(weight, device=device, dtype=dtype)
        # bias
        self.use_bias = bias
        if self.use_bias:
            k = 1.0 / (self.in_channels * self.kernel_size ** 2) ** 0.5
            b = ops.reshape(init.uniform(self.out_channels, 1, k), (self.out_channels,))
            self.bias = Parameter(b, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        # NCHW -> NHCW -> NHWC
        x = ops.transpose(x, (2, 1))
        x = ops.transpose(x, (3, 2))
        # NHWC
        output = ops.conv(x, self.weight, self.stride, self.padding)
        # 1C -> 111C -> NHWC
        if self.use_bias:
            bias = ops.reshape(self.bias, (1, 1, 1, self.out_channels)) 
            bias = ops.broadcast_to(bias, output.shape)
            output += bias
        # NHWC-> NHCW -> NCHW
        output = ops.transpose(output, (3, 2))
        output = ops.transpose(output, (2, 1))
        
        return output
