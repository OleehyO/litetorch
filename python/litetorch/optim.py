"""Optimization module"""
import litetorch as ltt
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for param in self.params:
            if param.grad is None:
                continue
            k = hash(param)
            newu = self.momentum * self.u[k] + (1 - self.momentum) * (param.grad.data + self.weight_decay * param.data)
            param.data -= self.lr * newu
            self.u[k] = newu.data

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        raise NotImplementedError()


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for param in self.params:
            if param.grad is None:
                continue
            k = hash(param)
            grad_with_decay = param.grad.data + self.weight_decay * param.data
            self.m[k] = self.beta1 * self.m[k].data + (1 - self.beta1) * grad_with_decay.data
            self.v[k] = self.beta2 * self.v[k].data + (1 - self.beta2) * grad_with_decay.data ** 2
            m_hat = self.m[k].data / (1 - self.beta1 ** self.t)
            v_hat = self.v[k].data / (1 - self.beta2 ** self.t)
            param.data -= self.lr * m_hat.data / (v_hat.data ** 0.5 + self.eps)

