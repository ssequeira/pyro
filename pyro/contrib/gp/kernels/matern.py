from __future__ import absolute_import, division, print_function

import torch
from torch.nn import Parameter

from .kernel import Kernel


class Matern32(Kernel):
    """
    Implementation of Matern 3/2 kernel.

    By default, parameters will be `torch.nn.Parameter`s containing `torch.FloatTensor`s.
    To cast them to the correct data type or GPU device, we can call methods such as
    `.double()`, `.cuda(device=None)`,... See
    `torch.nn.Module <http://pytorch.org/docs/master/nn.html#torch.nn.Module>`_ for more information.

    :param int input_dim: Dimension of inputs for this kernel.
    :param torch.Tensor variance: Variance parameter of this kernel.
    :param torch.Tensor lengthscale: Length scale parameter of this kernel.
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None, name="Matern32"):
        super(Matern32, self).__init__(input_dim, active_dims, name)
        if variance is None:
            variance = torch.ones(1)
        self.variance = Parameter(variance)
        if lengthscale is None:
            lengthscale = torch.ones(input_dim)
        self.lengthscale = Parameter(lengthscale)

    def forward(self, X, Z=None):
        if Z is None:
            Z = X
        X = self._slice_X(X)
        Z = self._slice_X(Z)
        if X.size(1) != Z.size(1):
            raise ValueError("Inputs must have the same number of features.")

        _SQRT_3 = 1.7320508075688772
        
        scaled_X = X / self.lengthscale
        scaled_Z = Z / self.lengthscale
        X2 = (scaled_X ** 2).sum(1, keepdim=True)
        Z2 = (scaled_Z ** 2).sum(1, keepdim=True)
        XZ = scaled_X.matmul(scaled_Z.t())
        d2 = X2 - 2 * XZ + Z2.t()
        d = torch.sqrt(d2)

        return self.variance * (1 + _SQRT_3*d) * torch.exp(-_SQRT_3*d)

class Matern52(Kernel):
    """
    Implementation of Matern 5/2 kernel.

    By default, parameters will be `torch.nn.Parameter`s containing `torch.FloatTensor`s.
    To cast them to the correct data type or GPU device, we can call methods such as
    `.double()`, `.cuda(device=None)`,... See
    `torch.nn.Module <http://pytorch.org/docs/master/nn.html#torch.nn.Module>`_ for more information.

    :param int input_dim: Dimension of inputs for this kernel.
    :param torch.Tensor variance: Variance parameter of this kernel.
    :param torch.Tensor lengthscale: Length scale parameter of this kernel.
    """

    def __init__(self, input_dim, variance=None, lengthscale=None, active_dims=None, name="Matern52"):
        super(Matern52, self).__init__(input_dim, active_dims, name)
        if variance is None:
            variance = torch.ones(1)
        self.variance = Parameter(variance)
        if lengthscale is None:
            lengthscale = torch.ones(input_dim)
        self.lengthscale = Parameter(lengthscale)

    def forward(self, X, Z=None):
        if Z is None:
            Z = X
        X = self._slice_X(X)
        Z = self._slice_X(Z)
        if X.size(1) != Z.size(1):
            raise ValueError("Inputs must have the same number of features.")

        _SQRT_5 = 2.2360679774997898
        
        scaled_X = X / self.lengthscale
        scaled_Z = Z / self.lengthscale
        X2 = (scaled_X ** 2).sum(1, keepdim=True)
        Z2 = (scaled_Z ** 2).sum(1, keepdim=True)
        XZ = scaled_X.matmul(scaled_Z.t())
        d2 = X2 - 2 * XZ + Z2.t()
        d = torch.sqrt(d2)

        return self.variance * (1 + _SQRT_5*d + 5*d/3) * torch.exp(-_SQRT_5*d)
    
