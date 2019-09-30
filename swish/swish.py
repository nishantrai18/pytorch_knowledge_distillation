import torch
import unittest

import swish_cpp

from torch import nn
from torch.autograd import gradcheck


class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, beta, input):
        output = swish_cpp.forward(beta, input)
        ctx.save_for_backward(beta, input)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        beta, input = ctx.saved_tensors
        db_grad, dx_grad = swish_cpp.backward(beta, input)
        return db_grad * grad_out, dx_grad * grad_out


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input):
        return SwishFunction.apply(self.beta, input)


class TestSwishGrad(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.swish = SwishFunction.apply

    @classmethod
    def tearDownClass(self):
        pass

    def test_swish_grad(self):
        """
        Uses gradcheck to verify swish gradient
        """

        for i in range(100):
            inputs = (torch.randn(1, dtype=torch.double, requires_grad=True),
                      torch.randn(20, dtype=torch.double, requires_grad=True))
            success = gradcheck(self.swish, inputs, eps=1e-6, atol=1e-4)
            assert success, "Failure for: {}".format(inputs)


if __name__ == "__main__":
    unittest.main()
