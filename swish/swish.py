import torch
import unittest
import swish_cpp

import torch.nn.functional as F

from torch import nn
from torch.autograd import gradcheck
from tqdm import tqdm


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


class TestBasicSwishTraining(unittest.TestCase):
    """
    Tests whether Swish() is able to learn the beta appropriately
    """

    @classmethod
    def setUpClass(self):
        self.device = torch.device("cpu")

    @classmethod
    def tearDownClass(self):
        """
        This code code is ran once after all tests.
        """
        pass

    def target_func_1(self, x):
        return x * F.sigmoid(x)

    def target_func_2(self, x):
        return x

    def target_func_3(self, x):
        return x * F.sigmoid(2.5 * x)

    def run_training_for_instance(self, target_func):

        num_epochs = 1000

        inputs = torch.randn(20, requires_grad=False)

        swish = Swish()
        loss_fn = nn.SmoothL1Loss()
        target = target_func(inputs) + (torch.randn_like(inputs) * 0.01)

        optim = torch.optim.Adam(swish.parameters(), lr=1e-1)

        tq = tqdm(range(num_epochs), desc="Training progress")

        for t in tq:

            y = swish(inputs)
            loss = loss_fn(y, target)

            optim.zero_grad()
            loss.backward()
            optim.step()

            tq.set_postfix({
                "l1": loss.item(),
                "grad:": swish.beta.grad.abs().mean()
            })

        print("Learned values:", list(swish.parameters()))

    def test_basic_training(self):

        self.run_training_for_instance(self.target_func_1)
        self.run_training_for_instance(self.target_func_2)
        self.run_training_for_instance(self.target_func_3)


if __name__ == "__main__":
    unittest.main()
