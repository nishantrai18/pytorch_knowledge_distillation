import random
import torch

import torch.nn as nn


class WeighedLoss(nn.Module):
    """
    Class that implements automatically weighed loss from: https://arxiv.org/pdf/1705.07115.pdf
    """

    def __init__(self, num_losses=0):

        super(WeighedLoss, self).__init__()
        self.device = torch.device("cpu")
        self.coeffs = []
        for i in range(num_losses):
            init_value = random.random()
            param = nn.Parameter(torch.tensor(init_value))
            name = "auto_param_" + str(i)
            self.register_parameter(name, param)
            self.coeffs.append(param)

    def forward(self, losses=[]):
        """
        Forward pass

        Keyword Arguments:
            losses {list} -- List of tensors of losses

        Returns:
            torch.Tensor -- 0-dimensional tensor with final loss. Can backpropagate it.
        """

        assert len(losses) == len(self.coeffs), \
            "Loss mismatch, check how many losses are passed"

        net_loss = torch.zeros(1).to(self.device)

        for i, loss in enumerate(losses):
            net_loss += torch.exp(-self.coeffs[i]) * loss
            net_loss += 0.5 * self.coeffs[i]

        return net_loss
