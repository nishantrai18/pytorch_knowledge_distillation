"""
Implementation of Knowledge Distillation module
Wraps base student and pre-trained teacher networks
"""

import os
import torch

import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from swish.swish import Swish


def get_activation_factory(activation):
    def relu_factory():
        return torch.nn.ReLU()

    def swish_factory():
        return Swish()

    if activation == "relu":
        return relu_factory
    elif activation == "swish":
        return swish_factory
    else:
        return None


class ModelWrapper(nn.Module, ABC):
    """
    Base class for Model Wrapper. This base class is used
    to interact with the other urility functions present
    in other files.
    All new models should be based on top it in order to
    utilize the utility functions
    """

    def __init__(self, name):
        super(ModelWrapper, self).__init__()
        self.base_log_dir = "../logs/"
        self.name = name
        self.log_dir = os.path.join(self.base_log_dir, self.name)

    @abstractmethod
    def train_loss(self, result, labels):
        pass

    @abstractmethod
    def test_loss(self, result, labels):
        pass


class IndividualModel(ModelWrapper):

    def __init__(self, base_model, name):
        super(IndividualModel, self).__init__("indmod_" + name)

        self.model = base_model
        self.train_loss_criterion = nn.CrossEntropyLoss(reduction='mean')
        self.test_loss_criterion = nn.CrossEntropyLoss(reduction='sum')

    def forward(self, x):
        model_outs = self.model(x)
        model_preds = F.log_softmax(model_outs, dim=1)

        return {
            "outs": model_outs,
            "preds": model_preds
        }

    def train_loss(self, result, labels):
        return {
            "loss": self.train_loss_criterion(result["outs"], labels)
        }

    def test_loss(self, result, labels):
        return {
            "loss": self.test_loss_criterion(result["outs"], labels)
        }


class KnowledgeDistillModelWrapper(ModelWrapper, ABC):

    def __init__(self, args, name):
        super(KnowledgeDistillModelWrapper, self).__init__("kd_" + name)

        self.args = args

        # Hyper-parameters for Knowledge distillation
        self.temperature = args.temperature
        self.kd_weight = args.kd_weight

    def loss_criterion(self, result, labels):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.

        :param result: Expect the dictionary returned by forward()
        :param labels: GT label for the batch
        """

        w = self.kd_weight
        t = self.temperature

        kd_loss = nn.KLDivLoss()(
            F.log_softmax(result["outs"] / t, dim=1),
            F.softmax(result["teacher_outs"] / t, dim=1)
        )
        gt_loss = F.cross_entropy(result["outs"], labels)

        # Weigh kd_loss with t^2 to preserve scale of gradients
        final_loss = (kd_loss * w * t * t) + (gt_loss * (1 - w))

        return {
            "loss": final_loss,
            "gt_loss": gt_loss,
            "kd_loss": kd_loss
        }

    def train_loss(self, result, labels):
        return self.loss_criterion(result, labels)

    def test_loss(self, result, labels):
        # Need to multiply to preserve correct values after the mean
        return {
            k: v * self.args.batch_size for k, v in self.loss_criterion(result, labels).items()
        }


class OnTheFlyKDModel(KnowledgeDistillModelWrapper):
    """
    Implementation of on the fly knowledge distill model
    """

    def __init__(self, student, teacher, args, name):
        """
        Init KD object. Note that the models passed should be instances
        of ModelWrapper

        :param student: Base student model to be trained
        :param teacher: Pre-trained teacher model (Frozen)
        """

        super(OnTheFlyKDModel, self).__init__(args, "otf_" + name)

        # Init base models
        self.student = student
        self.teacher = teacher

        # Freeze the teacher module in order to avoid training it
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x):
        student_res = self.student(x)
        teacher_res = self.teacher(x)

        # Naming student results without prefix for compatibility
        return {
            "outs": student_res["outs"],
            "preds": student_res["preds"],
            "teacher_outs": teacher_res["outs"]
        }


class CachedKDModel(KnowledgeDistillModelWrapper):
    """
    Implementation of efficient cached knowledge distill model
    Note that the dataloaders returned by,
        fetch_cifar100_efficient_kd_dataloaders()
    should be used in conjunction with this
    """

    def __init__(self, student, teachers, args, name):
        """
        Init KD object. Note that the models passed should be instances
        of ModelWrapper

        :param student: Base student model to be trained
        :param teachers: Names of teachers to use
        """

        super(CachedKDModel, self).__init__(args, "cached_" + name)

        # Init base models
        self.student = student
        self.teachers = teachers

    def forward(self, x):
        student_res = self.student(x["data"])
        # Compute the average output for all the teachers
        valid_teacher_res = \
            [v for k, v in x.items() if 'model_out_' in k and k.replace('model_out_', '') in self.teachers]
        teacher_res = torch.mean(torch.stack(valid_teacher_res), dim=0)

        return {
            "outs": student_res["outs"],
            "preds": student_res["preds"],
            "teacher_outs": teacher_res
        }
