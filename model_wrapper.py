"""
Implementation of Knowledge Distillation module
Wraps base student and pre-trained teacher networks
"""

import torch

import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod


class ModelWrapper(nn.Module, ABC):
    """
    Base class for Model Wrapper. This base class is used
    to interact with the other urility functions present
    in other files.
    All new models should be based on top it in order to
    utilize the utility functions
    """

    def __init__(self):
        super(ModelWrapper, self).__init__()

    @abstractmethod
    def train_loss(self, result, labels):
        pass

    @abstractmethod
    def test_loss(self, result, labels):
        pass


class IndividualModel(ModelWrapper):

    def __init__(self, base_model):
        super(IndividualModel, self).__init__()

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
        return self.train_loss_criterion(result["outs"], labels)

    def test_loss(self, result, labels):
        return self.test_loss_criterion(result["outs"], labels)


class KnowledgeDistillModelWrapper(ModelWrapper, ABC):

    def __init__(self, args):
        super(KnowledgeDistillModelWrapper, self).__init__()

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

        return final_loss

    def train_loss(self, result, labels):
        return self.loss_criterion(result, labels)

    def test_loss(self, result, labels):
        # Need to multiply to preserve correct values after the mean
        return self.loss_criterion(result, labels) * self.args.batch_size


class OnTheFlyKDModel(KnowledgeDistillModelWrapper):
    """
    Implementation of on the fly knowledge distill model
    """

    def __init__(self, student, teacher, args):
        """
        Init KD object. Note that the models passed should be instances
        of ModelWrapper

        :param student: Base student model to be trained
        :param teacher: Pre-trained teacher model (Frozen)
        """

        super(OnTheFlyKDModel, self).__init__(args)

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
            "teacher_outs": teacher_res["outs"]
        }


class CachedKDModel(KnowledgeDistillModelWrapper):
    """
    Implementation of efficient cached knowledge distill model
    Note that the dataloaders returned by,
        fetch_cifar100_efficient_kd_dataloaders()
    should be used in conjunction with this
    """

    def __init__(self, student, teachers, args):
        """
        Init KD object. Note that the models passed should be instances
        of ModelWrapper

        :param student: Base student model to be trained
        :param teachers: Names of teachers to use
        """

        super(CachedKDModel, self).__init__(args)

        # Init base models
        self.student = student
        self.teachers = teachers

    def forward(self, x):
        student_res = self.student(x["data"])
        # Compute the average output for all the teachers
        teacher_res = torch.mean(torch.stack([v for k, v in x.items() if 'model_out_' in k]))

        return {
            "outs": student_res["outs"],
            "teacher_outs": teacher_res
        }
