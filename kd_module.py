"""
Implementation of Knowledge Distillation module
Wraps base student and pre-trained teacher networks
"""

import torch

import torch.nn as nn
import torch.nn.functional as F

# TODO: Add base class to dedupe KD code


class IndividualModel(nn.Module):

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


class OnTheFlyKDModel(nn.Module):

    def __init__(self, student, teacher, args):
        """
        Init KD object

        :param student: Base student model to be trained
        :param teacher: Pre-trained teacher model
        """

        super(OnTheFlyKDModel, self).__init__()

        # Init base models
        self.student = student
        self.teacher = teacher

        # Hyper-parameters for Knowledge distillation
        self.temperature = args.temperature
        self.kd_weight = args.kd_weight

        # Freeze the teacher module in order to avoid training it
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, x):
        student_outs = self.student(x)
        teacher_outs = self.teacher(x)

        return {
            "student_outs": student_outs,
            "teacher_outs": teacher_outs,
        }

    def loss_criterion(self, result, labels):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.

        :param result: Expect the dictionary returned by forward()
        :param labels: GT label for the batch
        """

        w = self.kd_weight
        t = self.temperature

        kd_loss = nn.KLDivLoss()(
            F.log_softmax(result["student_outs"] / t, dim=1),
            F.softmax(result["teacher_outs"] / t, dim=1)
        )
        gt_loss = F.cross_entropy(result["student_outs"], labels)

        # Weigh kd_loss with t^2 to preserve scale of gradients
        final_loss = (kd_loss * w * t * t) + (gt_loss * (1 - w))

        return final_loss

    def train_loss(self, result, labels):
        return self.loss_criterion(result, labels)

    def test_loss(self, result, labels):
        return self.loss_criterion(result, labels)
