import dataset_utils
import glob
import os
import torch

import torch.nn as nn
import torch.optim as optim

from kd_module import OnTheFlyKnowledgeDistiller
from models.basenet import BaseNet
from models.resnet import ResNet18, ResNet34
from models.mobilenetv2 import MobileNetV2
from models.squeezenet import SqueezeNet
from model_utils import train_model, test_model


def fetch_specified_model(model_name):
    """
    Inits and returns the specified model
    """

    # Specific hard-coding for CIFAR100
    in_ch, num_classes = 3, 100

    if model_name == "basenet":
        model = BaseNet(in_ch, num_classes)
    elif model_name == "resnet18":
        model = ResNet18(in_ch, num_classes)
    elif model_name == "resnet34":
        model = ResNet34(in_ch, num_classes)
    elif model_name == "mobnet2":
        model = MobileNetV2(in_ch, num_classes)
    elif model_name == "sqnet":
        model = SqueezeNet(in_ch, num_classes)
    else:
        assert False, "Unsupported base model: {}".format(model_name)
    
    return model


def fetch_pretrained_model(ckpt_pth):
    """
    Loads and returns a model from the ckpt
    """
    return torch.load(ckpt_pth)


def load_pretrained_ckpt_if_exists(model, model_save_dir):
    """
    Loads a pretrained ckpt if it exists
    """

    def get_epoch(x):
        return int(x.strip(".pt").split('_')[-1])

    latest_ckpt_list = \
        sorted(
            glob.glob(os.path.join(model_save_dir, "*.pt")),
            key=lambda x: get_epoch(x),
            reverse=True
        )

    if len(latest_ckpt_list) == 0:
        print("Pretrained model weights not found: {}".format(model_save_dir))
        return model, 0

    latest_ckpt = latest_ckpt_list[0]

    print("Using pretrained model weights: {}".format(latest_ckpt))
    model_ckpt = torch.load(latest_ckpt)
    model.load_state_dict(model_ckpt.state_dict())

    epoch = get_epoch(latest_ckpt)

    return model, epoch


def perform_single_model_training(args):
    """
    Helper function to perform training of a single specified model

    :param args: Details regarding the base model and training outline
    """

    # Only support CPU as a device
    device = torch.device("cpu")

    train_loader, test_loader = dataset_utils.fetch_cifar100_dataloaders(args)

    model_save_dir = os.path.join(args.model_dir, args.model + "_" + args.notes)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    model = fetch_specified_model(args.model)
    model, existing_epoch = load_pretrained_ckpt_if_exists(model, model_save_dir)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loss_criterion = nn.CrossEntropyLoss(reduction='mean')
    test_loss_criterion = nn.CrossEntropyLoss(reduction='sum')

    for epoch in range(existing_epoch, args.epochs):
        train_model(model, device, train_loader, optimizer, train_loss_criterion, epoch)
        test_model(model, device, test_loader, test_loss_criterion, ks=[1, 3, 5])

        # Save model during each epoch
        if args.save_model:
            torch.save(
                model,
                os.path.join(model_save_dir, "cifar100_" + str(epoch) + ".pt")
            )


def perform_knowledge_distillation(args):
    """
    Helper function to perform knowledge distillation using the specified students and teachers

    :param args: Details regarding the base model and training outline
    """

    # Only support CPU as a device
    device = torch.device("cpu")

    train_loader, test_loader = dataset_utils.fetch_cifar100_dataloaders(args)

    model_save_dir = os.path.join(args.model_dir, args.model + "_" + args.notes)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    student = fetch_specified_model(args.student_model)
    teacher = fetch_pretrained_model(args.teacher_ckpt_pth)

    model = OnTheFlyKnowledgeDistiller(student, teacher, args)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train_model(model, device, train_loader, optimizer, train_loss_criterion, epoch)
        test_model(model, device, test_loader, test_loss_criterion, ks=[1, 3, 5])

        # Save model during each epoch
        if args.save_model:
            torch.save(
                model,
                os.path.join(model_save_dir, "cifar100_" + str(epoch) + ".pt")
            )
