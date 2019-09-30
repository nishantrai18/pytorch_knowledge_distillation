import cached_dataset
import dataset_utils
import glob
import os
import torch

import torch.nn as nn
import torch.optim as optim

from model_wrapper import IndividualModel, OnTheFlyKDModel
from models.basenet import BaseNet
from models.resnet import ResNet18, ResNet34
from models.mobilenetv2 import MobileNetV2
from models.squeezenet import SqueezeNet
from model_utils import ModelTrainer


# Script constants
##################

# Only support CPU as a device
DEVICE = torch.device("cpu")


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
        return int(x.replace(".pt", '').split('_')[-1])

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


def perform_training(model, train_loader, test_loader, model_save_dir, args):
    """
    General utility to perform training of the passed model
    """

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model_trainer = ModelTrainer(model, DEVICE, train_loader, test_loader, optimizer)

    for epoch in range(args.epochs):
        # Perform train and test step
        model_trainer.train_step(epoch)
        model_trainer.test_step(ks=[1, 3, 5])

        # Save model during each epoch
        if args.save_model:
            model_trainer.save_model(model_save_dir, epoch)


def perform_single_model_training(args):
    """
    Helper function to perform training of a single specified model

    :param args: Details regarding the base model and training outline
    """

    train_loader, test_loader = dataset_utils.fetch_cifar100_dataloaders(args)

    model_save_dir = os.path.join(args.model_dir, args.base_model + "_" + args.notes)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    existing_epoch = 0

    model = fetch_specified_model(args.base_model)
    if args.preload_weights:
        model, existing_epoch = load_pretrained_ckpt_if_exists(model, model_save_dir)

    model = model.to(DEVICE)
    model = IndividualModel(model)

    perform_training(model, train_loader, test_loader, model_save_dir, args)


def perform_knowledge_distillation_on_the_fly(args):
    """
    Helper function to perform knowledge distillation using the specified students and teachers

    :param args: Details regarding the base model and training outline
    """

    train_loader, test_loader = dataset_utils.fetch_cifar100_dataloaders(args)

    model_save_dir = os.path.join(args.model_dir, args.student_model + "_otf_kd_" + args.notes)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    student = fetch_specified_model(args.student_model).to(DEVICE)
    student = IndividualModel(student)
    teacher = fetch_pretrained_model(args.teacher_ckpt_pth).to(DEVICE)
    model = OnTheFlyKDModel(student, teacher, args)

    perform_training(model, train_loader, test_loader, model_save_dir, args)


def perform_cached_knowledge_distillation(args):
    """
    Similar to the above, but performs it using the cached benchmark
    """

    train_loader, test_loader = cached_dataset.fetch_cifar100_efficient_kd_dataloaders(args)

    model_save_dir = os.path.join(args.model_dir, args.student_model + "_cached_kd_" + args.notes)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    student = fetch_specified_model(args.student_model).to(DEVICE)
    student = IndividualModel(student)

    perform_training(student, train_loader, test_loader, model_save_dir, args)
