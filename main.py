from __future__ import print_function

import argparse
import dataset_utils
import os
import torch

import torch.nn as nn
import torch.optim as optim

from models.basenet import BaseNet
from models.resnet import ResNet18, ResNet34
from models.mobilenetv2 import MobileNetV2
from models.squeezenet import SqueezeNet
from model_utils import train_model, test_model


MODEL_DIR = "../model_ckpt/"


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='SVL coding task')
    parser.add_argument('--model', type=str, required=True,
                        help='specifies the model to use (basenet, resnet, mobnet2, sqnet))')
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--perform-data-aug', type=str2bool, default=False,
                        help='whether to perform data augmentation')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--notes', type=str, default="",
                        help='Additional notes for current exp')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Only support CPU as a device
    device = torch.device("cpu")

    in_ch, num_classes = 3, 100
    train_loader, test_loader = dataset_utils.fetch_cifar100_dataloaders(args)

    if args.model == "basenet":
        model = BaseNet(in_ch, num_classes)
    elif args.model == "resnet18":
        model = ResNet18(in_ch, num_classes)
    elif args.model == "resnet34":
        model = ResNet34(in_ch, num_classes)
    elif args.model == "mobnet2":
        model = MobileNetV2(in_ch, num_classes)
    elif args.model == "sqnet":
        model = SqueezeNet(in_ch, num_classes)
    else:
        assert False, "Unsupported base model: {}".format(args.model)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_loss_criterion = nn.CrossEntropyLoss(reduction='mean')
    test_loss_criterion = nn.CrossEntropyLoss(reduction='sum')

    for epoch in range(1, args.epochs + 1):
        train_model(model, device, train_loader, optimizer, train_loss_criterion, epoch)
        test_model(model, device, test_loader, test_loss_criterion, ks=[1, 3, 5])

        # Save model during each epoch
        if args.save_model:
            torch.save(
                model,
                os.path.join(
                    MODEL_DIR,
                    args.model + "_" + args.notes,
                    "cifar100_" + str(epoch) + ".pt"
                )
            )


if __name__ == '__main__':
    main()
