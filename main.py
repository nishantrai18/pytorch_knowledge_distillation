from __future__ import print_function

import argparse
import dataset_utils
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.basenet import BaseNet
from models.resnet import ResNet18, ResNet34
from models.mobilenetv2 import MobileNetV2
from models.squeezenet import SqueezeNet
from tqdm import tqdm


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train(model, device, train_loader, optimizer, loss_criterion, epoch):
    model.train()
    tq = tqdm(train_loader, desc="Steps within train epoch {}:".format(epoch))
    for batch_idx, (data, target) in enumerate(tq):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_criterion(output, target)
        loss.backward()
        optimizer.step()
        tq.set_postfix({"loss": loss.item()})


def test(model, device, test_loader, loss_criterion):
    model.eval()
    test_loss = 0
    correct_top_1, correct_top_5 = 0, 0
    with torch.no_grad():
        tq = tqdm(test_loader, desc="Steps within test:")
        for data, target in tq:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += loss_criterion(output, target).item()
            # get the probable classes
            preds = torch.topk(output, k=5)[1]
            corrects = preds.eq(target.view(-1, 1).expand_as(preds))
            correct_top_1 += corrects[:, :1].sum().item()
            correct_top_5 += corrects[:, :5].sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy Top1: {}/{} ({:.1f}%), Accuracy Top5: {}/{} ({:.1f}%)\n".format(
            test_loss,
            correct_top_1, len(test_loader.dataset),
            100. * correct_top_1 / len(test_loader.dataset),
            correct_top_5, len(test_loader.dataset),
            100. * correct_top_5 / len(test_loader.dataset)
        )
    )


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
        train(model, device, train_loader, optimizer, train_loss_criterion, epoch)
        test(model, device, test_loader, test_loss_criterion)

    if args.save_model:
        torch.save(model.state_dict(), args.model + "_cifar100.pt")


if __name__ == '__main__':
    main()
