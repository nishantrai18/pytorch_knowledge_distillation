from __future__ import print_function

import argparse
import dataset_utils
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Net(nn.Module):
    def __init__(self, in_ch, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(5*5*50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    tq = tqdm(train_loader, desc="Steps within train epoch {}:".format(epoch))
    for batch_idx, (data, target) in enumerate(tq):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        tq.set_postfix({"loss": loss.item()})


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='SVL coding task')
    parser.add_argument('--dataset', type=str, default="cifar100",
                        help='Specifies the dataset (mnist, cifar100 - default)')
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
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Only support CPU as a device
    device = torch.device("cpu")

    # Set the relevant information based on the dataset
    if args.dataset == "mnist":
        in_ch, num_classes = 1, 10
        dataset_loader = dataset_utils.fetch_mnist_dataloaders
    elif args.dataset == "cifar100":
        in_ch, num_classes = 3, 100
        dataset_loader = dataset_utils.fetch_cifar100_dataloaders
    else:
        assert False, "Incorrect dataset {} provided".format(args.dataset)

    train_loader, test_loader = dataset_loader(args)

    model = Net(in_ch, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), args.dataset + "_cnn.pt")


if __name__ == '__main__':
    main()
