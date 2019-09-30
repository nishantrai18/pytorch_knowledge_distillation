import torch

from torchvision import datasets, transforms


# Util helpers for dataset specific transformations
###################################################

def add_data_augmentation(transform_list):
    # Add randomized cropping
    transform_list.append(
        transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.98, 1.02))
    )
    # Add random flipping
    transform_list.append(transforms.RandomHorizontalFlip())
    return transform_list


def extend_base_mnist_transforms(transform_list):
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return transform_list


def extend_base_cifar100_transforms(transform_list):
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
    return transform_list


# Data loader utilities
#######################


def fetch_mnist_dataloaders(args):

    # Prepare the transforms
    train_transform_list, test_transform_list = [], []

    # Only add data augmentation to train transforms
    if args.perform_data_aug:
        train_transform_list = add_data_augmentation(train_transform_list)

    # Add the base transformations
    train_transform_list = extend_base_mnist_transforms(train_transform_list)
    test_transform_list = extend_base_mnist_transforms(test_transform_list)

    train_loader = \
        torch.utils.data.DataLoader(
            datasets.MNIST(
                '../data',
                train=True,
                download=True,
                transform=transforms.Compose(train_transform_list)
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )
    test_loader = \
        torch.utils.data.DataLoader(
            datasets.MNIST(
                '../data',
                train=False,
                transform=transforms.Compose(test_transform_list)
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )

    return train_loader, test_loader


def fetch_cifar100_dataloaders(args, num_workers=4):

    # Prepare the transforms
    train_transform_list, test_transform_list = [], []

    # Only add data augmentation to train transforms
    if args.perform_data_aug:
        train_transform_list = add_data_augmentation(train_transform_list)

    # Add the base transformations
    train_transform_list = extend_base_cifar100_transforms(train_transform_list)
    test_transform_list = extend_base_cifar100_transforms(test_transform_list)

    train_loader = \
        torch.utils.data.DataLoader(
            datasets.CIFAR100(
                '../data',
                train=True,
                download=True,
                transform=transforms.Compose(train_transform_list)
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers
        )
    test_loader = \
        torch.utils.data.DataLoader(
            datasets.CIFAR100(
                '../data',
                train=False,
                transform=transforms.Compose(test_transform_list)
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers
        )

    return train_loader, test_loader

