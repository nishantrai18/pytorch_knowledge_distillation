"""
Contains all argparser templates
"""

import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser_for_driver():

    parser = argparse.ArgumentParser(description='SVL coding task')

    # General training settings
    parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--model-dir', type=str, default="../model_ckpt/",
                        help='Additional notes for current exp')
    parser.add_argument('--notes', type=str, default="",
                        help='Additional notes for current exp')

    # Training behavior flags
    parser.add_argument('--perform-data-aug', type=str2bool, default=False,
                        help='whether to perform data augmentation')
    parser.add_argument('--preload-weights', type=str2bool, default=False,
                        help='whether to preload if weights exist')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')

    # Flag specifying task to perform
    parser.add_argument('--task', type=str, required=True,
                        help='specifies what to perform (base_tr, kd)')

    # Flags for individual model training
    parser.add_argument('--base-model', type=str, default="",
                        help='specifies the model to use (basenet, resnet, mobnet2, sqnet)')

    # Flags for knowledge distillation training
    parser.add_argument('--student-model', type=str, default="",
                        help='specifies the student model')
    parser.add_argument('--teacher-ckpt-pth', type=str, default="",
                        help='specifies the teacher model')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature hyperparameter for KD')
    parser.add_argument('--kd-weight', type=float, default=0.5,
                        help='kd-weight hyperparameter for KD')

    # Flag for specifying On-The-Fly KD vs the optimized one
    parser.add_argument('--kd-variant', type=str, default="something",
                        help='specifies which variant of KD to use')

    return parser


def get_args_parser_for_cached_dataset():

    parser = argparse.ArgumentParser(description='Flags for cached dataset gen')

    parser.add_argument('--train-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--perform-data-aug', type=str2bool, default=False,
                        help='whether to perform data augmentation')

    return parser
