from __future__ import print_function

import argparse
import torch
import training_utils as tu


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args_parser():
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
    parser.add_argument('--preload-weights', type=str2bool, default=False,
                        help='whether to preload if weights exist')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--notes', type=str, default="",
                        help='Additional notes for current exp')
    parser.add_argument('--model-dir', type=str, default="../model_ckpt/",
                        help='Additional notes for current exp')

    return parser


def main():

    parser = get_args_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    tu.perform_single_model_training(args)


if __name__ == '__main__':
    main()
