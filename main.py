from __future__ import print_function

import args as ag
import torch
import training_utils as tu


def main():

    parser = ag.get_args_parser_for_driver()
    args = parser.parse_args()

    torch.manual_seed(42)

    print("Arguments passed:", args)

    if args.task == "base_tr":
        tu.perform_single_model_training(args)
    elif args.task == "kd":
        tu.perform_knowledge_distillation(args)
    else:
        print("Performing nothing")


if __name__ == '__main__':
    main()
