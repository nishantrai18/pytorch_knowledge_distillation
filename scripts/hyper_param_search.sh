#!/usr/bin/bash

python main.py --perform-data-aug True --notes t_2_kd_0.5_1oct --task kd_cached --student-model basenet --activation swish --teachers resnet18.re --temperature 2.0 --kd-weight 0.5 --epoch 6
python main.py --perform-data-aug True --notes t_5_kd_0.5_1oct --task kd_cached --student-model basenet --activation swish --teachers resnet18.re --temperature 5.0 --kd-weight 0.5 --epoch 6
python main.py --perform-data-aug True --notes t_10_kd_0.5_1oct --task kd_cached --student-model basenet --activation swish --teachers resnet18.re --temperature 10.0 --kd-weight 0.5 --epoch 6
python main.py --perform-data-aug True --notes t_5_kd_0.1_1oct --task kd_cached --student-model basenet --activation swish --teachers resnet18.re --temperature 5.0 --kd-weight 0.1 --epoch 6
python main.py --perform-data-aug True --notes t_5_kd_0.9_1oct --task kd_cached --student-model basenet --activation swish --teachers resnet18.re --temperature 5.0 --kd-weight 0.9 --epoch 6

# Defunct learning for lr >= 0.01, slow learning for lr <= 0.0001. 0.001 - 0.005 is okay
python main.py --perform-data-aug True --notes t_5_kd_0.5_lr_0.01_1oct --task kd_cached --student-model basenet --activation swish --teachers resnet18.re --temperature 5.0 --kd-weight 0.5 --epoch 15 --lr 0.01
python main.py --perform-data-aug True --notes t_5_kd_0.5_lr_0.1_1oct --task kd_cached --student-model basenet --activation swish --teachers resnet18.re --temperature 5.0 --kd-weight 0.5 --epoch 15 --lr 0.1
python main.py --perform-data-aug True --notes t_5_kd_0.5_lr_0.0001_1oct --task kd_cached --student-model basenet --activation swish --teachers resnet18.re --temperature 5.0 --kd-weight 0.5 --epoch 15 --lr 0.0001
