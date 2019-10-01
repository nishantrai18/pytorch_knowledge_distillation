#!/usr/bin/bash

python main.py --perform-data-aug True --notes t_2_kd_0.5_1oct --task kd_cached --student-model basenet --activation swish --teachers resnet18.re --temperature 2.0 --kd-weight 0.5 --epoch 5
python main.py --perform-data-aug True --notes t_5_kd_0.5_1oct --task kd_cached --student-model basenet --activation swish --teachers resnet18.re --temperature 5.0 --kd-weight 0.5 --epoch 5
python main.py --perform-data-aug True --notes t_10_kd_0.5_1oct --task kd_cached --student-model basenet --activation swish --teachers resnet18.re --temperature 10.0 --kd-weight 0.5 --epoch 5
python main.py --perform-data-aug True --notes t_5_kd_0.1_1oct --task kd_cached --student-model basenet --activation swish --teachers resnet18.re --temperature 5.0 --kd-weight 0.1 --epoch 5
python main.py --perform-data-aug True --notes t_5_kd_0.9_1oct --task kd_cached --student-model basenet --activation swish --teachers resnet18.re --temperature 5.0 --kd-weight 0.9 --epoch 5
