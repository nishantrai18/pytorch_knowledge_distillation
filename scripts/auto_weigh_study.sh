#!/usr/bin/bash

python main.py --perform-data-aug True --notes sq_rn_1oct --task kd_cached --student-model basenet --activation swish --teachers sqnet.sw_resnet18.re --auto-weigh false
#python main.py --perform-data-aug True --notes sq_rn_auto_weigh_1oct --task kd_cached --student-model basenet --activation swish --teachers sqnet.sw_resnet18.re --auto-weigh true
#python main.py --perform-data-aug True --notes sq_rn_1oct --task kd_cached --student-model basenet --activation relu --teachers sqnet.sw_resnet18.re --auto-weigh false
#python main.py --perform-data-aug True --notes sq_rn_auto_weigh_1oct --task kd_cached --student-model basenet --activation relu --teachers sqnet.sw_resnet18.re --auto-weigh true