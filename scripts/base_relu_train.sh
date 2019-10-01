#!/usr/bin/bash

notes="base_relu_30sep"
epochs=10
data_aug=true
task="base_tr"
activation="relu"

# Train different models sequentially
#python main.py --perform-data-aug $data_aug --notes $notes --epochs $epochs --task $task --base-model basenet --activation $activation
#python main.py --perform-data-aug $data_aug --notes $notes --epochs $epochs --task $task --base-model resnet18 --activation $activation
python main.py --perform-data-aug $data_aug --notes $notes --epochs $epochs --task $task --base-model sqnet --activation $activation
#python main.py --perform-data-aug $data_aug --notes $notes --epochs $epochs --task $task --base-model mobnet2 --activation $activation
#python main.py --perform-data-aug $data_aug --notes $notes --epochs $epochs --task $task --base-model resnet34 --activation $activation
