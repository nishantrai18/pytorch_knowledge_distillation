#!/usr/bin/bash

# Train different models sequentially
python main.py --perform-data-aug True --notes fresh_30sep --epochs 50 --task base_tr --base-model basenet
python main.py --perform-data-aug True --notes fresh_30sep --epochs 50 --task base_tr --base-model resnet18
python main.py --perform-data-aug True --notes fresh_30sep --epochs 50 --task base_tr --base-model resnet34
python main.py --perform-data-aug True --notes fresh_30sep --epochs 50 --task base_tr --base-model sqnet
python main.py --perform-data-aug True --notes fresh_30sep --epochs 50 --task base_tr --base-model mobnet2
