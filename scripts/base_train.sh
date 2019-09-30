#!/usr/bin/bash

# Train different models sequentially
python main.py --perform-data-aug True --notes fresh_30sep --epochs 15 --task base_tr --base-model basenet --preload-weights True
python main.py --perform-data-aug True --notes fresh_30sep --epochs 15 --task base_tr --base-model resnet18 --preload-weights True
python main.py --perform-data-aug True --notes fresh_30sep --epochs 15 --task base_tr --base-model resnet34 --preload-weights True
python main.py --perform-data-aug True --notes fresh_30sep --epochs 15 --task base_tr --base-model sqnet --preload-weights True
python main.py --perform-data-aug True --notes fresh_30sep --epochs 15 --task base_tr --base-model mobnet2 --preload-weights True
