#!/bin/bash

# Path to dataset
DATA=data/

# Classes: [airplane, bag, cap, car, chair, earphone, guitar, knife, lamp, laptop, motorbike, mug, pistol, rocket, skateboard, table]
CLASS=bag

export CUDA_VISIBLE_DEVICES=4
python main.py \
--modelname ViT-B/16 \
--classchoice ${CLASS} \
--datasetpath ${DATA}
