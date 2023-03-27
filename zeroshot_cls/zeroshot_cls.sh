#!/bin/bash

# Path to dataset
DATASET=modelnet40
# DATASET=scanobjectnn

TRAINER=PointCLIPV2_ZS
# Trainer configs: rn50, rn101, vit_b32 or vit_b16
CFG=vit_b16

export CUDA_VISIBLE_DEVICES=0
python main.py \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--no-train \
--zero-shot \
--post-search
