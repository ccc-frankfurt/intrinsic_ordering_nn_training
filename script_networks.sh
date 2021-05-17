#!/bin/bash

for arc in LeNet5 VGG ResNet DenseNet
do
  # parameters for PASCAL VOCDetection dataset
 # CUDA_VISIBLE_DEVICES=3, python3 main.py --dataset VOCDetection --patch-size 224 --batch-size 128 --architecture $arc -lr 1e-1 --epochs 150 --num-networks 5 --scheduler StepLR --step-gamma 0.2 --step-size 50 --train-networks --multilabel --weight-decay 5e-4 --visualize-results --agreement-type exact_match
CUDA_VISIBLE_DEVICES=0, python3 main.py --dataset CIFAR10 --patch-size 32 --batch-size 128 --architecture $arc -lr 1e-1 --epochs 60 --num-networks 5 --scheduler CosineAnnealingLR --train-networks --weight-decay 5e-4 --visualize-results
#CUDA_VISIBLE_DEVICES=0, python3 main.py --dataset KTH_TIPS --patch-size 190 --batch-size 64 --architecture $arc --optimizer-type Adam -lr 1e-4 --epochs 60 --num-networks 5 --scheduler OneCycleLR --train-networks --weight-decay 1e-5 --visualize-results
done