#!/bin/bash

for i in 1. 0.75 0.5 0.25 0.
do
 CUDA_VISIBLE_DEVICES=0, python3 main.py --dataset CIFAR10 --patch-size 32 --batch-size 128 --architecture DenseNet --epochs 90 --num-networks 5 --optimizer-type SGD -lr 1e-2 --weight-decay 5e-4 --scheduler-type StepLR --step-size 30 --step-gamma 0.3 --randomize-labels -cp $i --device-id 2 --train-networks --visualize-results
done