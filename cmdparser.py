"""
Command line argument options parser.
Adopted and modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py
and https://github.com/MrtnMndt/OCDVAEContinualLearning/blob/master/lib/cmdparser.py
Usage with two minuses "- -". Options are written with a minus "-" in command line, but
appear with an underscore "_" in the attributes' list.
"""

import argparse

parser = argparse.ArgumentParser(description='PyTorch Intrinsic Ordering Dataset Metrics Correlation')

# Dataset and loading
parser.add_argument('--dataset', default='CIFAR10', help='name of dataset')
parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('-p', '--patch-size', default=32, type=int, help='patch size for crops (default: 32)')

parser.add_argument('--multilabel', default=False, action='store_true', help='if dataset has multiple'
                                                                        ' labels per image')
parser.add_argument('--save-prob-vector', default=False, action='store_true', help='for single label:'
                                                                            'save whole prob vector and not only max')
parser.add_argument('--randomize-labels', default=False, action='store_true',
                    help='randomize labels for every batch')
parser.add_argument('-cp', '--corrupt-prob', default=0., type=float, help='label corruption/randomization probability/level (default: 0.)')

# Architecture and weight-init
parser.add_argument('-a', '--architecture', default='Net', help='model architecture')
parser.add_argument('--weight-init', default='kaiming-normal',
                    help='weight-initialization scheme (default: kaiming-normal)')

# Training hyper-parameters
parser.add_argument('--device-id', default=0, type=int, help='gpu device id on which to train')
parser.add_argument('--num-networks', default=5, type=int, help='number of networks to train')
parser.add_argument('--epochs', default=120, type=int, help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('-lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate (default: 1e-3)')
parser.add_argument('--sgd-momentum', default=0.9, type=float, help='SGD momentum (default: 0.9)')
parser.add_argument('-bn', '--batch-norm', default=1e-5, type=float, help='batch normalization (default 1e-5)')
parser.add_argument('-wd', '--weight-decay', default=1e-5, type=float, help='weight decay (default 1e-5)')

parser.add_argument('--optimizer-type', default='SGD', help='Adam or SGD')
parser.add_argument('--scheduler-type', default='', help='scheduler type: CosineAnnealingLR or OneCycleLR')
parser.add_argument('--eta-min',  default=5e-4, type=float, help='min learning rate (default: 5e-4)')
parser.add_argument('--step-gamma', default=0.2, type=float, help='StepLR: factor to reduce lr every step-size steps (default: 0.2)')
parser.add_argument('--step-size', default=10, type=float, help='StepLR: step-size (default: 10)')

# Resuming training
parser.add_argument('--resume', default=False, action='store_true',
                    help='if resume training')

# Computation steps
parser.add_argument('--save-dir', default='', help='directory with results to save/saved results')
parser.add_argument('--compute-dataset-metrics', default=False, action='store_true', help='compute dataset metrics, like entropy')
parser.add_argument('--train-networks', default=False, action='store_true', help='train several networks')
parser.add_argument('--visualize-results', default=False, action='store_true', help='visualize agreement '
                                                                          'and correlation with'
                                                                          'dataset metrics')
parser.add_argument('--agreement-type', default='single_label',
                    help='defines which kind of multilabel agreement/accuracy to use')