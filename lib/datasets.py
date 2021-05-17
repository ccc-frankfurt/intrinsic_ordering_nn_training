# in analogy to https://github.com/MrtnMndt/OCDVAEContinualLearning/blob/master/lib/Datasets/datasets.py

import os
import numpy as np
import glob
from PIL import Image
import pandas as pd
import urllib.request
import tarfile
import sys
import collections
from tqdm import tqdm

import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torchvision.utils import save_image

from lib.helpers.custom_img_folder import IndxImageFolder
from lib.helpers.labels_preprocessing import corrupt_labels
from lib.dataset_metrics import DatasetMetrics


class CIFAR10:
    """
    CIFAR-10 dataset featuring tiny 32x32 color images of
    objects belonging to hundred different classes.
    Dataloader implemented with torchvision.datasets.CIFAR10.
    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.
    Attributes:
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, horizontal flips, random
            translations of up to 10% in each direction and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):
        self.name = 'CIFAR10'
        self.num_classes = 10

        self.randomize_labels = args.randomize_labels
        self.corrupt_prob = args.corrupt_prob

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(
            args.batch_size, args.workers, is_gpu)

        # the indices have to be sorted alphabetically, if saving as pytorch's ImageFolder
        self.class_to_idx = {'airplane': 0,
                             'automobile': 1,
                             'bird': 2,
                             'cat': 3,
                             'deer': 4,
                             'dog': 5,
                             'frog': 6,
                             'horse': 7,
                             'ship': 8,
                             'truck': 9}

        self.idx_to_class = {0: 'airplane',
                             1: 'automobile',
                             2: 'bird',
                             3: 'cat',
                             4: 'deer',
                             5: 'dog',
                             6: 'frog',
                             7: 'horse',
                             8: 'ship',
                             9: 'truck'}

    def get_dataset(self):
        """
        Uses torchvision.datasets.CIFAR10 to load dataset.
        Downloads dataset if doesn't exist already.
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """
        transform = transforms.Compose([
            # you can add other transformations in this list
            transforms.ToTensor()
        ])

        trainset = datasets.CIFAR10('datasets/CIFAR10/train/', train=True, transform=transform,
                                    target_transform=None, download=True)

        valset = datasets.CIFAR10('datasets/CIFAR10/test/', train=False, transform=transform,
                                  target_transform=None, download=True)

        train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=2)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)

        check = 0
        directories = ["datasets/CIFAR10_indx/train", "datasets/CIFAR10_indx/test"]
        for directory in directories:
            if os.path.exists(directory):
                check += 1
            if not os.path.exists(directory):
                os.makedirs(directory)

        if check != len(directories):
            # create folders to save data
            indices_data = {}

            indx = 0
            for loader_name, loader in [('train', train_loader), ('test', val_loader)]:
                for (img, label) in loader:
                    img_name = str(int(label))+'_'+str(int(indx))+'.jpg'

                    directory = os.path.join("datasets/CIFAR10_indx/" + loader_name,
                                             format(int(label), '06d'))
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    save_image(img, os.path.join(directory, img_name))
                    # img_path is the index for the images
                    indices_data[os.path.join(directory, img_name)] = int(label)
                    indx += 1

            # store img_paths which serves as indices and the labels for further analysis
            indices_data = collections.OrderedDict(sorted(indices_data.items()))
            dataframe = pd.DataFrame({'img_paths': list(indices_data.keys()),
                                      'labels': list(indices_data.values())})
            DatasetMetrics.indices_paths(self.name, dataframe)

        if self.randomize_labels:
            evaluation = DatasetMetrics.load_evaluation_metrics(self.name)
            self.random_labels = corrupt_labels(evaluation['labels'].to_dict(),
                                           self.num_classes,
                                           self.corrupt_prob)
        else:
            self.random_labels = None

        # load the image dataset from folder with indices
        trainset = IndxImageFolder(root="datasets/CIFAR10_indx/train", transform = transform,
                                   random_labels=self.random_labels)
        valset = IndxImageFolder(root="datasets/CIFAR10_indx/test", transform=transform,
                                 random_labels=self.random_labels)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset
        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader

class VOCDetection:
    """
    PASCAL VOC 2012 multi-label dataset featuring unequally-sized images of 20 different classes.
    Dataloader implemented with torchvision.datasets.VOCDetection.
    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.
    Attributes:
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, horizontal flips, random
            translations of up to 10% in each direction and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):
        self.name = 'VOCDetection'
        self.num_classes = 20
        self.args = args

        self.load_path = 'datasets/VOCDetection'
        self.save_path = 'datasets/VOCDetection_indx'

        # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit
        # Person: person
        # Animal: bird, cat, cow, dog, horse, sheep
        # Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
        # Indoor: bottle, chair, dining table, potted plant, sofa, tv / monitor

        self.class_to_idx = {'person': 0,
                             'bird': 1,
                             'cat': 2,
                             'cow': 3,
                             'dog': 4,
                             'horse': 5,
                             'sheep': 6,
                             'aeroplane': 7,
                             'bicycle': 8,
                             'boat': 9,
                             'bus': 10,
                             'car': 11,
                             'motorbike': 12,
                             'train': 13,
                             'bottle': 14,
                             'chair': 15,
                             'diningtable': 16,
                             'pottedplant': 17,
                             'sofa': 18,
                             'tvmonitor': 19,
                             }

        self.idx_to_class = {0: 'person',
                             1: 'bird',
                             2: 'cat',
                             3: 'cow',
                             4: 'dog',
                             5: 'horse',
                             6: 'sheep',
                             7: 'aeroplane',
                             8: 'bicycle',
                             9: 'boat',
                             10: 'bus',
                             11: 'car',
                             12: 'motorbike',
                             13: 'train',
                             14: 'bottle',
                             15: 'chair',
                             16: 'diningtable',
                             17: 'pottedplant',
                             18: 'sofa',
                             19: 'tvmonitor'}

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers, is_gpu)

    def get_dataset(self):
        """
        Uses torchvision.datasets.VOCDetection to load dataset.
        Downloads dataset if doesn't exist already.
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        # https://developer.nvidia.com/blog/preparing-state-of-the-art-models-for-classification-and-object-detection-with-tlt/
        train_download = not os.path.exists(os.path.join(self.load_path, "train"))
        trainval_2012 = datasets.VOCDetection(os.path.join(self.load_path, "train"), image_set='trainval',
                                         transform=transforms.Compose([transforms.ToTensor()]),
                                         target_transform=None, download=train_download)
        trainval_2007 = datasets.VOCDetection(os.path.join(self.load_path, "train"), image_set='trainval',
                                              year='2007',
                                              transform=transforms.Compose([transforms.ToTensor()]),
                                              target_transform=None, download=train_download)
        test_download = not os.path.exists(os.path.join(self.load_path, "test"))
        valset = datasets.VOCDetection(os.path.join(self.load_path, "test"), image_set='test',
                                       year='2007',
                                       transform=transforms.Compose([transforms.ToTensor()]),
                                       target_transform=None, download=test_download)
        train_loader_2007 = torch.utils.data.DataLoader(trainval_2007, batch_size=1, shuffle=False, num_workers=2)
        train_loader_2012 = torch.utils.data.DataLoader(trainval_2012, batch_size=1, shuffle=False, num_workers=2)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=2)

        check = 0
        directories = [os.path.join(self.save_path, "train"), os.path.join(self.save_path, "test")]
        for directory in directories:
            if os.path.exists(directory):
                check += 1
            if not os.path.exists(directory):
                os.makedirs(directory)

        if check != len(directories):
            indices_data = {}
            # create folders to save data
            for loader_name, loader in [('train', train_loader_2007),
                                        ('train', train_loader_2012),
                                        ('test', val_loader)]:
                for (img, annotation) in tqdm(loader):

                    #print(annotation)
                    # there may be multiple labels, they are concatenated to: 'label1_label2_'
                    label = ''
                    int_label = []

                    elems = annotation['annotation']['object']
                    # if only 1 label - it is a dictionary, but not list of dictionaries
                    # for consistency reasons and to be able to use the loop later
                    if not isinstance(elems, list):
                        elems = [elems]

                    # get bboxes, compute object size, add all object sizes and divide by img size (h*w)
                    obj_sizes = 0
                    num_instances = 0

                    for elem in elems:
                        # every name is in a list
                        # there may be multiple instances of the same object
                        # those are disregarded for label

                        if not (bool(int(elem['difficult'][0])) and loader_name == 'test'):
                            if not str(self.class_to_idx[elem['name'][0]]) in label:
                                label += str(self.class_to_idx[elem['name'][0]]) + '_'
                                int_label.append(self.class_to_idx[elem['name'][0]])

                                num_instances += 1
                                # percentage of objects in the image: sum obj_size/img_size
                                obj_sizes += (int(elem['bndbox']['xmax'][0]) - int(elem['bndbox']['xmin'][0])) * \
                                             (int(elem['bndbox']['ymax'][0]) - int(elem['bndbox']['ymin'][0]))
                                obj_sizes /= float(int(annotation['annotation']['size']['width'][0]) *
                                                   int(annotation['annotation']['size']['height'][0]))

                    img_name = label + '_' + annotation['annotation']['filename'][0]

                    directory = os.path.join(os.path.join(self.save_path, loader_name), label)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    save_image(img, os.path.join(directory, img_name))

                    indices_data[os.path.join(directory, img_name)] = (int_label,
                                              obj_sizes, num_instances)

            # store img_paths which serves as indices and the labels for further analysis
            indices_data = collections.OrderedDict(sorted(indices_data.items()))

            dataframe = pd.DataFrame({'img_paths': list(indices_data.keys()),
                                      'labels': np.array(list(indices_data.values()), dtype=object)[:, 0],
                                      'obj_sizes': np.array(list(indices_data.values()), dtype=object)[:, 1],
                                      'num_instances': np.array(list(indices_data.values()), dtype=object)[:, 2]})
            DatasetMetrics.indices_paths(self.name, dataframe)

        train_transform = transforms.Compose([
            # you can add other transformations in this list
            # resize (256 x remaining larger size) and RandomCrop(224)
            # like in https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
            # https://arxiv.org/pdf/1409.1556.pdf
            transforms.Resize(256),  # resize smaller size to 256
            transforms.RandomCrop(self.args.patch_size),  # 224
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            # you can add other transformations in this list
            # resize (256 x remaining larger size) and RandomCrop(224)
            transforms.Resize(256),  # resize smaller size to 256
            transforms.CenterCrop((self.args.patch_size, self.args.patch_size)),  # 224
            transforms.ToTensor()
        ])

        if self.args.compute_dataset_metrics is True:
            # when computing dataset metrics, an original image should be used
            # - without randomness of RandomCrop
            train_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor()
            ])

            # if not already set, set batch-size to 1 for computing the metrics
            # due to different image sizes
            self.args.batch_size = 1

        # load the image dataset from folder with indices
        trainset = IndxImageFolder(root = os.path.join(self.save_path, "train"), transform=train_transform,
                                   num_classes=len(self.class_to_idx), multilabel=self.args.multilabel)
        valset = IndxImageFolder(root=os.path.join(self.save_path, "test"), transform=test_transform,
                                 num_classes=len(self.class_to_idx), multilabel=self.args.multilabel)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset
        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader

class ImageNet:
    """
    ImageNet dataset featuring unequally-sized images of 1000 different classes.
    Dataloader implemented with torchvision.datasets.ImageNet.
    On how to train ImageNet with DenseNet: https://arxiv.org/pdf/1608.06993.pdf

    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.
    Attributes:
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, horizontal flips, random
            translations of up to 10% in each direction and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):
        self.name = 'ImageNet'
        self.num_classes = 1000
        self.args = args

        self.load_path = '/home/data/ILSVRC12'

        self.idx_to_class = {}
        self.class_to_idx = {}

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size, args.workers,
                                                                     is_gpu)

    def get_dataset(self):
        """
        Downloads dataset if doesn't exist already.
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_transform = transforms.Compose([
            # you can add other transformations in this list
            # resize (256 x remaining larger size) and RandomCrop(224)
            # like in https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
            # https://arxiv.org/pdf/1409.1556.pdf
            transforms.Resize(256),  # resize smaller size to 256
            transforms.RandomCrop(self.args.patch_size),  # 224
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            # you can add other transformations in this list
            # resize
            transforms.Resize(256),  # resize smaller size to 256
            transforms.CenterCrop((self.args.patch_size, self.args.patch_size)),  # 224
            transforms.ToTensor()
        ])

        if self.args.compute_dataset_metrics is True:

            train_transform = transforms.Compose([
                transforms.Resize((self.args.patch_size, self.args.patch_size)),  # resize smaller size to 256
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.Resize((self.args.patch_size, self.args.patch_size)),  # resize smaller size to 256
                transforms.ToTensor()
            ])

            '''
            train_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor()
            ])'''

            # if not already set, set batch-size to 1 for computing the metrics
            # due to different image sizes
            self.args.batch_size = 1
            self.args.workers = 0

        trainset = IndxImageFolder(os.path.join(self.load_path, "train"),
                                         transform=train_transform,
                                         target_transform=None)

        valset = IndxImageFolder(os.path.join(self.load_path, "val"),
                                       transform=test_transform,
                                       target_transform=None)

        # https://pytorch.org/vision/0.8/_modules/torchvision/datasets/imagenet.html#ImageNet
        # https://www.kaggle.com/c/imagenet-object-localization-challenge/data?select=LOC_synset_mapping.txt
        # https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
        self.class_to_idx = trainset.class_to_idx
        self.idx_to_class = dict((v, k) for k, v in self.class_to_idx.items())

        if not os.path.exists(os.path.join('metrics/datasets', self.name) +'.csv'):
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)

            indices_data = {}
            for loader_name, loader in [('train', train_loader), ('test', val_loader)]:
                for (img, label, img_path) in tqdm(loader):
                    # since img_path is a tuple, it is converted to a list
                    # since batch-size is 1, the first element is taken
                    indices_data[list(img_path)[0]] = label.detach().cpu().numpy()[0]

                # store img_paths which serves as indices and the labels for further analysis
            indices_data = collections.OrderedDict(sorted(indices_data.items()))

            dataframe = pd.DataFrame({'img_paths': list(indices_data.keys()),
                                      'labels': list(indices_data.values())})
            DatasetMetrics.indices_paths(self.name, dataframe)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset
        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader

class KTH_TIPS:
    """
    KTH-TIPS2b dataset featuring 200x200 color texture image patches belonging to 11 different
    textures with several rotation, scale and lightning conditions.
    https://www.csc.kth.se/cvap/databases/kth-tips/download.html

    VGG may be overfitting due to small size: https://arxiv.org/pdf/1707.07394v1.pdf
    Parameters:
        args (dict): Dictionary of (command line) arguments.
            Needs to contain batch_size (int) and workers(int).
        is_gpu (bool): True if CUDA is enabled.
            Sets value of pin_memory in DataLoader.
    Attributes:
        train_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor, horizontal flips, random
            translations of up to 10% in each direction and normalization.
        val_transforms (torchvision.transforms): Composition of transforms
            including conversion to Tensor and normalization.
        trainset (torch.utils.data.TensorDataset): Training set wrapper.
        valset (torch.utils.data.TensorDataset): Validation set wrapper.
        train_loader (torch.utils.data.DataLoader): Training set loader with shuffling
        val_loader (torch.utils.data.DataLoader): Validation set loader.
    """

    def __init__(self, is_gpu, args):
        self.name = 'KTH_TIPS'
        self.num_classes = 11

        self.args = args
        self.load_path = 'datasets/KTH-TIPS2-b'
        self.save_path = 'datasets/KTH_TIPS_indx'
        self.metrics_path = 'metrics/datasets/KTH_TIPS'

        # there are also scales of image from 2 to 10
        # because in KTH-TIPS2 the scale closest to the camera corresponds to Scale #2 of KTH-TIPS.
        # rotation: frontal (1,2,3,10), 22.5 right (4,5,6,11), left (7,8,9,12)
        self.img_indx2rotation = {1:0,2:0,3:0,
                                  4:1,5:1,6:1,
                                  7:2,8:2,9:2,
                                  10:0,11:1,12:2}
        self.rotation_marker = ['o', '>', '<']
        self.rotation_type = ['frontal', 'right', 'left']

        # illumination: frontal (1,4,7), 45 from top (2,5,8), 45 from side (3,6,9), ambient (10,11,12)
        self.img_indx2illumination = {1:0,2:1,3:2,
                                      4:0,5:1,6:2,
                                      7:0,8:1,9:2,
                                      10:3,11:3,12:3}
        self.illumination_marker = ['o', '^', '<', 'D']
        self.illumination_type = ['frontal', 'top', 'side', 'ambient']

        # the indices have to be sorted alphabetically, if saving as pytorch's ImageFolder
        self.class_to_idx = {'aluminium_foil': 0,
                             'brown_bread': 1,
                             'corduroy': 2,
                             'cork': 3,
                             'cotton': 4,
                             'cracker': 5,
                             'lettuce_leaf': 6,
                             'linen': 7,
                             'white_bread': 8,
                             'wood': 9,
                             'wool': 10}

        self.idx_to_class = {0: 'aluminium_foil',
                             1: 'brown_bread',
                             2: 'corduroy',
                             3: 'cork',
                             4: 'cotton',
                             5: 'cracker',
                             6: 'lettuce_leaf',
                             7: 'linen',
                             8: 'white_bread',
                             9: 'wood',
                             10: 'wool'}

        self.trainset, self.valset = self.get_dataset()
        self.train_loader, self.val_loader = self.get_dataset_loader(args.batch_size,
                                                                     args.workers, is_gpu)

    def get_dataset(self):
        """
        Downloads dataset from website if doesn't exist already.
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        if not os.path.exists(self.load_path):
            print("Downloading KTH-TIPS2b dataset")
            link = "https://www.csc.kth.se/cvap/databases/kth-tips/kth-tips2-b_col_200x200.tar"

            if sys.version_info < (3, 7, 0):
                # from https://gist.github.com/devhero/8ae2229d9ea1a59003ced4587c9cb236
                ftpstream = urllib.request.urlopen(link)
                thetarfile = tarfile.open(fileobj=ftpstream, mode="r|*")

            else:
                os.system("wget --cipher 'DEFAULT:!DH' " + link + " --directory-prefix=./datasets")
                thetarfile = tarfile.open("./datasets/kth-tips2-b_col_200x200.tar", mode="r|*")

            thetarfile.extractall('./datasets')
            thetarfile.close()
            print("Extracted files")

        print("Assigning train/test and computing indices")
        check = 0
        directories = [os.path.join(self.save_path, "train"), os.path.join(self.save_path, "test")]
        for directory in directories:
            if os.path.exists(directory):
                check += 1
            if not os.path.exists(directory):
                os.makedirs(directory)

        if check != len(directories):
            indices_data = {}
            # create folders to save data
            indx = 0

            # get subfolders which are at the same time the texture classes
            class_names = [f.name for f in os.scandir(self.load_path) if f.is_dir()]

            # for each texture there are 4 samples with different scaling, lightning and rotation
            for class_name in class_names:
                class_path = os.path.join(self.load_path, class_name)
                sample_paths = [f.path for f in os.scandir(class_path) if f.is_dir()]

                for i, sample_path in enumerate(sample_paths):
                    # use first found folder for testing and other 3 for training
                    # similar to https://www.robots.ox.ac.uk/~vgg/publications/2015/Cimpoi15/cimpoi15.pdf
                    if sample_path.split('/')[-1] == 'sample_c':
                        data_mode = 'test'
                    else:
                        data_mode = 'train'

                    for filepath in glob.glob(os.path.join(sample_path, '*.png')):
                        filename = os.path.basename(filepath)

                        scale = int(filename.split('_')[1])
                        img_indx = int(filename.split('_')[3])

                        img = Image.open(filepath)

                        label = self.class_to_idx[class_name]
                        img_name = filepath.split('/')[-2] + '_' + filepath.split('/')[-1]

                        # formatting is needed, because ImageFolder sorts e.g. 0,1,10,2,20,3
                        # instead of 0,1,2,3
                        directory = os.path.join(os.path.join(self.save_path, data_mode),
                                                 format(label, '06d'))

                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        img_path = os.path.join(directory, img_name)
                        img.save(img_path, 'png')

                        # additional dataset metrics
                        indices_data[img_path] = (label,
                                                  self.img_indx2rotation[int(img_indx)],
                                                  self.img_indx2illumination[int(img_indx)],
                                                  int(scale))
                        indx += 1

            # store img_paths which serves as indices and the labels for further analysis
            indices_data = collections.OrderedDict(sorted(indices_data.items()))

            dataframe = pd.DataFrame({'img_paths': list(indices_data.keys()),
                                      'labels': np.array(list(indices_data.values()))[:, 0],
                                      'rotation': np.array(list(indices_data.values()))[:, 1],
                                      'illumination': np.array(list(indices_data.values()))[:, 2],
                                      'scale': np.array(list(indices_data.values()))[:, 3]})
            DatasetMetrics.indices_paths(self.name, dataframe)

        # load the image dataset from folder with indices
        train_transform = transforms.Compose([
            # you can add other transformations in this list
            # resize (256 x remaining larger size) and RandomCrop(224)
            # like in https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
            # https://arxiv.org/pdf/1409.1556.pdf
            transforms.Resize(200),  # resize smaller size to 200
            transforms.RandomCrop(self.args.patch_size),  # 190
            transforms.ToTensor()
        ])
        test_transform = transforms.Compose([
            # you can add other transformations in this list
            # resize (256 x remaining larger size) and RandomCrop(224)
            # like in https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
            # https://arxiv.org/pdf/1409.1556.pdf
            transforms.Resize(200),  # resize smaller size to 200
            transforms.CenterCrop((self.args.patch_size, self.args.patch_size)),  # 190
            transforms.ToTensor()
        ])
        if self.args.compute_dataset_metrics is True:
            # when computing dataset metrics, an original image should be used
            # - without randomness of RandomCrop
            train_transform = transforms.Compose([
                transforms.ToTensor()
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor()
            ])

            # if not already set, set batch-size to 1 for computing the metrics
            # due to different image sizes
            self.args.batch_size = 1

        trainset = IndxImageFolder(root=os.path.join(self.save_path, "train"), transform=train_transform)
        valset = IndxImageFolder(root=os.path.join(self.save_path, "test"), transform=test_transform)

        return trainset, valset

    def get_dataset_loader(self, batch_size, workers, is_gpu):
        """
        Defines the dataset loader for wrapped dataset
        Parameters:
            batch_size (int): Defines the batch size in data loader
            workers (int): Number of parallel threads to be used by data loader
            is_gpu (bool): True if CUDA is enabled so pin_memory is set to True
        Returns:
             torch.utils.data.TensorDataset: trainset, valset
        """

        train_loader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=is_gpu, sampler=None)

        val_loader = torch.utils.data.DataLoader(
            self.valset,
            batch_size=batch_size, shuffle=False,
            num_workers=workers, pin_memory=is_gpu)

        return train_loader, val_loader