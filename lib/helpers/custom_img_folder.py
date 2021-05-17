import os

import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader


class IndxImageFolder(ImageFolder):

    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, num_classes=None,
                 multilabel=False, random_labels=None):
        super().__init__(root, transform, target_transform, loader)
        # for one-hot encoding of multilabel
        self.num_classes = num_classes
        self.multilabel = multilabel
        self.random_labels = random_labels

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, index) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        # if self.num_classes is not None - we have a multi-label scenario
        if self.multilabel:
            # list with labels as integers
            target_list = list(map(int, path.split(os.path.sep)[-1].split("__")[0].split("_")))
            # converting one-hot-encoding
            target = torch.Tensor([1 if i in target_list else 0 for i in range(self.num_classes)])

        if self.random_labels is not None:
            target = self.random_labels[path]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path
