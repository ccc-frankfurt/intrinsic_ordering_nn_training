import numpy as np


def labels2one_hot(label_list, num_classes):
    """ transforms the list of labels from numbers to one-hot for the multilabel case """
    labels_one_hot = {}
    for index, target_list in label_list.items():
        if isinstance(target_list, str):
            target_list = list(map(int, target_list.strip('[]').split(',')))
        labels_one_hot[index] = [1 if i in target_list else 0 for i in range(num_classes)]

    return labels_one_hot

def corrupt_labels(targets, num_classes, corrupt_prob):
    """
    Randomize a certain percentage of labels with corrupt_prob
    Adapted from https://github.com/pluskid/fitting-random-labels/blob/master/cifar10_data.py
    targets: dictionary of paths and labels
    """
    labels = np.array(list(targets.values()))
    np.random.seed(12345)
    mask = np.random.rand(len(labels)) <= corrupt_prob
    rnd_labels = np.random.choice(num_classes, mask.sum())
    labels[mask] = rnd_labels
    # we need to explicitly cast the labels from npy.int64 to
    # builtin int type, otherwise pytorch will fail...
    labels = dict(zip(list(targets.keys()), [int(x) for x in labels]))
    return labels