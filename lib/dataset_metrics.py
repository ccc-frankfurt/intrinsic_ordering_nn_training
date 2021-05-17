import os
import pandas as pd

import subprocess
import collections
import shutil
import urllib

from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from PIL import Image

from tqdm import tqdm

import torchvision.datasets as torch_datasets
from torchvision import transforms

from lib.helpers.dct import *


class DatasetMetrics:
    """ This class implements different image metrics.
        Needed information will be stored in the folder 'metrics'"""

    def __init__(self, dataset, dataset_name):
        # dataset is not a pytorch dataset, but a custom class with loaders and additonal convenience variables
        self.dataset = dataset
        self.dataset_name = dataset_name
        if dataset is not None:
            self.trainloader = dataset.train_loader
            self.testloader = dataset.val_loader

        self.metrics_path = os.path.join('metrics/datasets', dataset_name)

        if not os.path.exists(self.metrics_path):
            os.makedirs(self.metrics_path)

        try:
            # if some metrics have already been saved
            self.evaluation = pd.read_csv(self.metrics_path + '.csv')
            self.evaluation.set_index('img_paths', inplace=True)
        except FileNotFoundError:
            print("No evaluation available: indices_path() should be called in dataset")

    @staticmethod
    def indices_paths(dataset_name, dataframe):
        """ create a pandas dataframe and transfer indices2labels information there
        this dataframe can be extended column-wise with additional information
        like segment or instance count """

        metrics_path = os.path.join('metrics/datasets', dataset_name)

        if not os.path.exists(metrics_path):
            os.makedirs(metrics_path)

        if not os.path.exists(metrics_path+'.csv'):
            dataframe.set_index('img_paths', inplace=True)
            dataframe.to_csv(metrics_path + '.csv')
        else:
            print("ATTENTION: an evaluation file for this dataset already exists")

    @staticmethod
    def load_evaluation_metrics(dataset_name):
        metrics_path = os.path.join('metrics/datasets', dataset_name)
        evaluation = pd.read_csv(metrics_path + '.csv')
        evaluation.set_index('img_paths', inplace=True)
        return evaluation

    def segment_count(self):
        """ the following algorithm is used: http://cs.brown.edu/people/pfelzens/segment/ """

        if not 'segcount' in self.evaluation and self.dataset is not None:
            print('Calculating segment count')
            # https://svaderia.github.io/articles/downloading-and-unzipping-a-zipfile/
            # load the zip with the segmentation algorithm in C
            if not os.path.exists('./metrics/segment'):
                zipurl = 'http://cs.brown.edu/people/pfelzens/segment/segment.zip'
                with urlopen(zipurl) as zipresp:
                    with ZipFile(BytesIO(zipresp.read())) as zfile:
                        zfile.extractall('metrics/')

                # make the file
                os.system("make -C ./metrics/segment")

            # convert images to pnm
            # and call segment-library with standard parameters (sigma 0.5 k 500 min 20)
            # store number of segments for every img-id

            dataset_path = os.path.join('./datasets/tmp', self.dataset.name)
            if not os.path.exists(dataset_path):
                os.makedirs(dataset_path)

            # create a dictionary to store the results. Keys are img-indices, values are seg_counts
            indices2segcount = {}

            for loader in [self.trainloader, self.testloader]:
                for batch_indx, data in enumerate(tqdm(loader)):
                    # get the inputs; data is a list of [inputs, labels, indices]
                    inputs, labels, img_indx = data

                    # renormalize from range [0,1] to [0,255]
                    inputs = (inputs.cpu().numpy() * 255.).astype(np.uint8)
                    labels = labels.cpu().numpy()
                    img_indx = list(img_indx)
                    # iterate through the batch
                    for indx, label, input_img in zip(img_indx, labels, inputs):
                        # convert imgs to ppm
                        img = Image.fromarray(input_img.transpose(1, 2, 0)).convert('RGB')

                        img_name = 'tmp_img'
                        img_path = os.path.join(dataset_path, img_name)
                        img.save(img_path + '.ppm')

                        # call segment-library
                        result = subprocess.run(['./metrics/segment/segment', '0.5', '500', '20',
                                                 img_path + '.ppm', img_path + '_seg.ppm'], stdout=subprocess.PIPE)
                        output = result.stdout.decode("utf-8")
                        # only one split, take second element of split,
                        # then strip and split on space to get the number of segments
                        segment_count = output.split("got", 1)[1].strip().split(' ')[0]
                        indices2segcount[indx] = int(segment_count)

                        '''
                        test = Image.open(img_path + '_seg.ppm')
                        plt.imshow(test)
                        plt.show() '''

                    # save every 20 batches in a separate dataframe (backup)
                    if batch_indx % 20 == 0:
                        self.save_backup(indices2segcount, 'segcount')
                # after a loader has finished
                self.save_backup(indices2segcount, 'segcount')
            # after all loaders
            self.save_backup(indices2segcount, 'segcount')

            # delete temporarily created files
            shutil.rmtree('./datasets/tmp')

            # put all the data into the dataframe (which you can also save)
            self.evaluation['segcount'] = self.evaluation.index.map(lambda x: indices2segcount.get(x))

    def img_entropy(self, window_size=10):
        """ calculates image entropy over a window size and averages for an image
        https://github.com/TamojitSaha/Image_Entropy """
        if not 'entropy' in self.evaluation and self.dataset is not None:
            print("Calculating img_entropy")
            # create a dictionary to store the results.
            # Keys are img-indices, values are entropies
            indices2entropy = {}

            for loader in [self.trainloader, self.testloader]:
                for batch_indx, data in enumerate(tqdm(loader)):
                    # get the inputs; data is a list of [inputs, labels, indices]
                    inputs, labels, img_indx = data
                    # renormalize from range [0,1] to [0,255]
                    inputs = (inputs.cpu().numpy() * 255.).astype(np.uint8)
                    img_indx = list(img_indx)

                    # convert imgs to grayscale
                    # inputs has shape (batch_size, channels, h, w)
                    if inputs.shape[1] == 3:
                        # average r, g, and b channels
                        gray = 0.2989 * inputs[:, 0, :, :] + \
                               0.5870 * inputs[:, 1, :, :] + 0.1140 * inputs[:, 2, :, :]
                    elif inputs.shape[1] == 1:  # image is already grayscale
                        gray = inputs.squeeze(dim=1)
                    else:
                        print('Error: unknown image dimension, neither grayscale nor rgb')

                    # compute the entropy batch-wise for a window of certain size
                    for i in range(len(gray)):
                        window_entropies = []
                        for h in range(gray.shape[1] + 1 - window_size):
                            for w in range(gray.shape[2] + 1 - window_size):
                                # get window-elements for every img of the batch
                                window = gray[i, h:h + window_size, w:w + window_size]
                                values, counts = np.unique(window, return_counts=True)
                                probs = counts / float((np.sum(counts)))
                                window_entropies.append(-np.sum(probs * np.log2(probs)))
                        # average window entropies for an image
                        indices2entropy[img_indx[i]] = np.mean(window_entropies)

                    # save every 20 batches in a separate dataframe (backup)
                    if batch_indx % 20 == 0:
                        self.save_backup(indices2entropy, 'entropy')
                # after a loader has finished
                self.save_backup(indices2entropy, 'entropy')
            # after all loaders
            self.save_backup(indices2entropy, 'entropy')

            # put all the data into the dataframe
            self.evaluation['entropy'] = self.evaluation.index.map(lambda x: indices2entropy.get(x))

    def edge_strength(self):
        """ For edge strength, first execute find_all_boundaries.m first, which needs
        https://github.com/phillipi/crisp-boundaries
        to generate edge_strength.csv in the metrics/dataset folder.
        This method just takes evaluation results and incorporates them into self.evaluation """
        path = os.path.join(os.path.join(
                'metrics/datasets', self.dataset.name), 'edge_strengths.csv')
        if not 'edge_strength' in self.evaluation and os.path.exists(path):
            print('Storing edge strength')
            # edge strength based on crisp boundary detection
            edge_strength = pd.read_csv(path)
            # convert label_index.jpg to index.jpg to int(index)
            indices = [str(path).split(self.dataset.name)[-1] for path in edge_strength['Image'].to_numpy()]
            values = edge_strength['EdgeStrength'].to_numpy()

            dictionary = dict(zip(indices, values))
            self.evaluation['edge_strength'] = self.evaluation.index.map(lambda x: dictionary.get(x.split(self.dataset.name)[-1]))

    def img_frequency(self, device):
        """ based on https://github.com/LTS4/hold-me-tight
            requires pip install torch-dct """

        if (not 'freq_biggest_coeff' in self.evaluation or \
                self.evaluation['freq_biggest_coeff'].isnull().values.any()) \
                and self.dataset is not None:
            print("Calculating img_frequency")
            # create a dictionary to store the results.
            # Keys are img-indices, values are entropies
            self.evaluation['freq_biggest_coeff'] = None
            self.evaluation['freq_coeff_percentage'] = None
            indices2freq = {}

            for loader in [self.trainloader, self.testloader]:
                for batch_indx, data in enumerate(tqdm(loader)):
                    # get the inputs; data is a list of [inputs, labels, indices]
                    inputs, labels, img_indx = data
                    img_indx = list(img_indx)

                    # convert imgs to grayscale
                    # inputs has shape (batch_size, channels, h, w)
                    if inputs.shape[1] == 3:
                        # average r, g, and b channels
                        gray = 0.2989 * inputs[:, 0, :, :] + \
                               0.5870 * inputs[:, 1, :, :] + 0.1140 * inputs[:, 2, :, :]
                    elif inputs.shape[1] == 1:  # image is already grayscale
                        gray = inputs.squeeze(dim=1)
                    else:
                        print('Unclear image shape - neither rgb nor gray')

                    # if batch-size bigger 1
                    if inputs.shape[0] > 1:
                        batch_dct_2d = lambda x: apply_linear_2d(x, LinearDCT(x.size(1), type='dct', norm='ortho')).data
                        batch_coeffs = batch_dct_2d(gray.to(device)).cpu()

                    for i, img in enumerate(gray):

                        # if dct coefficients have not been computed yet
                        if inputs.shape[0] == 1:
                            coeffs = dct_2d(img.to(device), norm='ortho')
                        else:
                            # the coefficients have already been computed in the batch before
                            coeffs = batch_coeffs[i]

                        # https://de.mathworks.com/help/signal/ref/dct.html
                        # Find what percentage of DCT coefficients contain 99.98% of the energy in the image.

                        # if needed, one can get indices of sorted elements
                        sorted_coeffs, _ = torch.sort(torch.abs(coeffs).flatten(), descending=True)
                        num_coeffs = 1
                        while torch.norm(sorted_coeffs[0:num_coeffs]) / torch.norm(sorted_coeffs) < 0.9998:
                            num_coeffs = num_coeffs + 1

                        # num_coeffs - 1, because else for 1 coeffs it would be 2/size, instead of 1 due to while-loop
                        coeff_percentage = (float(num_coeffs - 1) / np.prod(list(coeffs.size())) * 100.)
                        #print('%f of the coefficients are sufficient\n' % coeff_percentage)
    
                        '''inverse = idct_2d(coeffs, norm='ortho')
                        import matplotlib.pyplot as plt
                        f, axarr = plt.subplots(1, 3)
                        axarr[0].imshow(img, cmap='gray')
                        axarr[1].imshow(coeffs)
                        axarr[1].set_xlabel('freq coeffs left upper [low-low]')
                        axarr[2].imshow(inverse, cmap='gray')
                        plt.show()'''
                        self.evaluation.at[img_indx[i], 'freq_biggest_coeff'] = sorted_coeffs[0].item()
                        self.evaluation.at[img_indx[i], 'freq_coeff_percentage'] = coeff_percentage
                        indices2freq[img_indx[i]] = coeff_percentage

                    # save every 20 batches in a separate dataframe (backup)
                    if batch_indx % 20 == 0:
                        self.save_backup(indices2freq, 'freq_coeff_percentage')
                # after a loader has finished
                self.save_backup(indices2freq, 'freq_coeff_percentage')
            # after all loaders
            self.save_backup(indices2freq, 'freq_coeff_percentage')

    def human_uncertainty_CIFAR10(self):
        """ Based on https://github.com/jcpeterson/cifar-10h
        Download https://github.com/jcpeterson/cifar-10h/blob/master/data/cifar10h-probs.npy
        into metrics/datasets/CIFAR10 first """

        if not 'human_uncertainty' in self.evaluation:
            print("Evaluating human uncertainty on CIFAR10")
            uncertainties_path = 'metrics/datasets/CIFAR10/cifar10h-probs.npy'
            if not os.path.exists(uncertainties_path):
                link = "https://github.com/jcpeterson/cifar-10h/blob/master/data/cifar10h-probs.npy?raw=true"
                f = urllib.request.urlopen(link)
                myfile = f.read()
                writeFileObj = open(uncertainties_path, 'wb')
                writeFileObj.write(myfile)
                writeFileObj.close()

            # The order of the 10,000 labels matches the original CIFAR-10
            # test set order
            uncertainties = np.load(uncertainties_path)

            # since there is 1 eval for train and test, but only test values exist for CIFAR10 human uncertainty
            self.evaluation['human_uncertainty'] = 0.

            # human uncertainty has been calculated only for the testset
            for loader in [self.testloader]:
                for batch_indx, data in enumerate(tqdm(loader)):
                    # get the inputs; data is a list of [inputs, labels, indices]
                    inputs, labels, img_indx = data
                    img_indx = list(img_indx)

                    eps = 1e-11  # since log2 of 0 is undefined
                    # compute the predictive entropy batch-wise
                    # due to eps check that uncertainty is not bigger than 1
                    start_indx = batch_indx*len(inputs)
                    pred_entropy = - np.sum(uncertainties[start_indx:start_indx+len(inputs)]*np.log2(
                        np.minimum(uncertainties[start_indx:start_indx+len(inputs)]+eps, 1.)), axis=1)

                    # save in the dictionary with indices
                    for i, entropy in enumerate(pred_entropy):
                        self.evaluation.at[img_indx[i], 'human_uncertainty'] = entropy

    def additional_metrics_VOCDetection(self):
        """ Img_difficulty is based on Ionescu et al (2016). How hard can it be?
        Estimating the difficulty of visual search in an image.
        Computer Vision and Pattern Recognition (CVPR), 2157–2166.
        https://doi.org/10.1109/CVPR.2016.237

        We removed all the response times longer than 20 seconds, and then,
        we normalized each annotator’s responsetimes by subtracting the annotator’s mean time
        and by di-viding the resulted times by the standard deviation.
        We re-moved all the annotators with less than 3 annotations since their mean time is
        not representative.  We also excluded all the annotators with less than 10 annotations
        with an aver-age response time higher than 10 seconds.
        After removing all the outliers, the difficulty score per image is computed as the
        geometric mean of the remaining times. https://arxiv.org/pdf/1705.08280.pdf

        Object size is computed as the percentage of objects with respect to background in the img.
        """

        # load the image difficulty scores from  https://image-difficulty.herokuapp.com
        if not os.path.exists('./metrics/datasets/VOCDetection/Visual_Search_Difficulty_v1.0'):
            zipurl = 'https://image-difficulty.herokuapp.com/Visual_Search_Difficulty_v1.0.zip'
            with urlopen(zipurl) as zipresp:
                with ZipFile(BytesIO(zipresp.read())) as zfile:
                    zfile.extractall('metrics/datasets/VOCDetection')

        # load csv
        difficulty = pd.read_csv('./metrics/datasets/VOCDetection/Visual_Search_Difficulty_v1.0/VSD_dataset.csv',
                                 names=['img_name', 'difficulty'])

        indices2difficulty = {}

        # since the data are not shuffled, the index is the same as when creating the dataset
        for loader_name, loader in [('train', self.trainloader)]:
            for (img, annotation, indx) in tqdm(loader):
                # get img difficulty score
                # since batch-size is 1
                img_name = indx[0].split(os.path.sep)[-1].split("__")[-1].split('.')[0]
                if img_name in difficulty.img_name.tolist():
                    indices2difficulty[indx[0]] = difficulty[difficulty.img_name == img_name].difficulty.to_numpy()[0]

        # put all the data into the dataframe
        # first, sort the dictionary according to the indices
        self.evaluation['img_difficulty'] = self.evaluation.index.map(lambda x: indices2difficulty.get(x))

    def save(self):
        self.evaluation.to_csv(self.metrics_path + '.csv')

    def save_backup(self, metric_dict, metric_name):
        backup = pd.DataFrame({'img_paths': list(metric_dict.keys()),
                                metric_name: list(metric_dict.values())})
        backup.to_csv(os.path.join(os.path.join('metrics/datasets/',
                                      self.dataset_name), metric_name + '.csv'))