import numpy as np
import torch
import os
import pickle


class AccuracyEvaluator:
    """  """
    def __init__(self, num_classes=0, multilabel=False, save_prob_vector=False, indices_path=''):

        self.num_classes = num_classes
        self.multilabel = multilabel
        self.save_prob_vector = save_prob_vector

        if indices_path == '':
            self.indices_correct = []
        else:
            with open(indices_path, "rb") as fp:  # Unpickling
                self.indices_correct = pickle.load(fp)

        # indices are stored in the list above after each epoch, before that
        # they are temporarily stored here
        self.indices_correct_tmp = []

        self.reset()

    def reset(self):
        """ reset every epoch """

        # number of total seen instances
        self.total = 0
        self.correct = 0

        if self.multilabel:
            # the same for multilabel case: total is array of length num_classes
            self.correct_multilabel_per_class = np.zeros(self.num_classes)

    def save_indices(self, outputs, labels, img_indx, epoch, batch, num_batches):
        """ calculate the correct predictions for image and save indices """

        if not (self.multilabel):
            _, predicted = torch.max(outputs.data, 1)
        else:
            predicted = torch.sigmoid(outputs)

        if self.multilabel:
            # in multilabel case exact agreement, e.g. (0,0,1,0,1)=(0,0,1,0,1) is wanted
            # where is 1 in label will be 1 in predicted, if it has been predicted correctly
            # those summed give amount of labels for an image predicted correctly
            predicted_binary = (predicted.detach() > 0.5).float()

            predicted_correct = torch.sum(predicted_binary * labels, axis=1)
            predicted_all = torch.sum(predicted_binary, axis=1)

            # to check whether there were false positives
            predicted_single = (predicted_correct == predicted_all).float() * predicted_correct
            labels_single = torch.sum(labels, axis=1)

            # exact match accuracy
            self.correct += (predicted_single == labels_single).sum().item()
            self.total += labels.size(0)

            # mean average precision
            # https://stats.stackexchange.com/questions/12702/what-are-the-measure-for-accuracy-of-multilabel-data
            # due to the multiplication, there will be 1 only if the label was correct, summing over batch
            self.correct_multilabel_per_class += torch.sum(predicted_binary * labels, axis=0).cpu().numpy()
            # if correctly a class has not been predicted - sum it too
            self.correct_multilabel_per_class += torch.sum((1.-predicted_binary) * (1.-labels), axis=0).cpu().numpy()
        else:
            # single label accuracy
            self.correct += (predicted == labels).sum().item()
            self.total += labels.size(0)

        if self.multilabel:
            # since img_indx is a tuple, it is converted to a list
            indices_correct = list(zip(list(img_indx), predicted.detach().cpu().numpy().astype(float)))
        else:
            if self.save_prob_vector:
                indices_correct = list(zip(list(img_indx), outputs.detach().cpu().numpy().astype(float)))
            else:
                indices_correct = list(zip(list(img_indx), predicted.detach().cpu().numpy().astype(float)))

        # save indices of correctly identified examples for one epoch
        if batch == 0:  # for the first batch
            self.indices_correct_tmp = indices_correct
        else:
            self.indices_correct_tmp += indices_correct

        # if last batch - sort and save for the given epoch.
        # zip(*) makes out of [(1,2),(3,4),(5,6)] [(1,3,5),(2,4,6)]
        if batch == num_batches - 1:  # len(self.train_loader)
            # store the indices also for the first epoch and not only the agreement
            # this is because indices for train and test are different
            if epoch == 0:
                self.indices_correct.append(
                    list(list(zip(*sorted(self.indices_correct_tmp)))[0]))
                self.indices_correct.append(
                    list(list(zip(*sorted(self.indices_correct_tmp)))[1]))
            else:
                self.indices_correct.append(
                    list(list(zip(*sorted(self.indices_correct_tmp)))[1]))

    def accuracy_single_label(self):
        return 100. * self.correct / self.total

    def mAP(self):
        return 100. * np.sum(self.correct_multilabel_per_class)/\
               (self.total*len(self.correct_multilabel_per_class))

    @staticmethod
    def save_indices_correct(indices_path, network_indices_correct=None):
        if network_indices_correct is None or indices_path == '':
            print('No indices given to save or no path given')
        else:
            with open(indices_path, "wb") as fp:  # Pickling
                pickle.dump(network_indices_correct, fp)


class ConfusionMeter:
    """
    Maintains a confusion matrix for a given calssification problem.
    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.
    Parameters:
        k (int): number of classes in the classification problem
        normalized (boolean): Determines whether or not the confusion matrix
            is normalized or not
    Copied from https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    to avoid installation of the entire torchnet package!
    BSD 3-Clause License
    Copyright (c) 2017- Sergey Zagoruyko,
    Copyright (c) 2017- Sasank Chilamkurthy,
    All rights reserved.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """

    def __init__(self, k, normalized=False):
        super(ConfusionMeter, self).__init__()
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """
        Computes the confusion matrix of K x K size where K is no of classes
        Paramaters:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of
                integer values between 0 and K-1.
            target (tensor): Can be a N-tensor of integer values assumed to be integer
                values between 0 and K-1 or N x K tensor, where targets are
                assumed to be provided as one-hot vectors
        """

        predicted = predicted.cpu().numpy()
        target = target.cpu().numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bin-counting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.k ** 2)
        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf


def accuracy(output, target, topk=(1,)):
    """
    Evaluates a model's top k accuracy
    Parameters:
        output (torch.autograd.Variable): model output
        target (torch.autograd.Variable): ground-truths/labels
        topk (list): list of integers specifying top-k precisions
            to be computed
    Returns:
        float: percentage of correct predictions
    """

    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res