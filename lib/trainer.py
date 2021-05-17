import collections

import torch
import torch.utils.data
from lib.helpers.utils import GPUMem
from lib.architectures import calc_gpu_mem_req

import os
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

from lib.helpers.accuracy_metrics import AccuracyEvaluator
from lib.helpers.accuracy_metrics import ConfusionMeter


class Trainer:
    def __init__(self, dataset, net, optimizer, criterion, net_name, scheduler_type=None, save_path='.',
                 resume=False, metrics=None, args=None):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("Calculating on " + str(self.device))
        self.save_path = save_path
        self.resume = resume

        self.dataset = dataset
        self.multilabel = False if not(args.multilabel) else True

        self.split_batch_size = self.check_if_model_fits_memory(self.device, args.batch_size, net, dataset)

        self.train_loader, self.test_loader = dataset.get_dataset_loader(self.split_batch_size,
                                                                         args.workers,
                                                                         torch.cuda.is_available())
        self.net = net.to(self.device)
        self.net_name = net_name
        self.optimizer = optimizer
        self.criterion = criterion

        self.learning_rate = []
        self.train_losses = []
        self.test_losses = []
        self.train_accuracy = []
        self.test_accuracy = []

        if not self.resume:

            self.scheduler = None
            if scheduler_type == 'OneCycleLR':
                # initialize the cyclic scheduler
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.learning_rate,
                                                                     steps_per_epoch=len(self.train_loader),
                                                                     epochs=args.epochs)
            elif scheduler_type == 'CosineAnnealingLR':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                                       eta_min=args.eta_min)
            elif scheduler_type == 'StepLR':
                self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)

            '''elif scheduler_type == 'ReduceOnePlateau':
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')'''

            self.start_epoch = 0
        else:
            checkpoint = torch.load(os.path.join(self.save_path, 'model_parameters.tar'))
            self.net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler = checkpoint['scheduler']
            self.start_epoch = checkpoint['epoch']
            self.net_name = checkpoint['net_name']
            self.train_losses = checkpoint['train_losses']
            self.test_losses = checkpoint['test_losses']
            self.train_accuracy = checkpoint['train_accuracy']
            self.test_accuracy = checkpoint['test_accuracy']
            self.learning_rate = checkpoint['learning_rate']

        # epochwise: (img_indices, whether prediction was correct (1, else 0))
        self.indices_correct_train = []
        self.indices_correct_test = []

    def check_if_model_fits_memory(self, device, batch_size, model, dataset):
        """ Adapted from https://github.com/MrtnMndt/meta-learning-CODEBRIM """

        print('*' * 80)
        print('Check if model fits memory')
        # gets number of available gpus and total gpu memory
        num_gpu = float(torch.cuda.device_count())
        gpu_mem = GPUMem(torch.device('cuda') == device, device_id=self.args.device_id)

        # gets available gpu memory
        if torch.device('cuda') == device:
            gpu_avail = (gpu_mem.total_mem - gpu_mem.total_mem * gpu_mem.get_mem_util()) / 1024.
            print('gpu memory available:{gpu_avail:.4f}'.format(gpu_avail=gpu_avail))

        # prints estimated gpu requirement of model but actual memory requirement is higher than what's estimated (from
        # experiments)
        model_gpu_mem_req = calc_gpu_mem_req(model, batch_size, self.args.patch_size, self.args.num_colors)
        print("model's estimated gpu memory requirement: {gpu_mem_req:.4f} GB".format(gpu_mem_req=model_gpu_mem_req))

        # scaling factor and buffer for matching expected memory requirement with empirically observed memory requirement
        scale_factor = 4.0
        scale_buffer = 1.0
        if torch.device('cuda') == device:
            scaled_gpu_mem_req = (scale_factor / num_gpu) * model_gpu_mem_req + scale_buffer
            print("model's empirically scaled gpu memory requirement: {scaled_gpu_mem_req:.4f}".format(scaled_gpu_mem_req=
                                                                                                   scaled_gpu_mem_req))
        split_batch_size = batch_size
        # splits batch into smaller batches
        if (torch.device('cuda') == device) and gpu_avail < scaled_gpu_mem_req:
            # estimates split batch size as per available gpu mem. (may not be a factor of original batch size)
            approx_split_batch_size = int(((gpu_avail - scale_buffer) * num_gpu / scale_factor) //
                                          (model_gpu_mem_req / float(batch_size)))

            diff = float('inf')
            temp_split_batch_size = 1
            # sets split batch size such that it's close to the estimated split batch size, is also a factor of original
            # batch size & should give a terminal batch size of more than 1
            for j in range(2, approx_split_batch_size + 1):
                if batch_size % j == 0 and abs(j - approx_split_batch_size) < diff and (len(dataset.trainset) % j > 1):
                    diff = abs(j - approx_split_batch_size)
                    temp_split_batch_size = j
            split_batch_size = temp_split_batch_size

        print('split batch size:{}'.format(split_batch_size))
        print('*' * 80)
        return split_batch_size

    def train(self, num_epochs):

        factor = self.args.batch_size // self.split_batch_size
        last_batch = int(math.ceil(len(self.train_loader.dataset) / float(self.split_batch_size)))


        indices_path_train = os.path.join(self.save_path,
                                              self.net_name + ' ' +
                                              'network_indices_correct_train.pkl')
        indices_path_test = os.path.join(self.save_path,
                                             self.net_name + ' ' +
                                             'network_indices_correct_test.pkl')
        if self.resume:
            acc_evaluator_train = AccuracyEvaluator(self.dataset.num_classes,
                                                    self.args.multilabel,
                                                    indices_path=indices_path_train)
            acc_evaluator_test = AccuracyEvaluator(self.dataset.num_classes,
                                                   self.args.multilabel,
                                                   indices_path=indices_path_test)
        else:
            acc_evaluator_train = AccuracyEvaluator(self.dataset.num_classes,
                                                    self.args.multilabel)
            acc_evaluator_test = AccuracyEvaluator(self.dataset.num_classes,
                                                   self.args.multilabel)

        for epoch in range(self.start_epoch, num_epochs, 1):  # loop over the dataset multiple times

            # after each epoch, self.test is called with eval
            self.net.train()
            self.optimizer.zero_grad()

            epoch_loss = 0.0
            running_loss = 0.0

            acc_evaluator_train.reset()

            for i, data in enumerate(self.train_loader, 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, img_indx = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # forward + backward + optimize
                outputs = self.net(inputs)

                # the multiplicative factor is the fraction of smaller batch-size to the wished one
                # for the case that the batch-size has been split
                loss = self.criterion(outputs, labels) * inputs.size(0) / float(self.args.batch_size)
                loss.backward()

                acc_evaluator_train.save_indices(outputs, labels, img_indx,
                                                 epoch, i, len(self.train_loader))

                # update the weights after every 'factor' times 'batch count'
                # for the case that batch-size has been split
                if (i + 1) % factor == 0 or i == (last_batch - 1):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # print statistics
                running_loss += loss.item()
                epoch_loss += loss.item()
                if i % int(len(self.train_loader)/10) == int(len(self.train_loader)/10) -1:  # print every x mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / int(len(self.train_loader)/10)))
                    running_loss = 0.0

                # a step every batch
                if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()

            self.train_losses.append(epoch_loss / (i + 1))
            self.train_accuracy.append(acc_evaluator_train.accuracy_single_label())

            print('Train accuracy: %d %%' % (
                    acc_evaluator_train.accuracy_single_label()))
            if self.args.multilabel:
                print('mAP : %d %%' % (
                    acc_evaluator_train.mAP()))
            print('Train loss: ', epoch_loss / (i + 1))

            if epoch % 100 == 99:
                self.print_losses()

            self.test(epoch, acc_evaluator_test)

            self.print_losses()
            self.print_accuracy()

            # after an epoch of training
            if self.scheduler is not None:
                lr = self.scheduler.get_last_lr()
                print("learning rate")
                print(lr)
                self.learning_rate.append(lr)
                self.print_lr()

                if isinstance(self.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)\
                        or isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR)\
                        or isinstance(self.scheduler, torch.optim.lr_scheduler.MultiStepLR):
                    self.scheduler.step()

            # save network and parameters to resume training later
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': self.net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler': self.scheduler,
                'train_losses': self.train_losses,
                'test_losses': self.test_losses,
                'train_accuracy': self.train_accuracy,
                'test_accuracy': self.test_accuracy,
                'learning_rate': self.learning_rate,
                'net_name': self.net_name
            }, os.path.join(self.save_path, 'model_parameters.tar'))

            # save correct_indices for train and test for the given epoch
            self.indices_correct_train = acc_evaluator_train.indices_correct
            self.indices_correct_test = acc_evaluator_test.indices_correct
            AccuracyEvaluator.save_indices_correct(indices_path_train,
                network_indices_correct=self.indices_correct_train)

            AccuracyEvaluator.save_indices_correct(indices_path_test,
                network_indices_correct=self.indices_correct_test)
        print('Finished Training')

    def test(self, epoch, acc_evaluator):
        epoch_loss = 0

        acc_evaluator.reset()
        if not self.args.multilabel:
            confusion = ConfusionMeter(self.dataset.num_classes, normalized=True)

        self.net.eval()
        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                images, labels, img_indx = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                loss = self.criterion(outputs, labels)

                acc_evaluator.save_indices(outputs, labels, img_indx,
                                           epoch, i, len(self.test_loader))
                if not self.args.multilabel:
                    confusion.add(outputs.data, labels.detach().cpu())
                epoch_loss += loss.item()

        self.test_losses.append(epoch_loss / (i + 1))
        self.test_accuracy.append(acc_evaluator.accuracy_single_label())

        print('Accuracy of the network on the test images: %d %%' % (
                acc_evaluator.accuracy_single_label()))
        if not self.args.multilabel:
            print('Confusion matrix')
            print(confusion.value())
        if self.args.multilabel:
            print('mAP : %d %%' % (
                acc_evaluator.mAP()))

    def print_losses(self):
        plt.figure(figsize=(15, 8))
        plt.plot(self.train_losses, label='train')
        plt.plot(self.test_losses, label='test')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend()

        loss_path = os.path.join(self.save_path, self.net_name + '_losses.png')
        plt.savefig(loss_path)
        plt.close()

    def print_accuracy(self):
        plt.figure(figsize=(15, 8))
        plt.plot(self.train_accuracy, label='train')
        plt.plot(self.test_accuracy, label='test')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend()

        loss_path = os.path.join(self.save_path, self.net_name + '_accuracies.png')
        plt.savefig(loss_path)
        plt.close()

    def print_lr(self):
        """ Print the learning rate changes. Important if you use a scheduler with a changing lr """
        plt.figure(figsize=(8, 8))
        plt.plot(self.learning_rate, label='train')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Learning rate', fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend()

        loss_path = os.path.join(self.save_path, self.net_name + '_learning_rate.png')
        plt.savefig(loss_path)
        plt.close()

    def check_accuracy(self):
        """ works not for multilabel case"""
        classes = self.dataset.idx_to_class
        class_correct = list(0. for i in range(len(classes)))
        class_total = list(0. for i in range(len(classes)))
        with torch.no_grad():
            for data in self.test_loader:
                images, labels, _ = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.net(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(len(classes)):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))