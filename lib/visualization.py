import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

import torchvision

class Visualizer:

    def __init__(self):
        self.fontsize = 28 * 1.25
        self.titlesize = self.fontsize + 10
        self.legendsize = self.fontsize - 8
        self.ticksize = self.fontsize

    def set_plt_sizes(self):
        plt.rc('font', size=self.fontsize)  # controls default text sizes
        plt.rc('axes', titlesize=self.ticksize)  # fontsize of the axes title
        plt.rc('axes', labelsize=self.ticksize+2)  # fontsize of the x and y labels
        plt.rc('xtick', labelsize=self.ticksize)  # fontsize of the tick labels
        plt.rc('ytick', labelsize=self.ticksize)  # fontsize of the tick labels
        plt.rc('legend', fontsize=self.legendsize)  # legend fontsize
        plt.rc('figure', titlesize=self.titlesize)  # fontsize of the figure title

    def check_images(self, dataset):

        train_dataloader = dataset.train_loader
        # if needed, check also test data
        # test_dataloader = dataset.val_loader

        images, labels, _ = next(iter(train_dataloader))
        # print(images.shape)
        # print(labels[0:10])

        imgs_numpy = images.numpy()[0:10].transpose(0, 2, 3, 1).squeeze()
        labels_numpy = labels.numpy()

        fig = plt.figure(figsize=(8, 8))
        columns = 4
        rows = 5
        for i in range(0, len(imgs_numpy)):  # columns*rows
            img = imgs_numpy[i]
            ax = fig.add_subplot(rows, columns, i + 1)

            # if it is the multi-label scenario, labels are one-hot
            if len(labels_numpy.shape) > 1:
                ax.set_title(dataset.idx_to_class[labels_numpy[i].argmax()])
            else:
                ax.set_title(dataset.idx_to_class[labels_numpy[i]])
            plt.imshow(img)

        script_dir = Path(__file__).parent.parent
        directory = os.path.join(str(script_dir), os.path.join('results', dataset.name))
        if not os.path.exists(directory):
            os.makedirs(directory)
        fig.savefig(os.path.join(directory, 'dataset_img_samples.pdf'))
        fig.savefig(os.path.join(directory, 'dataset_img_samples.png'))
        plt.close(fig)

    def check_images_torchvision(self, dataset):

        def imshow(img):
            # img = img / 2 + 0.5     # unnormalize
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.show()

        test_dataloader = dataset.val_loader

        classes = dataset.idx_to_class
        dataiter = iter(test_dataloader)
        images, labels, _ = dataiter.next()

        # print images
        imshow(torchvision.utils.make_grid(images[0:4]))
        print('GroundTruth: ', ' '.join('%5s' % classes[labels.numpy()[j]] for j in range(4)))

    def check_dataset_class_balance(self, dataset):
        """ This method is for the single-label case """

        train_dataloader = dataset.train_loader
        test_dataloader = dataset.val_loader

        trainset_labels = []
        testset_labels = []

        for (images, labels, indices) in train_dataloader:
            labels = labels.cpu().numpy() # pushes to cpu and converts to numpy
            trainset_labels.append(labels)

        for (images, labels, indices) in test_dataloader:
            labels = labels.cpu().numpy() # pushes to cpu and converts to numpy
            testset_labels.append(labels)

        trainset_labels = np.concatenate(trainset_labels)
        testset_labels = np.concatenate(testset_labels)

        unique_train_labels, train_counts = np.unique(trainset_labels, return_counts=True)
        unique_test_labels, test_counts = np.unique(testset_labels, return_counts=True)

        # print(train_counts)
        # print(test_counts)

        # check label distribution
        plt.figure(figsize=(8, 8))
        plt.bar([dataset.idx_to_class[i] for i in unique_train_labels], train_counts, label='train')

        plt.bar([dataset.idx_to_class[i] for i in unique_test_labels], test_counts, label='test')
        plt.legend()
        plt.tight_layout()
        script_dir = Path(__file__).parent.parent
        directory = os.path.join(str(script_dir), os.path.join('results', dataset.name))
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, 'dataset_class_balance.pdf'))
        plt.close()

    def visualize_instance_agreement(self, intersections_per_epoch, lower_bound=None, accuracy=None,
                                     std_accuracy=None,
                                     title='Dataset Network TrainTest', save_path='.', pdf=None):
        num_epochs = len(intersections_per_epoch)

        sns.set("paper", font_scale=2.5)
        sns.set_style("whitegrid")
        self.set_plt_sizes()

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(intersections_per_epoch[:num_epochs], label='Agreement (correct)',
                linewidth=5)
        if accuracy is not None:
            ax.plot(accuracy[:num_epochs], linewidth=3.,
                        c='red', label='Accuracy')
            ax.fill_between([i for i in range(num_epochs)],
                            accuracy[:num_epochs] - std_accuracy,
                            accuracy[:num_epochs] + std_accuracy,
                            alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        if lower_bound is not None:
            ax.plot(lower_bound[:num_epochs], c='green', label='Lower bound',
                    linewidth=3., linestyle='dotted')
            ax.fill_between([i for i in range(num_epochs)], intersections_per_epoch[:num_epochs],
                            lower_bound[:num_epochs], alpha=0.2, hatch='/')
        if len(intersections_per_epoch) > 10:
            ax.set_xticks(np.arange(0, num_epochs, num_epochs / 10))
        ax.set_xlabel('Epochs', weight='bold')
        ax.set_ylabel('Percent', weight='bold')
        # ax.set_title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, title + '_instance_agreement.pdf'))
        pdf.savefig(fig)
        plt.close(fig)

    def visualize_metric_histogram(self, metric, metric_name, metric_units, bins, dataset_name, save_path='./results/metrics_hist'):

        save_path_dataset = os.path.join(save_path, dataset_name)
        if not os.path.exists(save_path_dataset):
            os.makedirs(save_path_dataset)

        sns.set("paper", font_scale=2.5)
        sns.set_style("whitegrid")
        self.set_plt_sizes()
        plt.rc('axes', labelsize=self.ticksize + 8)  # fontsize of the x and y labels
        plt.rc('legend', fontsize=self.legendsize + 8)
        if metric is not None:

            fig, ax = plt.subplots(figsize=(16, 12))

            ax.hist(metric, bins=bins, label=metric_name)
            ax.set_xlabel(metric_units, weight='bold')
            ax.set_ylabel('Count', weight='bold')
            plt.legend()
            plt.tight_layout()
            fig.savefig(os.path.join(save_path_dataset, 'histogram_' + metric_name + '.pdf'))
            plt.close(fig)


    # for example, Entropy in Bits
    def visualize_metric(self, metric, metric_name, metric_units, instance_agreement_per_epoch, lower_bound=None,
                          accuracy=None, std_accuracy=None, title='Dataset Network TrainTest', save_path='.', pdf=None):
        sns.set("paper", font_scale=2.5)
        sns.set_style("whitegrid")
        self.set_plt_sizes()
        plt.rc('axes', labelsize=self.ticksize + 8)  # fontsize of the x and y labels
        plt.rc('legend', fontsize=self.legendsize + 8)

        if metric is not None:
            num_epochs = len(instance_agreement_per_epoch)
            agreement_threshold = 20.
            # visualize agreement only starting from a certain epoch
            agreement_epoch_threshold = 5

            fig, ax1 = plt.subplots(figsize=(16, 12))
            ax2 = ax1.twinx()

            ax1.plot(instance_agreement_per_epoch[:num_epochs], label='Agreement (correct)',
                     linewidth=5)
            if accuracy is not None:
                ax1.plot(accuracy[:num_epochs], c='red', label='Accuracy', linewidth=3.)
                ax1.fill_between([i for i in range(num_epochs)],
                                accuracy[:num_epochs] - std_accuracy[:num_epochs],
                                accuracy[:num_epochs] + std_accuracy[:num_epochs],
                                alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
            if lower_bound is not None:
                ax1.plot(lower_bound[:num_epochs], c='green', label='Lower bound', linewidth=3.,
                         linestyle='dotted')
                ax1.fill_between([i for i in range(num_epochs)],
                                 instance_agreement_per_epoch[:num_epochs], lower_bound[:num_epochs], alpha=0.2, hatch='/')

            if num_epochs > 50:
                ax1.set_xticks(np.arange(0, num_epochs,
                                         num_epochs / 10))
            ax1.set_xlabel('Epochs', weight='bold')
            ax1.set_ylabel('Percent', weight='bold')
            ax1.legend(loc=0)

            thresholded_metric = np.array(metric)

            #if agreement_threshold > 0:
            #    ax1.axhline(agreement_threshold, linestyle='--')

            # if not every value is below threshold
            if (np.sum(instance_agreement_per_epoch > agreement_threshold).astype('int')) > 0:
                thresholded_metric[instance_agreement_per_epoch < agreement_threshold] = None

                ax2.plot([i for i in np.arange(agreement_epoch_threshold, num_epochs, 1)],
                         thresholded_metric[agreement_epoch_threshold:num_epochs], c='orange', label=metric_name,
                         linewidth=7)
            ax2.set_ylabel(metric_units, weight='bold')
            ax2.yaxis.label.set_color('orange')
            ax2.grid(False)
            # plt.title(title)
            #h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h2, l2, loc='upper center')
            plt.tight_layout()
            fig.savefig(os.path.join(save_path, title + '_' + metric_name + '.pdf'))
            pdf.savefig(fig)
            plt.close(fig)

    def visualize_metric_distinct(self, metric, metric_name, metric_units, marker,
                                  instance_agreement_per_epoch, lower_bound=None,
                                  accuracy=None, std_accuracy = None,
                                  title='Dataset Network TrainTest',
                                  save_path='.', pdf=None):
        sns.set("paper", font_scale=2.5)
        sns.set_style("whitegrid")
        self.set_plt_sizes()
        plt.rc('axes', labelsize=self.ticksize + 8)  # fontsize of the x and y labels
        plt.rc('legend', fontsize=self.legendsize + 8)

        if metric is not None:
            num_epochs = len(instance_agreement_per_epoch)

            agreement_threshold = 20.

            fig, ax1 = plt.subplots(figsize=(16, 12))
            ax2 = ax1.twinx()

            ax1.plot(instance_agreement_per_epoch[:num_epochs], label='Agreement (correct)',
                     linewidth=5)
            if accuracy is not None:
                ax1.plot(accuracy[:num_epochs], c='red', label='Accuracy', linewidth=3.)
                ax1.fill_between([i for i in range(num_epochs)],
                                 accuracy[:num_epochs] - std_accuracy[:num_epochs],
                                 accuracy[:num_epochs] + std_accuracy[:num_epochs],
                                 alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
            if lower_bound is not None:
                ax1.plot(lower_bound[:num_epochs], c='green', label='Lower bound',
                         linewidth=3., linestyle='dotted')
                ax1.fill_between([i for i in range(num_epochs)],
                                 instance_agreement_per_epoch[:num_epochs], lower_bound[:num_epochs], alpha=0.2,
                                 hatch='/')
            #if agreement_threshold > 0:
            #    ax1.axhline(agreement_threshold, linestyle='--')

            if num_epochs > 50:
                ax1.set_xticks(np.arange(0, num_epochs,
                                         num_epochs / 10))
            ax1.set_xlabel('Epochs', weight='bold')
            ax1.set_ylabel('Percent', weight='bold')
            ax1.legend(loc=0)

            thresholded_metric = np.array(metric)
            # for each metric value a separate plot
            colors = ['black', 'orange', 'purple', 'darkgreen'] # change it if there are more than 5 curves you need to plot
            for i in range(thresholded_metric.shape[1]):
                thresholded_metric[:,i][instance_agreement_per_epoch < agreement_threshold] = None

                ax2.plot(thresholded_metric[:,i][:num_epochs], c=colors[i],
                         label=metric_name + ' ' + marker[1][i],
                         marker=marker[0][i], linewidth=5)
            ax2.set_ylabel(metric_units, weight='bold')
            ax2.yaxis.label.set_color('orange')
            ax2.grid(False)
            # plt.title(title)
            #h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h2, l2, loc='lower right')
            plt.tight_layout()
            fig.savefig(os.path.join(save_path, title + '_' + metric_name + '.pdf'))
            pdf.savefig(fig)
            plt.close(fig)

        # learned_labels_per_epoch - for each epoch the correctly recognized labels
    def visualize_label_agreement_distribution(
            self, learned_labels_per_epoch, num_labels, idx_to_label=None,
                    title='Dataset Network TrainTest', save_path='.', pdf=None):
        """ This method is for the single-label case """
        sns.set("paper", font_scale=2.5)
        sns.set_style("whitegrid")

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(learned_labels_per_epoch)):
            xs, ys = np.unique(learned_labels_per_epoch[i], return_counts=True)
            normalizer = sum(ys)
            # You can provide either a single color or an array. To demonstrate this,
            # the first bar of each set will be colored cyan.
            cs = [np.random.rand(3, )] * len(xs)
            ax.bar(xs, ys / float(normalizer), zs=i, zdir='y', color=cs, alpha=0.3)

        if idx_to_label is not None:
            ax.set_xticklabels([idx_to_label[i] for i in range(num_labels)])

        ax.set_xticks([i for i in range(num_labels)])
        ax.set_xlabel('Labels')
        # cannot invert yaxis (but in matplotlib)
        ax.set_yticks([i for i in range(0, len(learned_labels_per_epoch), max(len(learned_labels_per_epoch) // 6, 1))])
        ax.set_ylabel('Epochs')
        ax.set_zlabel('Normalized Count')
        ax.view_init(30, 60)
        plt.tight_layout()
        fig.savefig(os.path.join(save_path, title + '_label_agreement.pdf'))
        pdf.savefig(fig)
        plt.close(fig)