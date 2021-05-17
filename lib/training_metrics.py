import numpy as np


class TrainingMetrics:

    def calc_metric(self, agreement_indices_per_epoch, metric_evaluation, metric_name):
        """
        per epoch, a mean value for the epoch of a metric, e.g. entropy will be returned
        for agreement indices
        """

        if metric_name in metric_evaluation.columns:
            metric = []
            metric_column = metric_evaluation[metric_name]
            for epoch in range(len(agreement_indices_per_epoch)):
                metric_values = metric_column[agreement_indices_per_epoch[epoch]].to_numpy()
                if len(metric_values) > 0:
                    metric.append(np.nanmean(metric_values))
                else:
                    metric.append(0)

        else:
            metric = None

        return metric

    def calc_metric_distinct(self, learned_instance_network_lists, agreement_indices_per_epoch, metric_evaluation, metric_name):
        """
        per epoch, a mean value for the epoch of a metric, e.g. entropy will be returned
        for agreement indices
        the normalization is the percentage of agreed metric value by all metric values of
        that type in train/test
        """

        if metric_name in metric_evaluation.columns:
            indices = np.array(learned_instance_network_lists[0][0])

            metric_column = metric_evaluation[metric_name]
            unique_values, all_counts = np.unique(metric_column[indices].to_numpy(),
                                                  return_counts=True)

            metric = np.zeros((len(agreement_indices_per_epoch), len(unique_values)))
            for epoch in range(len(agreement_indices_per_epoch)):
                metric_values = metric_column[agreement_indices_per_epoch[epoch]].to_numpy()
                sorted_values, counts = np.unique(metric_values, return_counts=True)
                # counts for each value stored separately
                for i, val in enumerate(sorted_values):
                    val_index = list(unique_values).index(val)
                    metric[epoch][val_index] = 100.*counts[i]/all_counts[val_index]
        else:
            metric = None

        return metric

    def calc_agreement_labels(self, agreement_indices, instance_labels):
        """
        Get labels for agreement indices (single label or exact match case)

        Args:
            agreement_indices: of shape (epoch, num_agreed_indices)
            instance_labels: pandas column (img_paths, labels)

        Returns:

        """

        intersection_label_per_epoch = []
        # repeat indices for each epoch
        for epoch in range(len(agreement_indices)):
            # for each epoch, select correctly identified indices among networks
            intersection_label_per_epoch.append(instance_labels[agreement_indices[epoch]])

        return intersection_label_per_epoch

    def calc_agreement_multilabel(self, learned_instance_network_lists, labels_one_hot, agreement_type):
        """
        Chooses when to consider that 2 images agreed in a multilabel case:
        1.exact match,
        2. learning the same class, e.g. (0,1,0,0) for GT (0,1,1,0)
        3. learning at least one correct class

        Args:
            learned_instance_network_lists: of shape (networks, epochs+1, list of arrays num_classes)
            labels_one_hot: instance labels as one-hot-vectors

        Returns: new learned_instance_network_lists of shape (networks, epochs, 1 if img agreed else 0),
        lower_bound, mean_accuracy, agreement_indices
        """
        # for every first epoch, indices have also been written out. they have to be removed here
        net_list = [elem[1:] for elem in learned_instance_network_lists]
        indices = np.array(learned_instance_network_lists[0][0])

        predicted = np.array(net_list)
        num_nets, num_epochs, num_instances, num_classes = predicted.shape

        # since labels is just one list, it should be duplicated (networks, epochs)-time
        labels = np.tile(np.array(labels_one_hot[indices].tolist()).reshape(-1, num_classes),
                         (num_nets, num_epochs, 1, 1))

        if agreement_type == "exact_match":
            predicted_binary = (predicted > 0.5).astype('float')

            predicted_correct = np.sum(predicted_binary * labels, axis=-1)
            predicted_all = np.sum(predicted_binary, axis=-1)

            # to check whether there were false positives
            predicted_single = (predicted_correct == predicted_all).astype('int') * predicted_correct
            labels_single = np.sum(labels, axis=-1)

            exact_match = (predicted_single == labels_single).astype('int')

            # proceed as in a single label case
            # sum instances which all networks classified correctly
            # axis 0 is the networks one, axis 1 in that case - the instance one
            intersections_per_epoch = np.sum(np.prod(exact_match, axis=0), axis=1)

            # sum instances which at least one network classified correctly
            normalizer = np.sum(np.max(exact_match, axis=0), axis=1)
            # to avoid dividing by zero
            normalizer = np.where(normalizer == 0, 1, normalizer)

            # accuracy per epoch per network
            accuracy = np.sum(exact_match, axis=2) / float(exact_match.shape[2])
            mean_accuracy = np.mean(accuracy, axis=0)
            std_accuracy = np.std(accuracy, axis=0)

            # lower bound is 1 - min(sum_errors_per_network, 1)
            lower_bound = 1. - np.minimum(np.sum(1. - accuracy, axis=0), 1)

            # calculate agreement indices
            agreement_indices_per_epoch = []
            correctly_identified = np.prod(exact_match, axis=0).astype(bool)
            # multiply indices times epochs
            for epoch in range(exact_match.shape[1]):
                agreement_indices_per_epoch.append(indices[correctly_identified[epoch]])

        return 100. * intersections_per_epoch / normalizer, 100. * lower_bound, \
               100. * mean_accuracy, 100. * std_accuracy, agreement_indices_per_epoch


    def calc_agreement(self, learned_instance_network_lists, labels, save_prob_list):
        arr = np.array([elem[1:] for elem in learned_instance_network_lists])
        indices = np.array(learned_instance_network_lists[0][0])

        if save_prob_list:
            num_nets, num_epochs, num_instances, num_classes = arr.shape
            # since labels is just one list, it should be duplicated (networks, epochs)-time
            labels_tile = np.tile(labels[indices],
                             (num_nets, num_epochs, 1))

            # since arr are prediction probabilities, for exact match
            # max has to be calculated first
            arr = np.argmax(arr, axis=-1)
            arr = (arr == labels_tile).astype('int')
        else:
            num_nets, num_epochs, num_instances = arr.shape
            # since labels is just one list, it should be duplicated (networks, epochs)-time
            labels_tile = np.tile(labels[indices],
                                  (num_nets, num_epochs, 1))

            arr = (arr == labels_tile).astype('int')

        # sum instances which all networks classified correctly
        intersections_per_epoch = np.sum(np.prod(arr, axis=0), axis=1)

        # sum instances which at least one network classified correctly
        normalizer = np.sum(np.max(arr, axis=0), axis=1)
        normalizer = np.where(normalizer == 0, 1, normalizer)

        # accuracy per epoch per network
        # mean over instances
        accuracy = np.sum(arr, axis=2) / float(arr.shape[2])

        # mean over networks 
        mean_accuracy = np.mean(accuracy, axis=0)
        std_accuracy = np.std(accuracy, axis=0)
        
        # lower bound is 1 - min(sum_errors_per_network, 1)
        lower_bound = 1. - np.minimum(np.sum(1. - accuracy, axis=0), 1)

        # calculate agreement indices
        agreement_indices_per_epoch = []
        # correctly identified across networks
        correctly_identified = np.prod(arr, axis=0).astype(bool)
        # multiply indices times epochs
        for epoch in range(arr.shape[1]):
            agreement_indices_per_epoch.append(indices[correctly_identified[epoch]])

        return 100. * intersections_per_epoch / normalizer, \
               100. * lower_bound, 100. * mean_accuracy, 100. * std_accuracy, \
               agreement_indices_per_epoch
