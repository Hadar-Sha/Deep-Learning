"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy import stats


class NeuralNet(nn.Module):
    def __init__(self, params):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(params.input_size, params.hidden_size)
        self.fc2 = nn.Linear(params.hidden_size, params.hidden_size)
        self.fc3 = nn.Linear(params.hidden_size, params.hidden_size//2)
        self.fc4 = nn.Linear(params.hidden_size // 2, params.num_classes)

        self.dropout_rate = params.dropout_rate

    def forward(self, x):

        # first layer
        out = F.relu(self.fc1(x))
        out = F.dropout(out, self.dropout_rate, training=self.training)

        # second layer
        out = F.relu(self.fc2(out))
        out = F.dropout(out, self.dropout_rate, training=self.training)

        # third layer
        out = F.relu(self.fc3(out))
        out = F.dropout(out, self.dropout_rate, training=self.training)

        out = self.fc4(out)
        out = F.log_softmax(out, dim=1)

        return out


def convert_int_to_one_hot_vector(label, num_of_classes):

    # this is for 3d tensor . continue here!!!!!!
    # # Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
    # y1_int = torch.LongTensor(batch_size, 1).random_() % nb_digits
    # print(y1_int.size())
    # # y1_int = y1_int.view(-1,1)
    # # print(y1_int.size())
    # y1_int = y1_int.view(y1_int.size(0), y1_int.size(1), -1)
    # print(y1_int.size())
    # print(y1_int)
    #
    # # One hot encoding buffer that you create out of the loop and just keep reusing
    # y1_one_hot = torch.FloatTensor(batch_size, 1, nb_digits)
    # y1_one_hot.zero_()
    # print(y1_one_hot.size())
    # y1_one_hot.scatter_(2, y1_int, 1)

    size_in_list = list(label.size())
    size_in_int = size_in_list[0]
    one_hot_vector = torch.FloatTensor(size_in_int, num_of_classes)

    label_shaped = label.view(-1, 1)

    one_hot_vector.zero_()  # remove trash initial values
    one_hot_vector.scatter_(1, label_shaped, 1)
    return one_hot_vector


def loss_fn(outputs, labels, num_of_classes):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size x 10 - output of the model
        labels: (Variable) dimension batch_size, where each element is a value in [0- 9]
        num_of_classes: (int) value describing number of different classes (10)

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """

    kl_criterion = nn.KLDivLoss()
    one_hot_vector = convert_int_to_one_hot_vector(labels, num_of_classes)

    return kl_criterion(outputs, one_hot_vector)


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        val = -x*torch.exp(x)
        val = torch.sum(val, dim=1)
        return torch.mean(val)


def loss_fn_two_labels(outputs, labels, num_of_classes):
    """
        Compute the cross entropy loss given outputs and labels.

        Args:
            outputs: (Variable) dimension batch_size x 10 - output of the model
            labels: (Variable) dimension batch_size, where each element is a value in [0- 9]
            num_of_classes: (int) value describing number of different classes (10)

        Returns:
            loss (Variable): cross entropy loss for all images in the batch
    """

    kl_criterion = nn.KLDivLoss(size_average=True, reduce=True)
    min_entropy_criterion = HLoss()

    label_before_filter = torch.index_select(labels, 1, torch.tensor([0]))
    label_after_filter = torch.index_select(labels, 1, torch.tensor([1]))

    label_before_numpy = label_before_filter.numpy()
    label_after_numpy = label_after_filter.numpy()

    other_labels_before_numpy = np.setdiff1d(np.arange(num_of_classes-1), label_before_numpy)
    other_labels_after_numpy = np.setdiff1d(np.arange(num_of_classes-1), label_after_numpy)

    other_labels_before = torch.from_numpy(other_labels_before_numpy)
    other_labels_after = torch.from_numpy(other_labels_after_numpy)

    alpha = 0.5

    one_hot_vector_before_filter = convert_int_to_one_hot_vector(label_before_filter, num_of_classes)  # unneeded
    one_hot_vector_after_filter = convert_int_to_one_hot_vector(label_after_filter, num_of_classes)

    out_before_filter = torch.index_select(outputs, 1, torch.tensor(list(range(10))))
    out_after_filter = torch.index_select(outputs, 1, torch.tensor(list(range(10, 20))))

    completing_after_filter = (torch.ones(labels.shape[0], num_of_classes) - one_hot_vector_after_filter)\
                               / (num_of_classes-1)

    # temp = 1 - kl_criterion(out_before_filter, one_hot_vector_after_filter)
    # print(temp.item())
    # ent = min_entropy_criterion(out_before_filter)
    # print(ent)

    # func = kl_criterion(out_after_filter, one_hot_vector_after_filter) + \
    #        (1-kl_criterion(out_before_filter, one_hot_vector_after_filter))/labels.shape[0] + \
    #        min_entropy_criterion(out_before_filter)

    # func = kl_criterion(out_after_filter, one_hot_vector_after_filter) + \
    #        1/kl_criterion(out_before_filter, one_hot_vector_after_filter) + \
    #        min_entropy_criterion(out_before_filter)

    func = kl_criterion(out_after_filter, one_hot_vector_after_filter) + \
           kl_criterion(out_before_filter, completing_after_filter) + \
           min_entropy_criterion(out_before_filter)

    # func = kl_criterion(out_after_filter, one_hot_vector_after_filter) + \
    #        kl_criterion(out_before_filter, one_hot_vector_before_filter) + \
    #        min_entropy_criterion(out_after_filter)

    return func


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 10 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0-9]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs == labels)/float(labels.size)


def accuracy_two_labels(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 10 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0-9]

    Returns: (float) accuracy in [0,1]
    """
    if len(list(labels.shape)) == 1:
        return 0.0

    label_before_filter = labels[:, 0]  # unneeded
    label_after_filter = labels[:, 1]

    out_before_filter = outputs[:, :10]
    out_after_filter = outputs[:, 10:]

    out_int_before = np.argmax(out_before_filter, axis=1)
    out_int_after = np.argmax(out_after_filter, axis=1)

    # the classification before filter is correct as long as it is not equal to label after filter
    before_ind = np.nonzero(out_int_before != label_after_filter)
    after_ind = np.nonzero(out_int_after == label_after_filter)
    all_ind = np.intersect1d(before_ind, after_ind)

    all_count = all_ind.shape[0]
    return all_count / float(labels.shape[0])


def incorrect(images, outputs, labels):
    """
        Keep all images for which the classification is wrong

        Args:
            images: (np.ndarray) dimension batch_size x 24- input to the model
            outputs: (np.ndarray) dimension batch_size x 10 - log softmax output of the model
            labels: (np.ndarray) dimension batch_size, where each element is a value in [0- 9]

        Returns: (list) of images for which the classification is wrong, the classification and the correct label
        """
    mat_out = []
    outputs = np.argmax(outputs, axis=1)
    # find incorrect indexes
    current_incorrect_bin = (outputs != labels)
    current_incorrect_indexes = np.nonzero(current_incorrect_bin)

    # find compatible incorrect samples and save them in a list
    samples_numpy = images.numpy()

    # find samples
    incorrect_samples = (samples_numpy[current_incorrect_indexes]).astype(int)

    # find classifier result
    labels_pred_numpy = outputs
    incorrect_labels = labels_pred_numpy[current_incorrect_indexes]

    # find true labels
    labels_actual_numpy = labels
    true_labels = labels_actual_numpy[current_incorrect_indexes]

    # organize data
    all_labels = np.column_stack((incorrect_labels, true_labels))
    numpy_mat_out = np.concatenate((incorrect_samples, all_labels), axis=1)
    length = len(numpy_mat_out.tolist())
    if length > 0:
        mat_out.extend(numpy_mat_out.tolist())

    return mat_out


def incorrect_two_labels(images, outputs, labels):
    """
        Keep all images for which the classification is wrong

        Args:
            images: (np.ndarray) dimension batch_size x 24- input to the model
            outputs: (np.ndarray) dimension batch_size x 10 - log softmax output of the model
            labels: (np.ndarray) dimension batch_size, where each element is a value in [0- 9]

        Returns: (list) of images for which the classification is wrong, the classification and the correct label
        """
    mat_out = []

    label_before_filter = labels[:, 0]  # unneeded
    label_after_filter = labels[:, 1]

    out_before_filter = outputs[:, :10]
    out_after_filter = outputs[:, 10:]

    out_int_before = np.argmax(out_before_filter, axis=1)
    out_int_after = np.argmax(out_after_filter, axis=1)

    # find incorrect indexes
    # the classification before filter is incorrect only if it is equal to label after filter
    correct_before_indexes = np.nonzero(out_int_before != label_after_filter)  # label_after_filter : not a typo!!!!
    correct_after_indexes = np.nonzero(out_int_after == label_after_filter)
    all_indexes = np.arange(out_int_before.shape[0])  # to get all indices
    all_correct_indexes = np.intersect1d(correct_before_indexes, correct_after_indexes)
    incorrect_indexes = np.setdiff1d(all_indexes, all_correct_indexes)

    # find compatible incorrect samples and save them in a list
    samples_numpy = images.numpy()

    # find samples
    incorrect_samples = (samples_numpy[incorrect_indexes]).astype(int)

    # find classifier result for before filter
    incorrect_before_labels = out_int_before[incorrect_indexes]

    # find classifier result for after filter
    incorrect_after_labels = out_int_after[incorrect_indexes]

    # find true labels for before filter
    true_before_labels = label_before_filter[incorrect_indexes]

    # find true labels for after filter
    true_after_labels = label_after_filter[incorrect_indexes]

    # organize data
    before_labels = np.column_stack((incorrect_before_labels, true_before_labels))
    after_labels = np.column_stack((incorrect_after_labels, true_after_labels))
    all_labels = np.column_stack((before_labels, after_labels))

    numpy_mat_out = np.concatenate((incorrect_samples, all_labels), axis=1)
    length = len(numpy_mat_out.tolist())
    if length > 0:
        mat_out.extend(numpy_mat_out.tolist())

    return mat_out


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    'accuracy_two_labels': accuracy_two_labels
    # could add more metrics such as accuracy for each token type
}
