"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from scipy import stats


# DISTANCE_MAT = torch.tensor([
DISTANCE_MAT = np.array([
            [0, 4, 3, 3, 4, 3, 2, 3, 1, 2],
            [4, 0, 7, 3, 2, 5, 6, 1, 5, 4],
            [3, 7, 0, 4, 5, 4, 3, 6, 2, 3],
            [3, 3, 4, 0, 3, 2, 3, 2, 2, 1],
            [4, 2, 5, 3, 0, 3, 4, 1, 3, 2],
            [3, 5, 4, 2, 3, 0, 1, 4, 2, 1],
            [2, 6, 3, 3, 4, 1, 0, 5, 1, 2],
            [3, 1, 6, 2, 1, 4, 5, 0, 4, 3],
            [1, 5, 2, 2, 3, 2, 1, 4, 0, 1],
            [2, 4, 3, 1, 2, 1, 2, 3, 1, 0]])
# dtype=torch.float)


class NeuralNet(nn.Module):
    def __init__(self, params):
        super(NeuralNet, self).__init__()
        self.fcIn = nn.Linear(params.input_size, params.hidden_size)
        self.fc2 = nn.Linear(params.hidden_size, params.hidden_size)
        self.fc3 = nn.Linear(params.hidden_size, params.hidden_size)  # //2)
        # self.fc4 = nn.Linear(params.hidden_size, params.hidden_size)  # params.num_classes)  # , bias=False)
        self.fcOut = nn.Linear(params.hidden_size, params.num_classes)  # , bias=False)

        self.dropout_rate = params.dropout_rate

    def forward(self, x):

        # first layer
        out = F.relu(self.fcIn(x))
        out = F.dropout(out, self.dropout_rate, training=self.training)

        # second layer
        out = F.relu(self.fc2(out))
        out = F.dropout(out, self.dropout_rate, training=self.training)

        # third layer
        out = F.relu(self.fc3(out))
        out = F.dropout(out, self.dropout_rate, training=self.training)

        # # forth layer
        # out = F.relu(self.fc4(out))
        # out = F.dropout(out, self.dropout_rate, training=self.training)

        # last layer
        out = self.fcOut(out)  # fc4(out)
        out = F.softmax(out, dim=1)
        # out = F.relu(self.fc4(out))
        # out = F.log_softmax(out, dim=1)

        return out


def convert_int_to_one_hot_vector(label, num_of_classes):

    if len(list(label.size())) < 3:
        label_shaped = label.view(-1, 1)

        one_hot_vector = torch.zeros([list(label.size())[0], num_of_classes], device=label.device)
        one_hot_vector.scatter_(1, label_shaped, 1)
        one_hot_vector = one_hot_vector.type(torch.FloatTensor)

        if torch.cuda.is_available():
            return one_hot_vector.cuda()
        return one_hot_vector

    else:
        # this is for 3d tensor . continue here!!!!!!
        labels_shaped = label.view(label.size(0), label.size(1), -1)

        one_hot_matrix = torch.zeros([list(labels_shaped.size())[0], list(labels_shaped.size())[1], num_of_classes], device=label.device)
        one_hot_matrix.scatter_(2, labels_shaped, 1)
        # added to keep a 2d dimension of labels
        one_hot_matrix = one_hot_matrix.view(-1, list(labels_shaped.size())[1] * num_of_classes)
        one_hot_matrix = one_hot_matrix.type(torch.FloatTensor)

        if torch.cuda.is_available():
            return one_hot_matrix.cuda()
        return one_hot_matrix


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
    schi_criterion = SCHILoss()
    if torch.cuda.is_available():
        schi_criterion = SCHILoss().cuda()

    return schi_criterion(outputs, labels)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # distance_mat = torch.from_numpy(DISTANCE_MAT).float().to(device)

    # mult = distance_mat[labels] * outputs
    # mult_sum = torch.sum(mult, dim=1)
    # mult_sum_avg = torch.mean(mult_sum)

    # return mult_sum_avg


class SCHILoss(nn.Module):
    def __init__(self):
        super(SCHILoss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.distance_mat = torch.from_numpy(DISTANCE_MAT).float().to(self.device)

    def forward(self, out, label):
        val = self.distance_mat[label] * out
        val = torch.sum(val, dim=1)
        return torch.mean(val)


class SCHITwoLabelsLoss(nn.Module):
    def __init__(self):
        super(SCHITwoLabelsLoss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.distance_mat = torch.from_numpy(DISTANCE_MAT).float().to(self.device)

    def forward(self, out_bef, out_aft, label_bef, label_aft):
        val_bef = self.distance_mat[label_bef] * out_bef
        val_bef = torch.sum(val_bef, dim=1)
        max_val_bef = torch.max(self.distance_mat[label_bef], dim=1)[0]
        val_bef = torch.mean(val_bef - max_val_bef)

        val_aft = self.distance_mat[label_aft] * out_aft
        val_aft = torch.sum(val_aft, dim=1)
        val_aft = torch.mean(val_aft)
        return val_bef + val_aft


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

    label_before_filter = torch.index_select(labels, 1, torch.tensor([0], device=labels.device))
    label_after_filter = torch.index_select(labels, 1, torch.tensor([1], device=labels.device))

    out_before_filter = torch.index_select(outputs, 1, torch.tensor(list(range(num_of_classes)), device=outputs.device))
    out_after_filter = torch.index_select(outputs, 1, torch.tensor(list(range(num_of_classes, 2*num_of_classes)), device=outputs.device))

    schi_two_labels_criterion = SCHITwoLabelsLoss()
    if torch.cuda.is_available():
        schi_two_labels_criterion = SCHITwoLabelsLoss().cuda()

    return schi_two_labels_criterion(out_before_filter, out_after_filter, label_before_filter, label_after_filter)

    # before_filter_dist_mat = []
    # after_filter_dist_mat = []
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # distance_mat = torch.from_numpy(DISTANCE_MAT).float().to(device)
    #
    # mult_before = distance_mat[label_before_filter] * out_before_filter
    # mult_sum_before = torch.sum(mult_before, dim=1)
    # max_before = torch.max(distance_mat[label_before_filter])
    # mult_sum_avg_before = torch.mean(mult_sum_before - max_before)
    #
    # mult_after = distance_mat[label_after_filter] * out_after_filter
    # mult_sum_after = torch.sum(mult_after, dim=1)
    # mult_sum_avg_after = torch.mean(mult_sum_after)
    #
    # return mult_sum_avg_before + mult_sum_avg_after


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
    samples_numpy = images.cpu().numpy()

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
    samples_numpy = images.cpu().numpy()

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
