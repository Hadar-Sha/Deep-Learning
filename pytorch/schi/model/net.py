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
        self.fc4 = nn.Linear(params.hidden_size // 2, params.num_classes)  # , bias=False)

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
        # out = F.relu(self.fc4(out))
        out = F.log_softmax(out, dim=1)

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
        # this is for 3d tensor .
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
        Compute the loss given outputs and labels.
        we will achieve max KL dist between out_bef_filt and lab_aft_filt by the following:
        we wish the output to be not equal to lab_aft_filt.
        given lab_aft_filt in [0-9] we can create a binary vector of size [num_of_classes] with all entrances = 1 if
        entrance != lab_aft_filt and entrance = 0 otherwise.
        we normalize this vector v to have sum = 1 by dividing in (num_of_classes-1)
        we then calculate KL loss between out_bef_filt and v

        Args:
            outputs: (Variable) dimension batch_size x 10 - output of the model
            labels: (Variable) dimension batch_size, where each element is a value in [0- 9]
            num_of_classes: (int) value describing number of different classes (10)

        Returns:
            loss (Variable): loss for all samples in the batch
    """

    # kl_criterion = nn.KLDivLoss(size_average=True, reduce=True)
    kl_criterion = nn.KLDivLoss()
    min_entropy_criterion = HLoss()

    label_before_filter = torch.index_select(labels, 1, torch.tensor([0], device=labels.device))
    label_after_filter = torch.index_select(labels, 1, torch.tensor([1], device=labels.device))

    one_hot_vector_after_filter = convert_int_to_one_hot_vector(label_after_filter, num_of_classes)
    one_hot_vector_before_filter = convert_int_to_one_hot_vector(label_before_filter, num_of_classes)  # unneeded

    out_before_filter = torch.index_select(outputs, 1, torch.tensor(list(range(10)), device=outputs.device))
    out_after_filter = torch.index_select(outputs, 1, torch.tensor(list(range(10, 20)), device=outputs.device))

    completing_after_filter = (torch.ones(labels.shape[0], num_of_classes) - one_hot_vector_after_filter)\
                               / (num_of_classes-1)

    func = kl_criterion(out_after_filter, one_hot_vector_after_filter) + \
           kl_criterion(out_before_filter, completing_after_filter) + \
           min_entropy_criterion(out_before_filter)

    return func


def loss_fn_two_labels_exp_kl(outputs, labels, num_of_classes):
    """
        Compute the loss given outputs and labels.
        we will achieve max KL dist between out_bef_filt and lab_aft_filt by the mathematical expression:
        1 - exp(-kl(out_bef_filt,lab_aft_filt)

        Args:
            outputs: (Variable) dimension batch_size x 10 - output of the model
            labels: (Variable) dimension batch_size, where each element is a value in [0- 9]
            num_of_classes: (int) value describing number of different classes (10)

        Returns:
            loss (Variable): loss for all samples in the batch
    """

    kl_criterion = nn.KLDivLoss()
    min_entropy_criterion = HLoss()

    label_before_filter = torch.index_select(labels, 1, torch.tensor([0], device=labels.device))
    label_after_filter = torch.index_select(labels, 1, torch.tensor([1], device=labels.device))

    one_hot_vector_after_filter = convert_int_to_one_hot_vector(label_after_filter, num_of_classes)
    one_hot_vector_before_filter = convert_int_to_one_hot_vector(label_before_filter, num_of_classes)  # unneeded

    out_before_filter = torch.index_select(outputs, 1, torch.tensor(list(range(10)), device=outputs.device))
    out_after_filter = torch.index_select(outputs, 1, torch.tensor(list(range(10, 20)), device=outputs.device))

    func = min_entropy_criterion(out_before_filter) + kl_criterion(out_after_filter, one_hot_vector_after_filter) + \
           (1 - torch.exp(- kl_criterion(out_before_filter, one_hot_vector_after_filter)))

    return func


def loss_fn_two_labels_min(outputs, labels, num_of_classes):
    """
        Compute the loss given outputs and labels.
        we will achieve max KL dist between out_bef_filt and lab_aft_filt by the following:
        we wish the output to be not equal to lab_aft_filt.
        given lab_aft_filt in [0-9] we can create a list of all other possible labels.
        we will get (num_of_classes-1) labels per each sample.
        we then calculate KL loss between out_bef_filt and each other possible label and return
        the mininal KL distance to a cetrain label which is not lab_aft_filt

        Args:
            outputs: (Variable) dimension batch_size x 10 - output of the model
            labels: (Variable) dimension batch_size, where each element is a value in [0- 9]
            num_of_classes: (int) value describing number of different classes (10)

        Returns:
            loss (Variable): loss for all images in the batch
    """

    # kl_criterion = nn.KLDivLoss(size_average=True, reduce=True)
    kl_criterion = nn.KLDivLoss()
    min_entropy_criterion = HLoss()

    label_before_filter = torch.index_select(labels, 1, torch.tensor([0], device=labels.device))
    label_after_filter = torch.index_select(labels, 1, torch.tensor([1], device=labels.device))

    all_labels_mat = np.arange(num_of_classes)*np.ones((outputs.size()[0], 1), dtype=int)

    temp_bef = all_labels_mat - label_before_filter.cpu().numpy()
    temp_aft = all_labels_mat - label_after_filter.cpu().numpy()

    other_labels_before = torch.from_numpy(all_labels_mat[np.nonzero(temp_bef)].reshape(outputs.size()[0], num_of_classes-1))
    other_labels_after = torch.from_numpy(all_labels_mat[np.nonzero(temp_aft)].reshape(outputs.size()[0], num_of_classes-1))

    other_labels_before = other_labels_before.type(torch.LongTensor)

    many_hot_vector_before_filter = convert_int_to_one_hot_vector(other_labels_before, num_of_classes)
    one_hot_vector_after_filter = convert_int_to_one_hot_vector(label_after_filter, num_of_classes)
    one_hot_vector_before_filter = convert_int_to_one_hot_vector(label_before_filter, num_of_classes)  # unneeded

    out_before_filter = torch.index_select(outputs, 1, torch.tensor(list(range(10)), device=outputs.device))
    out_after_filter = torch.index_select(outputs, 1, torch.tensor(list(range(10, 20)), device=outputs.device))

    # min_ind = 0

    # for each option of 9 other labels (all besides the label after filter)
    # calculate the kl distance from the output and keep the minimal one for the loss function

    min_dist = kl_criterion(out_before_filter, many_hot_vector_before_filter[:, 0])
    # ind_array = np.random.permutation(num_of_classes-1).tolist()
    for i in range(num_of_classes-1):
        # other_one_hot_vector_before_filter = many_hot_vector_before_filter[:, ind_array[i]]
        other_one_hot_vector_before_filter = many_hot_vector_before_filter[:, i]
        dist_before = kl_criterion(out_before_filter, other_one_hot_vector_before_filter)
        if dist_before < min_dist:
            min_dist = dist_before
            # min_ind = i

    func = kl_criterion(out_after_filter, one_hot_vector_after_filter) + \
            min_dist + min_entropy_criterion(out_before_filter)

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
