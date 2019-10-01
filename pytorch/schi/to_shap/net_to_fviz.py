"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(24, 98)
        self.fc2 = nn.Linear(98, 98)
        self.fc3 = nn.Linear(98, 98//2)
        self.fc4 = nn.Linear(98// 2, 10)  # , bias=False)

        self.dropout_rate = 0.25

    def forward(self, x):

        # first layer
        out_1_1 = F.relu(self.fc1(x))
        out_1_f = F.dropout(out_1_1, self.dropout_rate, training=self.training)

        # second layer
        out_2_1 = F.relu(self.fc2(out_1_f))
        out_2_f = F.dropout(out_2_1, self.dropout_rate, training=self.training)

        # third layer
        out_3_1 = F.relu(self.fc3(out_2_f))
        out_3_f = F.dropout(out_3_1, self.dropout_rate, training=self.training)

        out_4_l = self.fc4(out_3_f)
        # print(out)
        # out = F.relu(self.fc4(out))
        out_l_s = F.log_softmax(out_4_l, dim=1)   # on purpose
        out_s = F.softmax(out_4_l, dim=1)   # on purpose

        return out_1_f, out_2_f, out_3_f, out_4_l, out_l_s, out_s
        # return out


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

    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    one_hot_vector = convert_int_to_one_hot_vector(labels, num_of_classes)

    return kl_criterion(outputs, one_hot_vector)


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


def incorrect(images, outputs, labels, curr_min=0, curr_max=1, dest_min=0, dest_max=255):
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

    # convert back to range [0, 255]
    samples_numpy = \
        dest_min + (dest_max - dest_min) * (samples_numpy - curr_min) / (curr_max - curr_min)
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


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy
    # could add more metrics such as accuracy for each token type
}
