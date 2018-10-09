"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    def __init__(self, params):  # input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(params.input_size, params.hidden_size)
        self.fc2 = nn.Linear(params.hidden_size, params.hidden_size)  # (hidden_size, int(hidden_size*0.75))
        self.fc3 = nn.Linear(params.hidden_size, params.hidden_size//2)  # (int(hidden_size*0.75), hidden_size//2)
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
        # out = F.softmax(out, dim=1)
        out = F.log_softmax(out, dim=1)
        # print(out)

        return out


def convert_int_to_one_hot_vector(label, num_of_classes):

    size_in_list = list(label.size())
    size_in_int = size_in_list[0]
    # print(val)
    one_hot_vector = torch.FloatTensor(size_in_int, num_of_classes)
    # one_hot_vector = torch.FloatTensor(batch_size, num_of_classes)

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

    # crit = nn.NLLLoss(size_average=True, reduce=True)
    # # crit = nn.CrossEntropyLoss(size_average=True, reduce=True)
    #
    # return crit(outputs, labels)

    kl_criterion = nn.KLDivLoss()
    one_hot_vector = convert_int_to_one_hot_vector(labels, num_of_classes)

    return kl_criterion(outputs, one_hot_vector)

    # # num_examples = outputs.size()[0]
    # # print(num_examples)
    # return -torch.sum(outputs[range(num_examples), labels])/num_examples


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

# labels_test = labels_test.view(labels_test.size(0))
#
#         outputs_test = net(images_test)
#
#         _, predicted = torch.max(outputs_test.data, 1)
#
#         total += labels_test.size(0)
#         correct += (predicted == labels_test).sum()
#
#     if ep % 100 == 0 or ep == num_epochs-1:
#         print('Accuracy of the model on the {} test images: {} %' .format(total, (100 * correct / total)))


def incorrect(images, outputs, labels):
    """
        Compute the accuracy, given the outputs and labels for all images.
        Keep all images for which the classification is wrong

        Args:
            images: (np.ndarray) dimension batch_size x 24- input to the model
            outputs: (np.ndarray) dimension batch_size x 10 - log softmax output of the model
            labels: (np.ndarray) dimension batch_size, where each element is a value in [0- 9]

        Returns: (list) of images for which the classification is wrong, the classification and the correct label
        """
    current_mat_out = []
    outputs = np.argmax(outputs, axis=1)
    # find incorrect indexes
    current_incorrect_bin = (outputs != labels)
    current_incorrect_indexes = np.nonzero(current_incorrect_bin)

    # find compatible incorrect samples and save them in a list
    samples_numpy = images.numpy()

    # find samples
    incorrect_samples = samples_numpy[current_incorrect_indexes]
    incorrect_samples = incorrect_samples.astype(int)

    # find classifier result
    labels_pred_numpy = outputs
    incorrect_labels = labels_pred_numpy[current_incorrect_indexes]
    incorrect_labels = incorrect_labels.astype(int)

    # find true labels
    labels_actual_numpy = labels
    true_labels = labels_actual_numpy[current_incorrect_indexes]
    true_labels = true_labels.astype(int)

    # organize data
    temp_labels = np.column_stack((incorrect_labels, true_labels))
    current_numpy_mat_out = np.concatenate((incorrect_samples, temp_labels), axis=1)
    length = len(current_numpy_mat_out.tolist())
    if length > 0:
        current_mat_out.extend(current_numpy_mat_out.tolist())

    return current_mat_out


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    # could add more metrics such as accuracy for each token type
    # 'incorrect': incorrect,
}
