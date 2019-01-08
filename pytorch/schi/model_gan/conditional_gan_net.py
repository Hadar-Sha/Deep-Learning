"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd.variable import Variable


class DiscriminatorNet(nn.Module):
    def __init__(self, params):
        super(DiscriminatorNet, self).__init__()
        self.in_layer = nn.Sequential(
            # nn.Linear(params.input_size, params.hidden_size//4),
            # nn.Linear(params.input_size, params.hidden_size//2),
            nn.Linear(params.input_size, params.hidden_size),
            # nn.ReLU(),
            # nn.LeakyReLU(params.leaky_relu_slope),
            nn.Dropout(params.dropout_rate)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(params.hidden_size, params.hidden_size),
            nn.ReLU(),
            # nn.LeakyReLU(params.leaky_relu_slope),
            nn.Dropout(params.dropout_rate)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(params.hidden_size, params.hidden_size),
            # nn.Linear(params.hidden_size, params.hidden_size//2),
            # nn.Linear(params.hidden_size//2, params.hidden_size//4),
            nn.ReLU(),
            # nn.LeakyReLU(params.leaky_relu_slope),
            nn.Dropout(params.dropout_rate)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(params.hidden_size, params.hidden_size),
            # nn.Linear(params.hidden_size // 2, params.hidden_size // 2),
            # nn.Linear(params.hidden_size // 2, params.hidden_size // 4),
            nn.ReLU(),
            # nn.LeakyReLU(params.leaky_relu_slope),
            nn.Dropout(params.dropout_rate)
        )
        self.fc_insert_label = nn.Sequential(
            nn.Linear(params.num_classes, params.hidden_size),
            # nn.Linear(params.num_classes, params.hidden_size),
            # nn.Linear(1, params.hidden_size),
            nn.ReLU(),
            # nn.Dropout(params.dropout_rate)
        )

        self.hidden_with_label = nn.Sequential(
            nn.Linear(params.hidden_size + params.hidden_size, params.hidden_size),
            nn.ReLU(),
            # nn.Dropout(params.dropout_rate)
        )

        self.out_layer = nn.Sequential(
            nn.Linear(params.hidden_size, 1),
            # nn.Linear(params.hidden_size // 2, 1),
            # nn.Linear(params.hidden_size // 4, 1),
            # nn.ReLU(),
            nn.Sigmoid()
        )

        # self.fc_insert_label = nn.Linear(1, 500)
        # self.fc_for_label = nn.Linear(params.num_classes, 1000)

        # self.fc1 = nn.Linear((params.hidden_size//2)+500, params.hidden_size//2)
        # self.fc1 = nn.Linear((params.hidden_size//2)*(params.hidden_size//2)+1000, params.hidden_size//2)

    def forward(self, x, labels):

        batch_size = x.size(0)
        x_ = self.in_layer(x)
        x_ = self.hidden1(x_)
        x_ = self.hidden2(x_)
        x_ = self.hidden3(x_)
        # out = out.view(batch_size, 49*49)

        y_ = self.fc_insert_label(labels)
        # y_ = F.relu(y_)
        out = torch.cat([x_, y_], 1)
        out = self.hidden_with_label(out)
        out = self.out_layer(out)

        return out


def linear_transformation(x):
    min_v, _ = torch.min(x, 0, True)
    max_v, _ = torch.max(x, 0, True)
    range_v = max_v - min_v
    ones_mat = torch.ones(x.size(0), 1)  # , dtype=torch.int)

    min_mat = torch.mm(ones_mat, min_v)
    range_mat = torch.mm(ones_mat, range_v)

    zeros_range_ind = (range_mat == 0).nonzero()

    if zeros_range_ind.nelement() == 0:
        normalised = \
            (x - min_mat) / range_mat
    else:
        print("in else of linear_transformation function")
        normalised = torch.zeros(x.size())

    return normalised


class GeneratorNet(nn.Module):
    def __init__(self, params):
        super(GeneratorNet, self).__init__()

        self.fc_insert_label = nn.Sequential(
            nn.Linear(params.num_classes, params.hidden_size),
            # nn.Linear(params.num_classes, params.hidden_size),
            # nn.Linear(1, params.hidden_size),
            # nn.ReLU(),
            nn.LeakyReLU(params.leaky_relu_slope),
            # nn.Dropout(params.dropout_rate)
        )
        self.hidden_with_label = nn.Sequential(
            nn.Linear(params.noise_dim + params.hidden_size, params.hidden_size),
            # nn.ReLU(),
            nn.LeakyReLU(params.leaky_relu_slope),
            # nn.Dropout(params.dropout_rate)
        )

        # self.in_layer = nn.Sequential(
        #     nn.Linear(100, params.hidden_size // 2),
        #     # nn.Linear(params.num_classes, params.hidden_size // 2),
        #     nn.ReLU(),
        #     # nn.LeakyReLU(params.leaky_relu_slope),
        #     # nn.Linear(params.input_size, params.hidden_size),
        #     nn.Dropout(params.dropout_rate)
        # )
        self.hidden1 = nn.Sequential(
            nn.Linear(params.hidden_size, params.hidden_size),
            # nn.ReLU(),
            nn.LeakyReLU(params.leaky_relu_slope),
            # nn.Linear(params.hidden_size, params.hidden_size),
            nn.Dropout(params.dropout_rate)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(params.hidden_size, params.hidden_size),
            # nn.ReLU(),
            nn.LeakyReLU(params.leaky_relu_slope),
            # nn.Linear(params.hidden_size, params.hidden_size//2),
            nn.Dropout(params.dropout_rate)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(params.hidden_size, params.hidden_size//2),
            # nn.ReLU(),
            nn.LeakyReLU(params.leaky_relu_slope),
            # nn.Linear(params.hidden_size, params.hidden_size//2),
            nn.Dropout(params.dropout_rate)
        )
        self.out_layer = nn.Sequential(
            nn.Linear(params.hidden_size//2, params.input_size),
            # nn.Linear(params.hidden_size // 2, params.num_classes),
        )
        # self.linear_normalization = linear_transformation()

    def forward(self, x, labels):

        # out = self.in_layer(x)
        y_ = self.fc_insert_label(labels)
        out = torch.cat([x, y_], 1)
        out = self.hidden_with_label(out)
        # out = out.view(batch_size, 49, 49)
        out = self.hidden1(out)
        out = self.hidden2(out)
        out = self.hidden3(out)
        out = self.out_layer(out)
        out = linear_transformation(out)
        # print(out.size())

        return out


# Noise
def noise(size):
    # n = Variable(torch.rand(size, num_of_classes))
    n = Variable(torch.randint(256, (size, 100)))
    if torch.cuda.is_available():
        return n.cuda()
    return n


def convert_int_to_one_hot_vector(label, num_of_classes):

    if min(list(label.size())) == 1:
        label_shaped = label.view(-1, 1)

        one_hot_vector = torch.FloatTensor(list(label.size())[0], num_of_classes)
        one_hot_vector.zero_()  # set all values to zero

        one_hot_vector.scatter_(1, label_shaped, 1)
        return one_hot_vector

    else:
        # print(label.size())
        # this is for 3d tensor . continue here!!!!!!
        labels_shaped = label.view(label.size(0), label.size(1), -1)
        # print(labels_shaped.size())

        one_hot_matrix = torch.FloatTensor(list(labels_shaped.size())[0], list(labels_shaped.size())[1], num_of_classes)
        one_hot_matrix.zero_()  # set all values to zero
        # print(one_hot_matrix.size())
        one_hot_matrix.scatter_(2, labels_shaped, 1)
        # added to keep a 2d dimension of labels
        one_hot_matrix = one_hot_matrix.view(-1, list(labels_shaped.size())[1]*num_of_classes)
        return one_hot_matrix


def real_data_target(size):
    """
    Tensor containing ones, with shape = size
    """
    data = Variable(torch.ones(size, 1))
    if torch.cuda.is_available():
        return data.cuda()
    return data


def fake_data_target(size):
    """
    Tensor containing zeros, with shape = size
    """
    data = Variable(torch.zeros(size, 1))
    if torch.cuda.is_available():
        return data.cuda()
    return data


def vectors_to_samples(vectors):
    vectors = vectors.reshape(vectors.size()[0], -1, 3)
    vectors = vectors.numpy()
    vectors = vectors.tolist()
    return vectors


def labels_to_titles(labels):
    labels = (labels.numpy()).tolist()
    return labels


def create_random_labels(num_samples):
    all_but_eight = np.concatenate((np.arange(8), np.array([[9]])), axis=None)
    random_mat = np.zeros([num_samples, 2])
    for i in range(num_samples):
        random_mat[i][:] = np.random.choice(all_but_eight, 2, replace=False)

    random_tensor = torch.from_numpy(random_mat)
    random_tensor = random_tensor.type(torch.LongTensor)
    return random_tensor


def loss_fn(outputs, labels):
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

    binary_criterion = nn.BCELoss()
    return binary_criterion(outputs, labels)


