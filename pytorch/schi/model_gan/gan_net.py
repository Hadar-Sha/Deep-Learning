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
            nn.Linear(params.input_size, params.hidden_size),
            nn.BatchNorm1d(params.hidden_size),
            nn.ReLU()
            # nn.LeakyReLU(params.leaky_relu_slope),
            # nn.Dropout(params.dropout_rate)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(params.hidden_size, params.hidden_size),
            nn.BatchNorm1d(params.hidden_size),
            nn.ReLU()
            # nn.LeakyReLU(params.leaky_relu_slope),
            # nn.Dropout(params.dropout_rate)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(params.hidden_size, params.hidden_size),
            nn.BatchNorm1d(params.hidden_size),
            nn.ReLU()
            # nn.LeakyReLU(params.leaky_relu_slope),
            # nn.Dropout(params.dropout_rate)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(params.hidden_size, params.hidden_size),
            nn.BatchNorm1d(params.hidden_size),
            nn.ReLU()
            # nn.LeakyReLU(params.leaky_relu_slope),
            # nn.Dropout(params.dropout_rate)
        )
        self.out_layer = nn.Sequential(
            nn.Linear(params.hidden_size, 1),
            # nn.ReLU(),
            # nn.LogSoftmax(dim=1)
            nn.Sigmoid()
        )

    def forward(self, x):

        out = self.in_layer(x)
        out = self.hidden1(out)
        out = self.hidden2(out)
        out = self.hidden3(out)
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
        self.in_layer = nn.Sequential(
            nn.Linear(params.noise_dim, params.hidden_size),
            nn.BatchNorm1d(params.hidden_size),
            nn.ReLU()
            # nn.LeakyReLU(params.leaky_relu_slope),
            # nn.Dropout(params.dropout_rate)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(params.hidden_size, params.hidden_size),
            nn.BatchNorm1d(params.hidden_size),
            nn.ReLU()
            # nn.LeakyReLU(params.leaky_relu_slope),
            # nn.Dropout(params.dropout_rate)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(params.hidden_size, params.hidden_size),
            nn.BatchNorm1d(params.hidden_size),
            nn.ReLU()
            # nn.LeakyReLU(params.leaky_relu_slope),
            # nn.Dropout(params.dropout_rate)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(params.hidden_size, params.hidden_size),
            nn.BatchNorm1d(params.hidden_size),
            nn.ReLU()
            # nn.LeakyReLU(params.leaky_relu_slope),
            # nn.Dropout(params.dropout_rate)
        )
        self.out_layer = nn.Sequential(
            nn.Linear(params.hidden_size, params.input_size),
            # nn.ReLU(),
            # nn.LogSoftmax(dim=1)
            nn.Sigmoid()
            # nn.Softmax(dim=1)
        )
        # self.linear_normalization = linear_transformation()

    # def linear_transformation(self, x):
    #     min_v = torch.min(x, 0, True)
    #     range_v = torch.max(x, 0, True) - min_v
    #     if range_v > 0:
    #         normalised = (x - min_v) / range_v
    #     else:
    #         normalised = torch.zeros(x.size())
    #     return normalised

    def forward(self, x):

        out = self.in_layer(x)
        out = self.hidden1(out)
        out = self.hidden2(out)
        out = self.hidden3(out)
        out = self.out_layer(out)
        # out = linear_transformation(out)

        return out


# Noise
def noise(size, dim, noise_type='normal'):
    if noise_type == 'normal':
        n = Variable(torch.randn(size, dim))  # recommended to sample from normal distribution and not from uniform dist
    elif noise_type == 'uniform':
        n = Variable(-1 + 2 * torch.rand(size, dim))  # make sense for binary samples
    elif noise_type == 'binary':
        n = Variable(-1 + 2 * torch.randint(2, (size, dim), dtype=torch.float))  # make sense for binary samples
    # # n = Variable(torch.rand(size, num_of_classes))
    # # print(n)
    # # n = Variable(torch.randint(256, (size, 100)))
    # n = Variable(torch.randn(size, 10))
    if torch.cuda.is_available():
        return n.cuda()
    return n


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
        # this is for 3d tensor
        labels_shaped = label.view(label.size(0), label.size(1), -1)

        one_hot_matrix = torch.zeros([list(labels_shaped.size())[0], list(labels_shaped.size())[1], num_of_classes], device=label.device)
        one_hot_matrix.scatter_(2, labels_shaped, 1)
        # added to keep a 2d dimension of labels
        one_hot_matrix = one_hot_matrix.view(-1, list(labels_shaped.size())[1]*num_of_classes)
        one_hot_matrix = one_hot_matrix.type(torch.FloatTensor)

        if torch.cuda.is_available():
            return one_hot_matrix.cuda()
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
    vectors = vectors.cpu().numpy()
    vectors = vectors.tolist()
    # vectors = (vectors/255).tolist()
    return vectors


def labels_to_titles(labels):
    if len(labels.shape) > 1 and min(labels.shape) == 1:
        labels = labels.view(labels.size()[0],)
    labels = (labels.cpu().numpy()).tolist()
    return labels


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        # nn.init.uniform_(m.bias.data, -1, 1)

    if torch.cuda.is_available():
        for pa in m.parameters():
            pa.cuda()


def loss_fn(outputs, labels):  # , num_of_classes):
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
    # kl_criterion = nn.KLDivLoss()
    # one_hot_vector = convert_int_to_one_hot_vector(labels, num_of_classes)

    # return kl_criterion(outputs, one_hot_vector)


