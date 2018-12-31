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
            nn.LeakyReLU(params.leaky_relu_slope),
            nn.Dropout(params.dropout_rate)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(params.hidden_size, params.hidden_size),
            # nn.ReLU(),
            nn.LeakyReLU(params.leaky_relu_slope),
            nn.Dropout(params.dropout_rate)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(params.hidden_size, params.hidden_size//2),
            # nn.Linear(params.hidden_size//2, params.hidden_size//4),
            # nn.ReLU(),
            nn.LeakyReLU(params.leaky_relu_slope),
            nn.Dropout(params.dropout_rate)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(params.hidden_size // 2, params.hidden_size // 2),
            # nn.Linear(params.hidden_size // 2, params.hidden_size // 4),
            # nn.ReLU(),
            nn.LeakyReLU(params.leaky_relu_slope),
            nn.Dropout(params.dropout_rate)
        )
        self.out_layer = nn.Sequential(
            # nn.Linear(params.hidden_size // 4, 1),
            nn.Linear(params.hidden_size // 2, 1),
            # nn.ReLU(),
            # nn.LogSoftmax(dim=1)
            nn.Sigmoid()
        )

        # self.dropout_rate = params.dropout_rate

        # self.fc1 = nn.Linear(params.input_size, params.hidden_size)
        # self.fc2 = nn.Linear(params.hidden_size, params.hidden_size)
        # self.fc3 = nn.Linear(params.hidden_size, params.hidden_size//2)
        # self.fc4 = nn.Linear(params.hidden_size // 2, params.num_classes)
        #
        # self.dropout_rate = params.dropout_rate

    def forward(self, x):

        out = self.in_layer(x)
        out = self.hidden1(out)
        out = self.hidden2(out)
        out = self.hidden3(out)
        out = self.out_layer(out)
        # print(out.size())

        return out


class GeneratorNet(nn.Module):
    def __init__(self, params):
        super(GeneratorNet, self).__init__()
        self.in_layer = nn.Sequential(
            nn.Linear(params.num_classes, params.hidden_size // 2),
            nn.ReLU(True),
            # nn.LeakyReLU(params.leaky_relu_slope),
            # nn.Linear(params.input_size, params.hidden_size),
            # nn.Dropout(params.dropout_rate)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(params.hidden_size // 2, params.hidden_size),
            nn.ReLU(True),
            # nn.LeakyReLU(params.leaky_relu_slope),
            # nn.Linear(params.hidden_size, params.hidden_size),
            # nn.Dropout(params.dropout_rate)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(params.hidden_size, params.hidden_size),
            nn.ReLU(True),
            # nn.LeakyReLU(params.leaky_relu_slope),
            # nn.Linear(params.hidden_size, params.hidden_size//2),
            # nn.Dropout(params.dropout_rate)
        )
        self.hidden3 = nn.Sequential(
            nn.Linear(params.hidden_size, params.hidden_size),
            nn.ReLU(True),
            # nn.LeakyReLU(params.leaky_relu_slope),
            # nn.Linear(params.hidden_size, params.hidden_size//2),
            # nn.Dropout(params.dropout_rate)
        )
        self.out_layer = nn.Sequential(
            nn.Linear(params.hidden_size, params.input_size),
            # nn.ReLU(),
            # nn.Linear(params.hidden_size // 2, params.num_classes),
            # nn.LogSoftmax(dim=1)
            nn.Sigmoid()
            # nn.Softmax(dim=1)
        )

    def forward(self, x):

        out = self.in_layer(x)
        out = self.hidden1(out)
        out = self.hidden2(out)
        out = self.hidden3(out)
        out = self.out_layer(out)

        return out


# Noise
def noise(size):
    n = Variable(torch.rand(size, 10))
    # n = Variable(torch.randint(256, (size, 10)))
    # n = Variable(torch.randn(size, 10))
    if torch.cuda.is_available():
        return n.cuda()
    return n


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
    # vectors = (vectors/255).tolist()
    return vectors


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


