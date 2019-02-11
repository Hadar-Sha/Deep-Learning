"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd.variable import Variable


class DiscriminatorNet(nn.Module):
    def __init__(self, params):
        super(DiscriminatorNet, self).__init__()
        # self.is_one_hot = params.is_one_hot
        self.dc_dim = params.dc_dim
        self.cc_dim = params.cc_dim

        self.in_layer = nn.Sequential(
            nn.Linear(params.input_size, params.hidden_size*2),
            # nn.ReLU(),
            nn.LeakyReLU(params.leaky_relu_slope),
            nn.Dropout(params.dropout_rate)
        )
        self.hidden1 = nn.Sequential(
            nn.Linear(params.hidden_size*2, params.hidden_size*2),
            # nn.ReLU(),
            nn.LeakyReLU(params.leaky_relu_slope),
            nn.Dropout(params.dropout_rate)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(params.hidden_size*2, params.hidden_size),
            # nn.ReLU(),
            nn.LeakyReLU(params.leaky_relu_slope),
            nn.Dropout(params.dropout_rate)
        )

        # self.label_emb = nn.Embedding(params.num_classes, params.num_classes)
        #
        # self.fc_insert_label = nn.Sequential(
        #     nn.Linear(params.num_classes, params.num_classes),
        #     # nn.ReLU(),
        #     nn.LeakyReLU(params.leaky_relu_slope),
        #     # nn.Dropout(params.dropout_rate)
        # )

        self.out_layer = nn.Sequential(
            nn.Linear(params.hidden_size, 1 + self.cc_dim + self.dc_dim),
            # nn.ReLU(),
            # nn.Sigmoid()
        )

    def forward(self, x):  # , labels):

        out = self.in_layer(x)
        out = self.hidden1(out)
        out = self.hidden2(out)

        # if not self.is_one_hot:
        #     y_ = self.label_emb(labels)
        # else:
        #     y_ = self.fc_insert_label(labels)

        # out = torch.cat([x_, y_], 1)
        out = self.out_layer(out)

        return out


class GeneratorNet(nn.Module):
    def __init__(self, params):
        super(GeneratorNet, self).__init__()

        self.dc_dim = params.dc_dim
        self.cc_dim = params.cc_dim

        self.input_layer = nn.Sequential(
            nn.Linear(params.noise_dim + self.cc_dim + self.dc_dim, params.hidden_size),
            # nn.ReLU(),
            nn.LeakyReLU(params.leaky_relu_slope),
            nn.Dropout(params.dropout_rate)
        )

        self.hidden1 = nn.Sequential(
            nn.Linear(params.hidden_size, params.hidden_size*2),
            # nn.ReLU(),
            nn.LeakyReLU(params.leaky_relu_slope),
            nn.Dropout(params.dropout_rate)
        )
        self.hidden2 = nn.Sequential(
            nn.Linear(params.hidden_size*2, params.hidden_size*2),
            # nn.ReLU(),
            nn.LeakyReLU(params.leaky_relu_slope),
            nn.Dropout(params.dropout_rate)
        )

        self.out_layer = nn.Sequential(
            nn.Linear(params.hidden_size*2, params.input_size),
            nn.Tanh()
        )

    def forward(self, x):     # def forward(self, x, labels):

        out = self.input_layer(x)
        out = self.hidden1(out)
        out = self.hidden2(out)
        out = self.out_layer(out)

        return out


# Noise
def noise(size, dim):
    n = Variable(torch.randn(size, dim))  # recommended to sample from normal distribution and not from uniform dist
    if torch.cuda.is_available():
        return n.cuda()
    return n


def convert_int_to_one_hot_vector(label, num_of_classes):

    if min(list(label.size())) == 1:
        label_shaped = label.view(-1, 1)

        one_hot_vector = torch.FloatTensor(list(label.size())[0], num_of_classes)
        one_hot_vector.zero_()  # set all values to zero

        one_hot_vector.scatter_(1, label_shaped, 1)
        one_hot_vector = one_hot_vector.type(torch.FloatTensor)
        return one_hot_vector

    else:
        # this is for 3d tensor
        labels_shaped = label.view(label.size(0), label.size(1), -1)

        one_hot_matrix = torch.FloatTensor(list(labels_shaped.size())[0], list(labels_shaped.size())[1], num_of_classes)
        one_hot_matrix.zero_()  # set all values to zero
        one_hot_matrix.scatter_(2, labels_shaped, 1)
        # added to keep a 2d dimension of labels
        one_hot_matrix = one_hot_matrix.view(-1, list(labels_shaped.size())[1]*num_of_classes)
        one_hot_matrix = one_hot_matrix.type(torch.FloatTensor)
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
    if len(labels.shape) > 1 and min(labels.shape) == 1:
        labels = labels.view(labels.size()[0],)

    labels = (labels.numpy()).tolist()
    return labels


def create_random_labels(num_samples, num_classes, mode="One Label"):

    if mode == "One Label":
        test_labels = list(range(num_samples))
        test_labels = [it % num_classes for it in test_labels]
        test_labels = torch.Tensor(test_labels)
        test_labels = test_labels.view(num_samples, -1)
        test_labels = test_labels.type(torch.LongTensor)
        return test_labels

    if mode == "Two Labels":
        all_but_eight = np.concatenate((np.arange(8), np.array([[9]])), axis=None)
        random_mat = np.zeros([num_samples, 2])
        for i in range(num_samples):
            random_mat[i][:] = np.random.choice(all_but_eight, 2, replace=False)

        random_tensor = torch.from_numpy(random_mat)
        random_tensor = random_tensor.type(torch.LongTensor)
        return random_tensor


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class MILoss(nn.Module):
    def __init__(self):
        super(MILoss, self).__init__()

    def forward(self, x, mean, var):
        log_lilelihood = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mean).pow(2).div(var.mul(2.0) + 1e-6)
        normal_dist_negative_log_likelihood = -(log_lilelihood.sum(1).mean())
        return normal_dist_negative_log_likelihood
        # val = -x*torch.exp(x)
        # val = torch.sum(val, dim=1)
        # return torch.mean(val)


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


# InfoGAN Function (Gaussian)
def gen_cc(n_size, dim):
    return torch.Tensor(np.random.randn(n_size, dim) * 0.5 + 0.0)


# InfoGAN Function (Multi-Nomial)
def gen_dc(n_size, dim):
    codes = []
    code = np.zeros((n_size, dim))
    random_cate = np.random.randint(0, dim, n_size)
    code[range(n_size), random_cate] = 1
    codes.append(code)
    codes = np.concatenate(codes,1)
    return torch.Tensor(codes)
