"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class VAENeuralNet(nn.Module):
    def __init__(self, params):
        super(VAENeuralNet, self).__init__()
        self.fc11 = nn.Linear(params.input_size, params.input_size)  # , bias=False)
        self.fc12 = nn.Linear(params.input_size, params.input_size)  # , bias=False)

        # self.fc12 = nn.Linear(params.input_size, params.output_size, bias=False)
        # self.fc13 = nn.Linear(params.input_size, params.output_size, bias=False)
        # self.fc21 = nn.Linear(params.hidden_size, params.output_size)
        # self.fc22 = nn.Linear(params.hidden_size, params.output_size)
        self.fc3 = nn.Linear(params.input_size, params.input_size)  # , bias=False)
        # self.fc4 = nn.Linear(params.hidden_size, params.input_size)
        self.low_bound = 0
        self.high_bound = 255

    def encode(self, x):

        return F.relu(self.fc11(x)), F.relu(self.fc12(x))
        # h1 = F.relu(self.fc1(x))
        # return self.fc21(h1), self.fc22(h1)
        # return F.relu(self.fc11(x)), F.relu(self.fc12(x)), F.relu(self.fc13(x))

    def reparameterize(self, mu, logvar):  # r, g, b):  # mu, logvar):
        # return torch.randn_like(sample)
        # color = torch.cat((r, g, b), 1)
        # out_color = torch.randint_like(color, self.low_bound, self.high_bound)
        # return out_color.div(self.high_bound)
        # return color
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):  # r, g, b):  # z):
        return F.relu6(self.fc3(z))/6
        # h3 = z.view(-1, z.size(0)*z.size(1))
        # return F.relu(self.fc3(h3))    # how to make sure output is [0,1]?
        # h3 = F.relu(self.fc3(z))
        # return torch.sigmoid(self.fc4(h3))

    def forward(self, x):

        # sample = self.encode(x)
        # z = self.reparameterize(sample)
        # return self.decode(z)
        # r, g, b = self.encode(x)
        # z = self.reparameterize(r, g, b)
        # return self.decode(z), r, g, b
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

        return out

    def inference(self, output_size, n=1):

        batch_size = n
        # z = torch.randint(self.low_bound, self.high_bound, [batch_size, output_size])
        z = torch.randn([batch_size, output_size])
        if torch.cuda.is_available():
            z = z.cuda()

        recon_x = self.decode(z)

        return recon_x

    def generation_with_interpolation(self, x_one, x_two, alpha):
        hidden_one = self.encode(x_one)
        hidden_two = self.encode(x_two)
        # r_one = hidden_one[0]
        # g_one = hidden_one[1]
        # b_one = hidden_one[2]
        # r_two = hidden_two[0]
        # g_two = hidden_two[1]
        # b_two = hidden_two[2]
        # r = (1 - alpha) * r_one + alpha * r_two
        # g = (1 - alpha) * g_one + alpha * g_two
        # b = (1 - alpha) * b_one + alpha * b_two
        # z = self.reparameterize(r, g, b)
        mu_one = hidden_one[0]
        logvar_one = hidden_one[1]
        mu_two = hidden_two[0]
        logvar_two = hidden_two[1]
        mu = (1 - alpha) * mu_one + alpha * mu_two
        logvar = (1 - alpha) * logvar_one + alpha * logvar_two
        z = self.reparameterize(mu, logvar)
        generated_image = self.decode(z)
        return generated_image


def default_w_init(self):
    stdv = 1. / math.sqrt(self.weight.size(1))
    self.weight.data.uniform_(-stdv, stdv)
    if self.bias is not None:
        self.bias.data.uniform_(-stdv, stdv)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        stdv = max(0.002, 1. / math.sqrt(m.weight.size(1)))
        nn.init.uniform_(m.weight.data, -stdv, stdv)
        if m.bias is not None:
            nn.init.uniform_(m.bias.data, -stdv, stdv)
        # nn.init.normal_(m.weight.data, 0.0, 0.1)
        # nn.init.constant_(m.bias.data, 0)
        # nn.init.normal_(m.weight.data, 0.0, 0.1)
        # nn.init.constant_(m.bias.data, 0)

    if torch.cuda.is_available():
        for pa in m.parameters():
            pa.cuda()


def loss_fn(input_d, reconstructed, mean, logvar):
    """
    Compute the VAE loss given outputs and labels.

    Args:
        input_d:
        reconstructed:
        mean:
        logvar:

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """

    bce_criterion = nn.BCELoss(size_average=False)  # reduction=sum ?
    # print(reconstructed)
    bce_loss = bce_criterion(input_d, reconstructed)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    # for gaussian distribution when
    # generated data passed to the encorder is z~ N(0,1) and generated data is x~N(m,var)

    # kl_criterion = nn.KLDivLoss(size_average=False)
    # kl_loss = kl_criterion(input_d, )
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    scaled_kl_loss = 0.001*kl_loss

    return bce_loss + kl_loss, bce_loss, kl_loss
    # return bce_loss + scaled_kl_loss, bce_loss, kl_loss


def vectors_to_samples(vectors):
    vectors = vectors.reshape(vectors.size()[0], -1, 3)
    vectors = vectors.cpu().numpy()
    vectors = vectors.tolist()
    return vectors


def labels_to_titles(labels):
    if len(labels.shape) > 1 and min(labels.shape) == 1:
        labels = labels.view(labels.size()[0],)
    labels = (labels.cpu().numpy()).tolist()
    return labels

