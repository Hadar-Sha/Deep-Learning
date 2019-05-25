"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from scipy import stats


class VAENeuralNet(nn.Module):
    def __init__(self, params):
        super(VAENeuralNet, self).__init__()
        self.fc1 = nn.Linear(params.input_size, params.hidden_size)
        self.activeFuncEncoder = nn.ReLU(True)  # nn.ReLU6(True)  nn.Tanh()
        self.fc21 = nn.Linear(params.hidden_size, params.output_size)
        self.fc22 = nn.Linear(params.hidden_size, params.output_size)
        self.fc3 = nn.Linear(params.output_size, params.hidden_size)  # , bias=False)
        self.activeFuncDecoder = nn.ReLU(True)
        self.fc4 = nn.Linear(params.hidden_size, params.input_size)
        self.fc42 = nn.Linear(params.hidden_size, params.input_size)
        self.normFuncDecoder = nn.Sigmoid()

    def encode(self, x):

        # h1 = F.tanh(self.fc1(x))
        h1 = self.activeFuncEncoder(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = self.activeFuncDecoder(self.fc3(z))
        return self.normFuncDecoder(self.fc4(h3))

    def uniform_decode(self, z):
        h3 = self.activeFuncDecoder(self.fc3(z))
        # h3 = F.tanh(self.fc3(z))
        # return F.relu6(self.fc4(h3)), F.relu6(self.fc42(h3))
        return self.activeFuncDecoder(self.fc4(h3)), self.activeFuncDecoder(self.fc42(h3))

    def uniform_reparameterize(self, uni_param_a, uni_param_b):
        mult = uni_param_b - uni_param_a
        base = torch.rand_like(mult)
        return F.relu(base.mul(mult).add_(uni_param_a))  # maybe relu here?
        # return base.mul(mult).add_(uni_param_a)  # maybe relu here?
        # return torch.round(base.mul(mult).add_(uni_param_a))

    @staticmethod
    def min_max_normalization(val, min_value, max_value):
        min_val = val.min()
        val = (val - min_val)
        max_val = val.max()
        val = val / max_val
        val = val * (max_value - min_value) + min_value
        return val

    # def forward(self, x):
    #
    #     mu, logvar = self.encode(x)
    #     z = self.reparameterize(mu, logvar)
    #     uni_a, uni_b = self.uniform_decode(z)
    #     recon = self.uniform_reparameterize(uni_a, uni_b)
    #     return recon, mu, logvar, uni_a, uni_b

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    # return out

    def inference(self, output_size, n=1):

        batch_size = n
        z = torch.randn([batch_size, output_size])
        if torch.cuda.is_available():
            z = z.cuda()

        uni_a, uni_b = self.uniform_decode(z)
        recon_x = self.uniform_reparameterize(uni_a, uni_b)
        # recon_x = self.decode(z)

        return recon_x

    def generation_with_interpolation(self, x_one, x_two, alpha):
        hidden_one = self.encode(x_one)
        hidden_two = self.encode(x_two)
        mu_one = hidden_one[0]
        logvar_one = hidden_one[1]
        mu_two = hidden_two[0]
        logvar_two = hidden_two[1]
        mu = (1 - alpha) * mu_one + alpha * mu_two
        logvar = (1 - alpha) * logvar_one + alpha * logvar_two
        z = self.reparameterize(mu, logvar)
        uni_a, uni_b = self.uniform_decode(z)
        generated_image = self.uniform_reparameterize(uni_a, uni_b)
        # generated_image = self.decode(z)
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
    # if m is m.fc4 or m is

        # nn.init.normal_(m.weight.data, 0.0, 0.1)
        # nn.init.constant_(m.bias.data, 0)
        # nn.init.normal_(m.weight.data, 0.0, 0.1)
        # nn.init.constant_(m.bias.data, 0)

    if torch.cuda.is_available():
        for pa in m.parameters():
            pa.cuda()


def uniform_loss_fn(uniform_param_a, uniform_param_b, mean, logvar, beta=1, batch_size=1, input_size=1):

    log_diff = torch.log(uniform_param_b - uniform_param_a)
    # log_diff[log_diff == float("-Inf")] = 0
    log_diff[log_diff != log_diff] = 0

    uniform_loss = - torch.sum(log_diff)

    normalized_uniform_loss = uniform_loss / (batch_size * input_size)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    # for gaussian distribution when
    # generated data passed to the encorder is z~ N(0,1) and generated data is x~N(m,var)

    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    # print(logvar.min())
    # print(logvar.max())
    # # print(mean.min())
    # print(mean.max())

    normalized_kl_loss = kl_loss / (batch_size * input_size)
    return normalized_uniform_loss + normalized_kl_loss, normalized_uniform_loss, normalized_kl_loss


def loss_fn(input_d, reconstructed, mean, logvar, beta=1, batch_size=1, input_size=1):
    """
    Compute the VAE loss given outputs and labels.

    Args:
        input_d:
        reconstructed:
        mean:
        logvar:
        beta:

    Returns:
        loss (Variable): cross entropy loss for all images in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """

    # mse_criterion = nn.MSELoss()  # reduction=sum ?
    # mse_loss = mse_criterion(input_d, reconstructed)

    # bce_criterion = nn.BCELoss(size_average=False)  # reduction=sum ?
    bce_criterion = nn.BCELoss()  # reduction=sum ?
    bce_loss = bce_criterion(input_d, reconstructed)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    # for gaussian distribution when
    # generated data passed to the encorder is z~ N(0,1) and generated data is x~N(m,var)

    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

    normalized_kl_loss = kl_loss / (batch_size * input_size)
    scaled_kl_loss = beta*normalized_kl_loss
    # scaled_kl_loss = beta*kl_loss

    # return bce_loss + kl_loss, bce_loss, kl_loss
    return bce_loss + scaled_kl_loss, bce_loss, normalized_kl_loss
    # return mse_loss + scaled_kl_loss, mse_loss, kl_loss


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

