"""Train the model"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# from tqdm import tqdm

from utils_gan import Logger
from display_digit import *
import utils
import model_gan.gan_net as gan_net
import model_gan.one_label_data_loader as data_loader
from evaluate import evaluate
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/data-with-grayscale-4000', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/gan_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train_discriminator(d, optimizer, real_data, fake_data):
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = d(real_data)
    # Calculate error and backpropagate
    error_real = loss_fn(prediction_real, gan_net.real_data_target(real_data.size(0)))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = d(fake_data)
    # Calculate error and backpropagate
    error_fake = loss_fn(prediction_fake, gan_net.fake_data_target(real_data.size(0)))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error0
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(d, optimizer, fake_data):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = d(fake_data)
    # Calculate error and backpropagate
    error = loss_fn(prediction, gan_net.real_data_target(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error


def train(d_model, g_model, d_optimizer, g_optimizer, loss_fn, dataloader, params, epoch, fig):  # , axes):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        epoch:
    """

    # # set model to training mode
    # model.train()
    #
    # # summary for current training loop and a running average object for loss
    # summ = []
    # prop = []

    # loss_avg = utils.WeightedAverage()
    logger = Logger(model_name='VGAN', data_name='SCHI')

    # fig = create_figure()
    # fig, axes = create_grid(num_test_samples)

    # Use tqdm for progress bar
    # with tqdm(total=len(dataloader)) as t:
    for i, (real_batch, _) in enumerate(dataloader):
        # for i, (train_batch, labels_batch) in enumerate(dataloader):

        # 1. Train Discriminator
        real_data = Variable(real_batch)
        if torch.cuda.is_available(): real_data = real_data.cuda()
        # Generate fake data
        fake_data = g_model(gan_net.noise(real_data.size(0))).detach()
        # Train D
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_model, d_optimizer, real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = g_model(gan_net.noise(real_batch.size(0)))
        # Train G
        g_error = train_generator(d_model, g_optimizer, fake_data)
        # Log error
        logger.log(d_error, g_error, epoch, i, i)  # num_batches)

    # Display Progress
    if (epoch+1) % (0.01*params.num_epochs) == 0:

        # display.clear_output(True)
        # Display Images
        test_samples = generator(test_noise).data.cpu()
        test_samples = gan_net.vectors_to_samples(test_samples)

        # fig, axes = create_grid(num_test_samples)
        # fill_grid(test_samples, axes)
        fill_figure(test_samples, fig)

        # logger.log_images(test_images, num_test_samples, epoch, i, 17)
        # Display status Logs
        logger.display_status(
            epoch+1, params.num_epochs, i+1, i+1,
            d_error, g_error, d_pred_real, d_pred_fake)


def train_gan(d_model, g_model, train_dataloader, dev_dataloader, d_optimizer, g_optimizer, loss_fn, params, model_dir,
                       restore_file=None):

    # fig, axes = create_grid(num_test_samples)
    plt.ion()
    fig = create_figure()

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        train(d_model, g_model, d_optimizer, g_optimizer, loss_fn, train_dataloader, params, epoch, fig)  # , axes)
    return


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', 'dev'], args.data_dir, params)
    train_dl = dataloaders['train']
    dev_dl = dataloaders['dev']

    logging.info("data was loaded from {}".format(args.data_dir))
    logging.info("- done.")

    # Define the model and optimizer
    discriminator = gan_net.DiscriminatorNet(params)
    generator = gan_net.GeneratorNet(params)
    if torch.cuda.is_available():
        gan_net.discriminator.cuda()
        gan_net.generator.cuda()

    print(discriminator)
    print(generator)

    # Optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=params.learning_rate, betas=(params.beta1, 0.999))
    g_optimizer = optim.Adam(generator.parameters(), lr=params.learning_rate, betas=(params.beta1, 0.999))
    # d_optimizer = optim.SGD(discriminator.parameters(), lr=params.learning_rate)
    # g_optimizer = optim.SGD(generator.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = gan_net.loss_fn

    # metrics = gan_net.metrics
    # incorrect = gan_net.incorrect

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    num_test_samples = 16
    test_noise = gan_net.noise(num_test_samples)
    train_gan(discriminator, generator, train_dl, dev_dl, d_optimizer, g_optimizer, loss_fn, params, args.model_dir,
              args.restore_file)
