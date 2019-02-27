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

import utils
import model_vae.vae_net as vae_net
import model_vae.one_label_data_loader as data_loader
import display_digit as display_results
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/data-with-grayscale-4000', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/vae_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train(model, optimizer, loss_fn, dataloader, params, epoch, fig):
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

    # set model to training mode
    model.train()

    # # summary for current training loop and a running average object for loss
    # summ = []
    # prop = []
    #
    # loss_avg = utils.WeightedAverage()

    for i, (train_batch, labels_batch) in enumerate(dataloader):

        # move to GPU if available
        if params.cuda:
            train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)
        # convert to torch Variables
        train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

        if labels_batch.size(1) == 1:
            labels_batch = labels_batch.view(labels_batch.size(0))

        # compute model output and loss
        reconstructed_batch, mean, log_of_var = model(train_batch)
        loss = loss_fn(reconstructed_batch, train_batch, mean, log_of_var)

        print('loss:{:.4f}'.format(loss.item()))

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

    # Evaluate summaries only once in a while
    if (epoch + 1) % (0.01 * params.num_epochs) == 0:
        # if i % params.save_summary_steps == 0:

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        train_batch_shaped = vae_net.vectors_to_samples(train_batch)
        reconstructed_batch_shaped = vae_net.vectors_to_samples(reconstructed_batch.detach())  # .data.cpu().numpy()

        display_results.fill_figure(train_batch_shaped, fig, epoch + 1, args.model_dir, 'data')
        display_results.fill_figure(reconstructed_batch_shaped, fig, epoch + 1, args.model_dir, 'reconstructed')

        # proportions_batch = labels_batch.shape[0] / params.batch_size
        # prop.append(proportions_batch)
        #
        # # compute all metrics on this batch
        # summary_batch = {metric: metrics[metric](output_batch, labels_batch)
        #                  for metric in metrics}
        # summary_batch['loss'] = loss.item()
        # summ.append(summary_batch)

        # prop = train_batch.shape[0]/params.batch_size
        # # update the average loss
        # loss_avg.update(loss.item(), prop)

    # # compute mean of all metrics in summary
    # metrics_mean = {metric: np.sum([x[metric] for x in summ] / np.sum(prop)) for metric in summ[0]}
    # metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    # logging.info("- Train metrics: " + metrics_string)
    #
    # # print to screen every 1% of iterations
    # if (epoch+1) % (0.01*params.num_epochs) == 0:
    #     print("train Epoch {}/{}".format(epoch + 1, params.num_epochs))
    #     print(metrics_string)


def train_vae(model, train_dataloader, optimizer, loss_fn, params, model_dir):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
    """

    fig = display_results.create_figure()

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        print("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, params, epoch, fig)

    display_results.close_figure(fig)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # # Set the random seed for reproducible experiments
    # torch.manual_seed(230)
    # if params.cuda:
    #     torch.cuda.manual_seed(230)

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
    model = vae_net.VAENeuralNet(params).cuda() if params.cuda else vae_net.VAENeuralNet(params)

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, betas=(params.beta1, params.beta2))

    # fetch loss function and metrics
    loss_fn = vae_net.loss_fn

    # metrics = net.metrics
    # incorrect = net.incorrect

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_vae(model, train_dl, optimizer, loss_fn, params, args.model_dir)
