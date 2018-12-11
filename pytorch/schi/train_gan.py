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
from display_digit import display_digit
import utils
import model_gan.gan_net as gan_net
import model_gan.one_label_data_loader as data_loader
from evaluate import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/data', help="Directory containing the dataset")
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

    # Return error
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


def train_(model, optimizer, loss_fn, dataloader, metrics, params, epoch):
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

    # summary for current training loop and a running average object for loss
    summ = []
    prop = []

    loss_avg = utils.WeightedAverage()

    # Use tqdm for progress bar
    # with tqdm(total=len(dataloader)) as t:
    for i, (train_batch, labels_batch) in enumerate(dataloader):

        layer_data =[]

        # move to GPU if available
        if params.cuda:
            train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)
        # convert to torch Variables
        train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

        if labels_batch.size(1) == 1:
            labels_batch = labels_batch.view(labels_batch.size(0))

        # compute model output and loss
        output_batch = model(train_batch)
        loss = loss_fn(output_batch, labels_batch, params.num_classes)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        # Evaluate summaries only once in a while
        if i % params.save_summary_steps == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            proportions_batch = labels_batch.shape[0] / params.batch_size
            prop.append(proportions_batch)

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

        prop = train_batch.shape[0]/params.batch_size
        # update the average loss
        loss_avg.update(loss.item(), prop)

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.sum([x[metric] for x in summ] / np.sum(prop)) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

    # print to screen every 1% of iterations
    if (epoch+1) % (0.01*params.num_epochs) == 0:
        print("train Epoch {}/{}".format(epoch + 1, params.num_epochs))
        print(metrics_string)


def train(d_model, g_model, d_optimizer, g_optimizer, loss_fn, dataloader, params, epoch):
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

    # Use tqdm for progress bar
    # with tqdm(total=len(dataloader)) as t:
    for i, (real_batch, _) in enumerate(dataloader):
        # for i, (train_batch, labels_batch) in enumerate(dataloader):

        # 1. Train Discriminator
        real_data = Variable(real_batch)
        if torch.cuda.is_available(): real_data = real_data.cuda()
        # Generate fake data
        # print(real_data.size())
        fake_data = g_model(gan_net.noise(real_data.size(0))).detach()
        # Train D
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_model, d_optimizer, real_data, fake_data)

        # 2. Train Generator
        # Generate fake data
        fake_data = g_model(gan_net.noise(real_batch.size(0)))
        # Train G
        g_error = train_generator(d_model, g_optimizer, fake_data)
        # Log error
        logger.log(d_error, g_error, epoch, i, 17)  # num_batches)

    # Display Progress
    if (epoch+1) % (0.01*params.num_epochs) == 0:
        # print(i)
        # print(epoch+1)
        # if isinstance(d_error, torch.autograd.Variable):
        #     d_error = d_error.data.cpu().numpy()
        # if isinstance(g_error, torch.autograd.Variable):
        #     g_error = g_error.data.cpu().numpy()
        # print("d_error is : {}".format(d_error))
        # print("g_error is : {}".format(g_error))
        # print(d_pred_real)
        # print(d_pred_fake)

        # display.clear_output(True)
        # Display Images
        test_images = generator(test_noise).numpy()
        # test_images = display_digit(generator(test_noise)).data.cpu()
        # # test_images = vectors_to_images(generator(test_noise)).data.cpu()
        # logger.log_images(test_images, num_test_samples, epoch, i, 17)
        # Display status Logs
        logger.display_status(
            epoch+1, params.num_epochs, i+1, 17,
            d_error, g_error, d_pred_real, d_pred_fake)

    # # move to GPU if available
        # if params.cuda:
        #     train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)
        # # convert to torch Variables
        # train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
        #
        # if labels_batch.size(1) == 1:
        #     labels_batch = labels_batch.view(labels_batch.size(0))

        # # compute model output and loss
        # output_batch = model(train_batch)
        # loss = loss_fn(output_batch, labels_batch, params.num_classes)
        #
        # # clear previous gradients, compute gradients of all variables wrt loss
        # optimizer.zero_grad()
        # loss.backward()
        #
        # # performs updates using calculated gradients
        # optimizer.step()

    #     # Evaluate summaries only once in a while
    #     if i % params.save_summary_steps == 0:
    #         # extract data from torch Variable, move to cpu, convert to numpy arrays
    #         output_batch = output_batch.data.cpu().numpy()
    #         labels_batch = labels_batch.data.cpu().numpy()
    #
    #         proportions_batch = labels_batch.shape[0] / params.batch_size
    #         prop.append(proportions_batch)
    #
    #         # compute all metrics on this batch
    #         summary_batch = {metric: metrics[metric](output_batch, labels_batch)
    #                          for metric in metrics}
    #         summary_batch['loss'] = loss.item()
    #         summ.append(summary_batch)
    #
    #     prop = train_batch.shape[0]/params.batch_size
    #     # update the average loss
    #     loss_avg.update(loss.item(), prop)
    #
    # # compute mean of all metrics in summary
    # metrics_mean = {metric: np.sum([x[metric] for x in summ] / np.sum(prop)) for metric in summ[0]}
    # metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    # logging.info("- Train metrics: " + metrics_string)
    #
    # # print to screen every 1% of iterations
    # if (epoch+1) % (0.01*params.num_epochs) == 0:
    #     print("train Epoch {}/{}".format(epoch + 1, params.num_epochs))
    #     print(metrics_string)


# train_gan(discriminator, generator, train_dl, dev_dl, d_optimizer, g_optimizer,
# loss_fn, params, args.model_dir,args.restore_file)
def train_gan(d_model, g_model, train_dataloader, dev_dataloader, d_optimizer, g_optimizer, loss_fn, params, model_dir,
                       restore_file=None):
    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        train(d_model, g_model, d_optimizer, g_optimizer, loss_fn, train_dataloader, params, epoch)
    return


def train_and_evaluate(model, train_dataloader, dev_dataloader, optimizer, loss_fn, metrics, incorrect, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        dev_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        incorrect: a function that save all samples with incorrect classification
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_dev_acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params, epoch)

        # Evaluate for one epoch on validation set
        dev_metrics, incorrect_samples = evaluate(model, loss_fn, dev_dataloader, metrics, incorrect, params, epoch)

        dev_acc = dev_metrics['accuracy']
        is_best = dev_acc >= best_dev_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()}, is_best=is_best, checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            print("Epoch {}/{}".format(epoch + 1, params.num_epochs))
            print("- Found new best accuracy")
            best_dev_acc = dev_acc
            print("accuracy is {:05.3f}".format(best_dev_acc))

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_dev_best_weights.json")
            utils.save_dict_to_json(dev_metrics, best_json_path)

            best_csv_path = os.path.join(model_dir, "incorrect_best_samples.csv")
            utils.save_incorrect_to_csv(incorrect_samples, best_csv_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_dev_last_weights.json")
        utils.save_dict_to_json(dev_metrics, last_json_path)

        last_csv_path = os.path.join(model_dir, "incorrect_last_samples.csv")
        utils.save_incorrect_to_csv(incorrect_samples, last_csv_path)


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
    d_optimizer = optim.Adam(discriminator.parameters(), lr=params.learning_rate)
    # d_optimizer = optim.Adam(discriminator.parameters(), lr=params.learning_rate)
    g_optimizer = optim.Adam(generator.parameters(), lr=params.learning_rate)
    # g_optimizer = optim.Adam(generator.parameters(), lr=params.learning_rate)

    # model = net.NeuralNet(params).cuda() if params.cuda else net.NeuralNet(params)
    #
    # print(model)
    #
    # optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = gan_net.loss_fn

    # metrics = gan_net.metrics
    # incorrect = gan_net.incorrect

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    num_test_samples = 16
    test_noise = gan_net.noise(num_test_samples)
    train_gan(discriminator, generator, train_dl, dev_dl, d_optimizer, g_optimizer, loss_fn, params, args.model_dir,args.restore_file)
    # train_and_evaluate(model, train_dl, dev_dl, optimizer, loss_fn, metrics, incorrect, params, args.model_dir,
    #                    args.restore_file)
