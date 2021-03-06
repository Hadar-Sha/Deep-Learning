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
import torch.nn as nn
# from tqdm import tqdm

# print (sys.path)
# import model
import utils
import model.net as net
import model.data_loader as data_loader
from evaluate import evaluate

# from . import utils
# from .model import data_loader
# from .model import net
# from .evaluate import evaluate
# import model.net as net
# import model.data_loader as data_loader
# from evaluate import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='transfer-learning-data', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/transfer-learning-model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()
    # crit = nn.CrossEntropyLoss(size_average=True, reduce=True)

    # Use tqdm for progress bar
    # with tqdm(total=len(dataloader)) as t:
    for i, (train_batch, labels_batch) in enumerate(dataloader):

        # move to GPU if available
        if params.cuda:
            train_batch, labels_batch = train_batch.cuda(async=True), labels_batch.cuda(async=True)
        # convert to torch Variables
        train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)
        labels_batch = labels_batch.view(labels_batch.size(0))

        # compute model output and loss
        output_batch = model(train_batch)
        loss = loss_fn(output_batch, labels_batch)

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

            # compute all metrics on this batch
            summary_batch = {metric:metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)

        # if i % 100 == 0:
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #     i, i*len(labels_batch), len(dataloader.dataset),
        #     # i, i * len(labels_batch)), len(data_loader.dataset),
        #     100. * i / len(dataloader), loss.item()))

            # print('loss={:05.3f}' ''.format(loss_avg()))

        # update the average loss
        loss_avg.update(loss.item())

        # print('loss={:05.3f}'
        #       ''.format(loss_avg()))
        # t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
        # t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


# def train(ep):
#     net.train()      # set model in training mode (need this because of dropout)
#     for batch_idx, sample_batched in enumerate(train_loader):
#         images = Variable(sample_batched['image'])
#         labels = Variable(sample_batched['label'])
#         labels = labels.view(labels.size(0))
#         # Forward + Backward + Optimize
#         optimizer.zero_grad()
#         outputs = net(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         if ep % 100 == 0 or ep == num_epochs - 1:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 ep, batch_idx * len(images), len(train_loader.dataset),
#                        100. * batch_idx / len(train_loader), loss.item()))


def train_and_evaluate(model, train_dataloader, dev_dataloader, optimizer, loss_fn, metrics, incorrect, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
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
        # if (epoch) % params.save_summary_steps == 0:
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        dev_metrics, incorrect_samples = evaluate(model, loss_fn, dev_dataloader, metrics, incorrect, params)

        dev_acc = dev_metrics['accuracy']
        is_best = dev_acc >= best_dev_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict' : optimizer.state_dict()},
                               is_best=is_best,
                               checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_dev_acc = dev_acc

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

    # print('in main')

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', 'dev'], args.data_dir, params)
    train_dl = dataloaders['train']
    dev_dl = dataloaders['dev']

    logging.info("- done.")

    # Define the model and optimizer
    model = net.NeuralNet(params).cuda() if params.cuda else net.NeuralNet(params)
    # optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    # loss_fn = nn.CrossEntropyLoss(size_average=True, reduce=True)
    loss_fn = net.loss_fn

    metrics = net.metrics
    incorrect = net.incorrect

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, dev_dl, optimizer, loss_fn, metrics, incorrect, params, args.model_dir,
                       args.restore_file)
