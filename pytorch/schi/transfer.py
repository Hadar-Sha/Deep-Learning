import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import csv

import utils
import model.net as net
import model.two_labels_data_loader as two_labels_data_loader
from evaluate import evaluate
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/data-two-labels', help="Directory containing the destination dataset")
parser.add_argument('--model_dir', default='experiments/transfer_training_model/in-grayscale',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best',
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
# added by me
parser.add_argument('--model_out_dir', default='experiments/transfer_training_model/out-grayscale',
                    help="Directory containing params.json")


def load_model(model_dir, restore_file):
    # reload weights from restore_file if specified
    if restore_file is not None and model_dir is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, None)  # optimizer)
    return


def after_transfer_train_and_evaluate(model, train_dataloader, dev_dataloader, optimizer, loss_fn, metrics, incorrect, params, model_dir, model_out_dir, restore_file):

    best_dev_acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params, epoch)

        # Evaluate for one epoch on validation set
        dev_metrics, incorrect_samples = evaluate(model, loss_fn, dev_dataloader, metrics, incorrect, params, epoch)

        dev_acc = dev_metrics['accuracy_two_labels']
        is_best = dev_acc >= best_dev_acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                              is_best=is_best,
                              checkpoint=model_out_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            print("Epoch {}/{}".format(epoch + 1, params.num_epochs))
            print("- Found new best accuracy")
            best_dev_acc = dev_acc
            print("accuracy is {:05.3f}".format(best_dev_acc))
            print("loss is {:05.3f}".format(dev_metrics['loss']))

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_out_dir, "metrics_dev_best_weights.json")
            utils.save_dict_to_json(dev_metrics, best_json_path)

            best_csv_path = os.path.join(model_out_dir, "incorrect_best_samples.csv")
            utils.save_incorrect_to_csv(incorrect_samples, best_csv_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_out_dir, "metrics_dev_last_weights.json")
        utils.save_dict_to_json(dev_metrics, last_json_path)

        last_csv_path = os.path.join(model_out_dir, "incorrect_last_samples.csv")
        utils.save_incorrect_to_csv(incorrect_samples, last_csv_path)
    return


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_out_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Set the logger # for output model
    utils.set_logger(os.path.join(args.model_out_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = two_labels_data_loader.fetch_dataloader(['train', 'dev'], args.data_dir, params)
    train_dl = dataloaders['train']
    dev_dl = dataloaders['dev']

    logging.info("data was loaded from {}".format(args.data_dir))
    logging.info("- done.")

    # Define the model and optimizer
    model = net.NeuralNet(params).cuda() if params.cuda else net.NeuralNet(params)

    load_model(args.model_dir, args.restore_file)

    status_before_transfer = []
    for param_tensor in model.state_dict():
        status_before_transfer.append([param_tensor,
                   (model.state_dict()[param_tensor].norm()).item(), list(model.state_dict()[param_tensor].size())])
        status_before_transfer.append(((model.state_dict()[param_tensor]).numpy()).tolist())

    # changing last fully connected layer
    num_ftrs = model.fc4.in_features
    model.fc4 = nn.Linear(num_ftrs, 20)  # 10)

    model = model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = net.loss_fn_two_labels

    metrics = net.metrics
    incorrect = net.incorrect_two_labels

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    after_transfer_train_and_evaluate(model, train_dl, dev_dl, optimizer, loss_fn, metrics, incorrect, params,
                                      args.model_dir, args.model_out_dir, args.restore_file)

    load_model(args.model_out_dir, args.restore_file)

    status_after_transfer = []
    for param_tensor in model.state_dict():
        status_after_transfer.append([param_tensor,
             (model.state_dict()[param_tensor].norm()).item(), list(model.state_dict()[param_tensor].size())])
        status_after_transfer.append(((model.state_dict()[param_tensor]).numpy()).tolist())

    filepath = os.path.join(args.model_out_dir, 'wb_ext.csv')
    with open(filepath, "a", newline='') as myfile:
        csvwr = csv.writer(myfile)
        for elem in status_before_transfer:
            if isinstance(elem, (list,)) and isinstance(elem[0], (list,)):
                for row in elem:
                    csvwr.writerow(row)
            else:
                csvwr.writerow(elem)
        for elem_a in status_after_transfer:
            if isinstance(elem, (list,)) and isinstance(elem[0], (list,)):
                for row in elem:
                    csvwr.writerow(row)
            else:
                csvwr.writerow(elem)
