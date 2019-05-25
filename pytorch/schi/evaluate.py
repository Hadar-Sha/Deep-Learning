"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.net as net
import model.one_label_data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, loss_fn, dataloader, metrics, incorrect, params, epoch):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        incorrect: a function that save all samples with incorrect classification
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
        epoch:
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []
    prop = []

    # incorrect samples of current loop
    incorrect_samples = []

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(async=True), labels_batch.cuda(async=True)
        # fetch the next evaluation batch
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        if labels_batch.size(1) == 1:
            labels_batch = labels_batch.view(labels_batch.size(0))
        
        # compute model output
        output_batch = model(data_batch)
        loss = loss_fn(output_batch, labels_batch, params.num_classes)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        proportions_batch = labels_batch.shape[0] / params.batch_size
        prop.append(proportions_batch)

        # compute all metrics on this batch
        summary_batch = {metric: metrics[metric](output_batch, labels_batch)*proportions_batch
                         for metric in metrics}
        summary_batch['loss'] = loss.item()
        summ.append(summary_batch)

        # find incorrect samples
        incorrect_batch = incorrect(data_batch, output_batch, labels_batch)
        incorrect_samples.extend(incorrect_batch)

    # compute mean of all metrics in summary
    prop_sum = np.sum(prop)
    metrics_mean = {metric: np.sum([x[metric] for x in summ]/prop_sum) for metric in summ[0]}

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    # print to screen every 1% of iterations
    if (epoch+1) % (0.01*params.num_epochs) == 0:
        print("eval Epoch {}/{}".format(epoch + 1, params.num_epochs))
        print(metrics_string)

    return metrics_mean, incorrect_samples


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda: torch.cuda.manual_seed(230)
        
    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = net.NeuralNet(params).cuda() if params.cuda else net.NeuralNet(params)
    
    loss_fn = net.loss_fn
    metrics = net.metrics
    incorrect = net.incorrect
    
    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics, incorrect_samples = evaluate(model, loss_fn, test_dl, metrics, incorrect, params, params.num_epochs)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)

