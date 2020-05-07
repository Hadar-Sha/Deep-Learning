"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
# from pytorchtools import EarlyStopping
import utils
import model.net as net
import model.one_label_data_loader as data_loader
# import model_weighted_schi_distance.net as net
# import model_weighted_schi_distance.one_label_data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default="toSync/Thesis/DL-Pytorch-data", help='path to experiments and data folder. not for Server')
parser.add_argument('--data_dir', default='data/with-grayscale/data-with-grayscale-4000', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model_debug', help="Directory containing params.json")   # default='experiments/transfer_training_model/in-grayscale-4000'
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")   # default='fully_connected_best'
parser.add_argument('--is_dev', type=bool, default=False, help="")


def evaluate(model, loss_fn, dataloader, metrics, incorrect, correct_fn, params, epoch):
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

    # print(model.training)
    # set model to evaluation mode
    model.eval()

    # print(model.training)

    # summary for current eval loop
    summ = []
    prop = []

    # incorrect samples of current loop
    incorrect_samples = []
    correct_res_samples = []
    # need_to_stop = False

    # early_stopping = EarlyStopping(patience=(0.01 * params.num_epochs), verbose=True)

    # compute metrics over the dataset
    for data_batch, labels_batch in dataloader:

        # move to GPU if available
        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
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

        correct_res_batch = correct_fn(data_batch, output_batch, labels_batch)
        correct_res_samples.extend(correct_res_batch)

    # compute mean of all metrics in summary
    prop_sum = np.sum(prop)
    metrics_mean = {metric: np.sum([x[metric] for x in summ]/prop_sum) for metric in summ[0]}

    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    # print to screen every 1% of iterations
    if (epoch+1) % (0.01*params.num_epochs) == 0:
        print("eval Epoch {}/{}".format(epoch + 1, params.num_epochs))
        print(metrics_string)
        logging.info("eval Epoch {}/{}".format(epoch + 1, params.num_epochs))
        logging.info(metrics_string)

    return metrics_mean, incorrect_samples, correct_res_samples  # , need_to_stop


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    if args.parent_dir and not torch.cuda.is_available():
        past_to_drive = os.environ['OneDrive']
        os.chdir(os.path.join(past_to_drive, args.parent_dir))

    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()     # use GPU is available

    # # Set the random seed for reproducible experiments
    # torch.manual_seed(230)
    # if params.cuda: torch.cuda.manual_seed(230)

    if args.is_dev is True:
        logger_type = "dev"
    else:
        logger_type = "test"

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, "evaluate_" + logger_type + ".log"))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test', 'dev'], args.data_dir, params)
    test_dl = dataloaders['test']
    dev_dl = dataloaders['dev']

    if args.is_dev is True:
        chosen_dl = dev_dl
    else:
        chosen_dl = test_dl

    logging.info("data was loaded from {}".format(args.data_dir))
    logging.info("- done.")

    num_of_batches = max(1, len(chosen_dl.dataset) // chosen_dl.batch_size)
    logging.info("data-set size: {}".format(len(chosen_dl.dataset)))
    logging.info("data-set type: " + logger_type)
    # num_of_batches = max(1, len(test_dl.dataset) // test_dl.batch_size)
    # logging.info("data-set size: {}".format(len(test_dl.dataset)))
    logging.info("number of batches: {}".format(num_of_batches))

    # logging.info("- done.")

    # Define the model
    model = net.NeuralNet(params).cuda() if params.cuda else net.NeuralNet(params)

    # model.eval()  # important for dropout not to work in forward pass
    
    loss_fn = net.loss_fn
    metrics = net.metrics
    incorrect = net.incorrect
    correct_fn = net.correct_classification
    
    logging.info("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics, incorrect_samples, correct_res_samples = evaluate(model, loss_fn, chosen_dl, metrics,
                                                                    incorrect, correct_fn, params, params.num_epochs-1)
    save_path = os.path.join(args.model_dir, "evaluate_metrics_" + logger_type + "_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
    # print(save_path)

    best_inc_csv_path = os.path.join(args.model_dir, "evaluate_" + logger_type + "_incorrect_samples.csv")
    utils.save_incorrect_to_csv(incorrect_samples, best_inc_csv_path)
    # print(best_inc_csv_path)

    best_c_csv_path = os.path.join(args.model_dir, "evaluate_" + logger_type + "_correct_samples.csv")
    utils.save_incorrect_to_csv(correct_res_samples, best_c_csv_path)
    # print(best_c_csv_path)

