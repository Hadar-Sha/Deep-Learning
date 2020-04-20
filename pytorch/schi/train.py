"""Train the model"""

import argparse
import logging
import os
import re
import sys

import numpy as np
import torch

from torch.autograd import Variable
from pytorchtools import EarlyStopping

# from tqdm import tqdm

import utils
# import model_weighted_schi_distance.net as net
# import model_weighted_schi_distance.one_label_data_loader as data_loader
import model.net as net
import model.one_label_data_loader as data_loader
from evaluate import evaluate
import display_digit as display_results

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default="C:/Users/H/Documents/Haifa Univ/Thesis/DL-Pytorch-data", help='path to experiments and data folder. not for Server')
parser.add_argument('--data_dir', default='data/color-syn-one-color-big', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model_weighted_schi_dist/syn-color/three_layers/debug', help="Directory containing params.json")
parser.add_argument('--early_stop', type=int, default=0, help="Optional, do early stop")  # action='store_true'
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train(model, optimizer, loss_fn, dataloader, metrics, params, epoch, fig):
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
    num_of_batches = max(1, len(dataloader.dataset) // dataloader.batch_size)

    loss_avg = utils.WeightedAverage()

    # Use tqdm for progress bar
    # with tqdm(total=len(dataloader)) as t:
    for i, (train_batch, labels_batch) in enumerate(dataloader):

        # layer_data = []

        # move to GPU if available
        if params.cuda:
            train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
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

        prop_for_loss = train_batch.shape[0] / params.batch_size
        # update the average loss
        loss_avg.update(loss.item(), prop_for_loss)

        # # compute all metrics on this batch
        # summary_batch = {'loss': loss.item()}
        # # summary_batch['loss'] = loss.item()
        # summ.append(summary_batch)
        # # print(summ)
        #
        # proportions_batch = labels_batch.shape[0] / params.batch_size
        # prop.append(proportions_batch)

        # Evaluate summaries only once in a while
        # if (epoch + 1) % (0.01 * params.num_epochs) == 0:
        if i % params.save_summary_steps == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            if num_of_batches > 1:
                proportions_batch = labels_batch.shape[0] / params.batch_size
            else:
                proportions_batch = 1
            prop.append(proportions_batch)

            # compute all metrics on this batch
            summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                             for metric in metrics}
            summary_batch['loss'] = loss.item()
            summ.append(summary_batch)
            # print(summ)

        if ((i + 1) % max(1, round(0.5*num_of_batches)) == 0) and (epoch == 0):
        # if ((i + 1) % max(1, round(0.1*num_of_batches)) == 0) and (epoch == 0):
            # Display data Images
            real_samples_reshaped = net.vectors_to_samples(train_batch)  # ?
            real_titles = net.labels_to_titles(labels_batch)

            print('plotting batch #{} of input data'.format(i+1))
            display_results.fill_figure(real_samples_reshaped, fig, i + 1, args.model_dir, -1, 1, labels=real_titles,
                                        dtype='real')

    # compute mean of all metrics in summary
    # print(summ)
    metrics_mean = {metric: np.sum([x[metric] for x in summ] / np.sum(prop)) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

    # print to screen every 1% of iterations
    if (epoch+1) % (0.01*params.num_epochs) == 0:
        print("train Epoch {}/{}".format(epoch + 1, params.num_epochs))
        print(metrics_string)

    # compute mean of all metrics in summary (loss, bce part, kl part)
    if isinstance(loss, torch.autograd.Variable):
        loss_v = loss.data.cpu().numpy()
    losses.append(loss_v.item())


def train_and_evaluate(model, train_dataloader, dev_dataloader, optimizer, loss_fn, metrics, incorrect, correct_fn, params, model_dir,
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

    if args.early_stop:
        early_stopping = EarlyStopping(patience=round(0.1 * params.num_epochs), verbose=False)
        # early_stopping = EarlyStopping(patience=round(0.01 * params.num_epochs), verbose=False)

    fig = display_results.create_figure()

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params, epoch, fig)

        # Evaluate for one epoch on validation set
        dev_metrics, incorrect_samples, correct_samples = evaluate(model, loss_fn, dev_dataloader, metrics, incorrect, correct_fn, params, epoch)

        dev_loss = dev_metrics['loss']
        if args.early_stop:
            early_stopping(dev_loss, model)

        if args.early_stop and early_stopping.early_stop:
            # need_to_stop = True
            print("Early stopping")
            logging.info("Early stopping")
            break

        # grads_graph = collect_network_statistics(model)
        # grads_per_epoch.append(grads_graph)

        dev_acc = dev_metrics['accuracy']
        is_best = dev_acc > best_dev_acc

        grads_graph, _ = get_network_grads(model)
        vals_graph = collect_network_statistics(model)

        grads_per_epoch.append(grads_graph)
        vals_per_epoch.append(vals_graph)

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
            utils.save_dict_to_json(dev_metrics, best_json_path, epoch + 1)

            best_inc_csv_path = os.path.join(model_dir, "incorrect_best_samples.csv")
            utils.save_incorrect_to_csv(incorrect_samples, best_inc_csv_path)

            best_c_csv_path = os.path.join(model_dir, "correct_best_samples.csv")
            utils.save_incorrect_to_csv(correct_samples, best_c_csv_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_dir, "metrics_dev_last_weights.json")
        utils.save_dict_to_json(dev_metrics, last_json_path, epoch + 1)

        last_inc_csv_path = os.path.join(model_dir, "incorrect_last_samples.csv")
        utils.save_incorrect_to_csv(incorrect_samples, last_inc_csv_path)

        last_c_csv_path = os.path.join(model_dir, "correct_last_samples.csv")
        utils.save_incorrect_to_csv(correct_samples, last_c_csv_path)

        # compute mean of all metrics in summary (loss, bce part, kl part)
        accuracy.append(dev_acc)
        # if isinstance(loss, torch.autograd.Variable):
        #     loss_v = loss.data.cpu().numpy()
    display_results.close_figure(fig)
    return


def get_network_grads(net):
    weight_string = "weight"
    bias_string = "bias"
    output_gradients = []
    output_names = []

    parameters_names = list(net.state_dict().keys())
    j = 0
    for i in range(len(parameters_names)):
        par = parameters_names[i - j]
        is_rel_w = re.search(weight_string, par)
        is_rel_b = re.search(bias_string, par)
        if is_rel_w is None and is_rel_b is None:
            parameters_names.remove(par)
            j += 1

    for name, param in net.named_parameters():
        if name in parameters_names:
            all_net_grads = param.grad.data.cpu().numpy().tolist()
            flat_net_grads = []
            if isinstance(all_net_grads, (list,)):

                for elem in all_net_grads:
                    if isinstance(elem, (list,)) and isinstance(elem[0], (list,)):
                        for item in elem:
                            flat_net_grads.extend(item)
                    elif isinstance(elem, (list,)):
                        flat_net_grads.extend(elem)
                    else:
                        flat_net_grads.extend([elem])
            else:
                flat_net_grads = all_net_grads

            output_gradients.append([min(flat_net_grads), np.median(flat_net_grads), max(flat_net_grads)])
            output_names.append(name)

    return output_gradients, output_names


def collect_network_statistics(net):

    status_net = []
    net_grads_graph = []

    for param_tensor in net.state_dict():

        status_net.append([param_tensor,
                                       (net.state_dict()[param_tensor].norm()).item(),
                                       list(net.state_dict()[param_tensor].size())])

        all_net_grads = ((net.state_dict()[param_tensor]).cpu().numpy()).tolist()

        # if needed, flatten the list to get one nim and one max
        flat_net_grads = []
        if isinstance(all_net_grads, (list,)):

            for elem in all_net_grads:
                if isinstance(elem, (list,)) and isinstance(elem[0], (list,)):
                    for item in elem:
                        flat_net_grads.extend(item)
                elif isinstance(elem, (list,)):
                    flat_net_grads.extend(elem)
                else:
                    flat_net_grads.extend([elem])
        else:
            flat_net_grads = all_net_grads

        net_grads_graph.append([min(flat_net_grads), max(flat_net_grads)])

    return net_grads_graph


def plot_summary_graphs_layers(vals_to_plot, v_type, im_path):
    vals_np = np.array(vals_to_plot)
    for it in range(vals_np.shape[1]):
        val = vals_np[:, it].tolist()
        layer_indx = it // 2
        if it % 2:  # odd row numbers store bias vals
            display_results.plot_graph(val, None, "{}_layer_bias_{}".format(v_type, layer_indx), im_path)
            print('{}_layer_bias_{} graph plotted'.format(v_type, layer_indx))
        else:  # even row numbers store weight vals
            display_results.plot_graph(val, None, "{}_layer_weight_{}".format(v_type, layer_indx), im_path)
            print('{}_layer_weight_{} graph plotted'.format(v_type, layer_indx))

    return


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    if args.parent_dir and not torch.cuda.is_available():
        os.chdir(os.path.join(os.path.expanduser(os.environ['USERPROFILE']), args.parent_dir))
        # os.chdir(args.parent_dir)

    # print(args.early_stop)
    # exit()

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
    model = net.NeuralNet(params).cuda() if params.cuda else net.NeuralNet(params)

    print(model)
    logging.info("network structure is")
    logging.info("{}".format(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, betas=(params.beta1, params.beta2))
    # torch.optim.SGD(model.parameters(), lr=params.learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    # loss_fn = net.loss_fn_two_labels
    loss_fn = net.loss_fn
    # loss_fn = net.loss_fn_low_entropy

    # comment to self - print to file the name of loss function !!!!!!
    # print(loss_fn)

    metrics = net.metrics
    incorrect = net.incorrect
    correct_fn = net.correct_classification

    losses = []
    accuracy = []
    grads_per_epoch = []
    vals_per_epoch = []

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, dev_dl, optimizer, loss_fn, metrics, incorrect, correct_fn, params, args.model_dir,
                       args.restore_file)

    print('plotting graphs')
    display_results.plot_graph(losses, None, "General Loss", args.model_dir)
    print('loss graph plotted')
    display_results.plot_graph(accuracy, None, "General dev accuracy", args.model_dir)
    print('accuracy graph plotted')

    plot_summary_graphs_layers(grads_per_epoch, 'Grads', args.model_dir)
    # grads_np = np.array(grads_per_epoch)
    # for i in range(grads_np.shape[1]):
    #     val = grads_np[:, i].tolist()
    #     display_results.plot_graph(val, None, "Grads_layer_{}".format(i+1), args.model_dir)

