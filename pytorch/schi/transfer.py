import argparse
import logging
import os
import re
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
import display_digit as display_results

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default="C:/Users/H/Documents/Haifa Univ/Thesis/DL-Pytorch-data", help='path to experiments and data folder. not for Server')
parser.add_argument('--data_dir', default='data/two-labels/data-two-labels-big', help="Directory containing the destination dataset")
parser.add_argument('--model_dir', default='experiments/transfer_training_model/in-debug',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best',
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
# added by me
parser.add_argument('--model_out_dir', default='experiments/transfer_training_model/out-debug',
                    help="Directory to write transfer results")


def load_model(model_dir, restore_file):
    # reload weights from restore_file if specified
    if restore_file is not None and model_dir is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, None)  # optimizer)
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


def after_transfer_train_and_evaluate(model, train_dataloader, dev_dataloader, optimizer, loss_fn, metrics, incorrect, correct_fn, params, model_dir, model_out_dir, restore_file):

    best_dev_acc = 0.0

    fig = display_results.create_figure()

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params, epoch, fig, model_out_dir, losses, grayscale=True)

        # Evaluate for one epoch on validation set
        dev_metrics, incorrect_samples, correct_samples = evaluate(model, loss_fn, dev_dataloader, metrics, incorrect, correct_fn, params, epoch)

        dev_acc = dev_metrics['accuracy_two_labels']
        is_best = dev_acc >= best_dev_acc

        grads_graph, _ = get_network_grads(model)
        vals_graph = collect_network_statistics(model)

        grads_per_epoch.append(grads_graph)
        vals_per_epoch.append(vals_graph)


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
            utils.save_dict_to_json(dev_metrics, best_json_path, epoch + 1)

            best_csv_path = os.path.join(model_out_dir, "incorrect_best_samples.csv")
            utils.save_incorrect_to_csv(incorrect_samples, best_csv_path)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(model_out_dir, "metrics_dev_last_weights.json")
        utils.save_dict_to_json(dev_metrics, last_json_path, epoch + 1)

        last_csv_path = os.path.join(model_out_dir, "incorrect_last_samples.csv")
        utils.save_incorrect_to_csv(incorrect_samples, last_csv_path)

        accuracy.append(dev_acc)

    display_results.close_figure(fig)

    return


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
        os.chdir(args.parent_dir)

    json_path = os.path.join(args.model_out_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # # Set the random seed for reproducible experiments
    # torch.manual_seed(230)
    # if params.cuda:
    #     torch.cuda.manual_seed(230)

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

    num_of_batches = max(1, len(train_dl.dataset) // train_dl.batch_size)
    logging.info("data-set size: {}".format(len(train_dl.dataset)))
    logging.info("number of batches: {}".format(num_of_batches))

    # Define the model and optimizer
    model = net.NeuralNet(params).cuda() if params.cuda else net.NeuralNet(params)

    load_model(args.model_dir, args.restore_file)

    print(model)
    logging.info("network structure is")
    logging.info("{}".format(model))

    # status_before_transfer = []
    # for param_tensor in model.state_dict():
    #     status_before_transfer.append([param_tensor,
    #                (model.state_dict()[param_tensor].norm()).item(), list(model.state_dict()[param_tensor].size())])
    #     status_before_transfer.append(((model.state_dict()[param_tensor]).cpu().numpy()).tolist())

    # changing last fully connected layer
    num_ftrs = model.fc4.in_features
    model.fc4 = nn.Linear(num_ftrs, 20)  # 10)

    model = model.to(device)

    print(model)
    logging.info("network structure after transfer is")
    logging.info("{}".format(model))

    optimizer = torch.optim.SGD(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = net.loss_fn_two_labels

    metrics = net.metrics
    incorrect = net.incorrect_two_labels
    correct_fn = net.correct_classification_two_labels

    losses = []
    accuracy = []
    grads_per_epoch = []
    vals_per_epoch = []

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    after_transfer_train_and_evaluate(model, train_dl, dev_dl, optimizer, loss_fn, metrics, incorrect, correct_fn, params,
                                      args.model_dir, args.model_out_dir, args.restore_file)

    # load_model(args.model_out_dir, args.restore_file)
    #
    # status_after_transfer = []
    # for param_tensor in model.state_dict():
    #     status_after_transfer.append([param_tensor,
    #          (model.state_dict()[param_tensor].norm()).item(), list(model.state_dict()[param_tensor].size())])
    #     status_after_transfer.append(((model.state_dict()[param_tensor]).cpu().numpy()).tolist())
    #
    # filepath = os.path.join(args.model_out_dir, 'wb_ext.csv')
    # with open(filepath, "w", newline='') as myfile:
    #     csvwr = csv.writer(myfile)
    #     for elem in status_before_transfer:
    #         if isinstance(elem, (list,)) and isinstance(elem[0], (list,)):
    #             for row in elem:
    #                 csvwr.writerow(row)
    #         else:
    #             csvwr.writerow(elem)
    #     for elem_a in status_after_transfer:
    #         if isinstance(elem, (list,)) and isinstance(elem[0], (list,)):
    #             for row in elem:
    #                 csvwr.writerow(row)
    #         else:
    #             csvwr.writerow(elem)

    print('plotting graphs')
    display_results.plot_graph(losses, None, "General Loss", args.model_out_dir)
    print('loss graph plotted')
    display_results.plot_graph(accuracy, None, "General dev accuracy", args.model_out_dir)
    print('accuracy graph plotted')

    plot_summary_graphs_layers(grads_per_epoch, 'Grads', args.model_out_dir)