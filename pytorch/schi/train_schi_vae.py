"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

import math
import utils
import model_schi_vae.schi_vae_net as vae_net
import model_schi_vae.one_label_data_loader as data_loader
import display_digit as display_results
# from evaluate import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/with-grayscale/data-with-grayscale-4000', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/schi_vae_model', help="Directory containing params.json")
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

    reconstructed_batch_shaped = None

    # # summary for current training loop and a running average object for loss
    summ = []
    prop = []

    loss_avg = utils.WeightedAverage()

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
        loss, bce_loss_part, kl_loss_part = loss_fn(reconstructed_batch, train_batch, mean, log_of_var)

        # print('loss:{:.4f}'.format(loss.item()))
        # logging.info('loss:{:.4f}'.format(loss.item()))

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        prop_for_loss = train_batch.shape[0] / params.batch_size
        loss_avg.update(loss.item(), prop_for_loss)

        # save loss of this batch
        summary_batch = {'loss': loss.item(), 'BCE': bce_loss_part.item(), 'KL': kl_loss_part.item()}
        summ.append(summary_batch)

        proportions_batch = labels_batch.shape[0] / params.batch_size
        prop.append(proportions_batch)

    # Evaluate summaries only once in a while
    if (epoch + 1) % (0.01 * params.num_epochs) == 0:

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        train_batch_shaped = vae_net.vectors_to_samples(train_batch)
        reconstructed_batch_shaped = vae_net.vectors_to_samples(reconstructed_batch.detach())  # .data.cpu().numpy()

        # create new samples according to learnt parameters, mean, log_of_var
        batch_size = train_batch.shape[0]
        created_batch = model.inference(params.output_size, n=batch_size)
        created_batch_shaped = vae_net.vectors_to_samples(created_batch.detach())

        batch = iter(dataloader).next()[0]  # get random batch
        batch = Variable(batch)
        if torch.cuda.is_available():
            batch = Variable(batch).cuda()

        x_one = batch[0:1]  # get first sample
        x_two = batch[1:2]  # get second sample
        generated_images = []
        for alpha in torch.arange(0.0, 1.0, 0.05):
            sample = model.generation_with_interpolation(x_one, x_two, alpha)
            generated_images.append(sample)
        generated_images = torch.cat(generated_images, 0).cpu().data
        generated_images_shaped = vae_net.vectors_to_samples(generated_images)

        display_results.fill_figure(train_batch_shaped, fig, epoch + 1, args.model_dir, True, 'data')
        display_results.fill_figure(reconstructed_batch_shaped, fig, epoch + 1, args.model_dir, True, 'reconstructed')
        display_results.fill_figure(created_batch_shaped, fig, epoch + 1, args.model_dir, True, 'created')
        display_results.fill_figure(generated_images_shaped, fig, epoch+1, args.model_dir, True, 'generated')

    # compute mean of all metrics in summary (loss, bce part, kl part)
    if isinstance(loss, torch.autograd.Variable):
        loss_v = loss.data.cpu().numpy()
    losses.append(loss_v.item())

    if isinstance(bce_loss_part, torch.autograd.Variable):
        bce_v = bce_loss_part.data.cpu().numpy()
    bce_losses.append(bce_v)

    if isinstance(kl_loss_part, torch.autograd.Variable):
        kl_v = kl_loss_part.data.cpu().numpy()
    kl_losses.append(kl_v)

    metrics_mean = {metric: np.sum([x[metric] for x in summ] / np.sum(prop)) for metric in summ[0]}
    # stats = get_stats(bce_loss_part, kl_loss_part)
    # stats
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

    # print to screen every 1% of iterations
    if (epoch+1) % (0.01*params.num_epochs) == 0:
        print("train Epoch {}/{}".format(epoch + 1, params.num_epochs))
        print(metrics_string)

    return reconstructed_batch_shaped, metrics_mean['loss']


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

    best_loss = math.inf

    fig = display_results.create_figure()

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        reconstructed_samples, loss_mean = train(model, optimizer, loss_fn, train_dataloader, params, epoch, fig)

        is_best = loss_mean <= best_loss

        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()}, is_best=is_best, checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best loss")
            print("Epoch {}/{}".format(epoch + 1, params.num_epochs))
            print("- Found new best loss")
            best_loss = loss_mean
            print("mean loss is {:05.3f}".format(loss_mean))
            loss_metric_dict = {'loss': loss_mean}

            # utils.save_checkpoint({'epoch': epoch + 1,
            #                        'state_dict': model.state_dict(),
            #                        'optim_dict': optimizer.state_dict()}, is_best=is_best, checkpoint=model_dir)

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_min_avg_loss_best_weights.json")
            utils.save_dict_to_json(loss_metric_dict, best_json_path)

            # best_csv_path = os.path.join(model_dir, "reconstructed_min_avg_loss_best_samples.csv")
            # utils.save_incorrect_to_csv(reconstructed_samples, best_csv_path)

        if reconstructed_samples is not None:
            np_reconstructed_samples = np.array(reconstructed_samples)
            np_reconstructed_samples = np.around(np_reconstructed_samples * 255).astype(int)

            data_path = os.path.join(model_dir, 'data')
            if not os.path.isdir(data_path):
                os.mkdir(data_path)

            last_csv_path = os.path.join(data_path, "samples_epoch_{}.csv".format(epoch+1))
            utils.save_incorrect_to_csv(np_reconstructed_samples, last_csv_path)

    display_results.close_figure(fig)


def get_stats(bce_loss, kl_loss):
    stats = {}
    if isinstance(bce_loss, torch.autograd.Variable):
        bce_loss = bce_loss.data.cpu().numpy()
        stats['bce_loss'] = bce_loss
    if isinstance(kl_loss, torch.autograd.Variable):
        kl_loss = kl_loss.data.cpu().numpy()
        stats['kl_loss'] = kl_loss

    return stats


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
    # model.apply(vae_net.weights_init)
    print(model)
    logging.info("network structure is")
    logging.info("{}".format(model))

    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate, betas=(params.beta1, params.beta2))

    # fetch loss function and metrics
    loss_fn = vae_net.loss_fn

    # metrics = net.metrics
    # incorrect = net.incorrect
    losses = []
    bce_losses = []
    kl_losses = []

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_vae(model, train_dl, optimizer, loss_fn, params, args.model_dir)

    display_results.plot_graph(losses, None, "General Loss", args.model_dir)
    display_results.plot_graph(bce_losses, kl_losses, "VAE Loss", args.model_dir)

    grads_graph = collect_network_statistics(model)
    display_results.plot_graph(grads_graph, None, "Grads", args.model_dir)
