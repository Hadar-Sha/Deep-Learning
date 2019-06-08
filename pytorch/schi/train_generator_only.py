"""Train the model"""

import argparse
import logging
import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
# from tqdm import tqdm

# from utils_gan import Logger
import display_digit as display_results
import utils
import model_gan.generator_only_net as gan_net
# import model_gan.two_labels_data_loader as data_loader
import model_gan.one_label_data_loader as data_loader
# import model_gan.single_sample_data_loader as data_loader


parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default="C:/Users/H/Documents/Haifa Univ/Thesis/DL-Pytorch-data", help='path to experiments and data folder. not for Server')
parser.add_argument('--data_dir', default='data/black-white', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/generator_model/debug', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'

# data-two-labels-big # grayscale-logits # data/data-w-gray-only-2 data/data-with-grayscale-4000


def train_generator(optimizer, fake_data, real_data, mse_loss_fn):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data

    # Calculate error and backpropagate
    mse_error = mse_loss_fn(fake_data, real_data)
    mse_error.backward()
    # Update weights with gradients
    optimizer.step()

    return mse_error


def get_stats(val, val_type):

    if isinstance(val, torch.autograd.Variable) and val_type == 'error':
        ret_val = val.data.cpu().numpy()

    elif isinstance(val, torch.autograd.Variable) and val_type == 'pred':
        ret_val = val.data.mean()

    else:
        print('invalid input')
        ret_val = None

    return ret_val


def train(g_model, g_optimizer, mse_loss_fn, dataloader, params, epoch, fig):
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        epoch:
        fig:
    """
    test_samples = None
    prop = []
    summ = []

    for i, (real_batch, real_label) in enumerate(dataloader):

        # 1. Train Discriminator
        real_data = Variable(real_batch)
        real_label = Variable(real_label)

        if torch.cuda.is_available():
            real_data = real_data.cuda()
            real_label = real_label.cuda()

        # Generate fake data
        noisy_input = gan_net.noise(real_data.size(0), params.noise_dim)

        noisy_label = Variable(torch.randint(params.num_classes, (real_data.size(0),)))
        noisy_label = noisy_label.type(torch.LongTensor).to(device)

        real_one_hot_v = gan_net.convert_int_to_one_hot_vector(real_label, params.num_classes).to(device)

        real_label = real_label.view(real_label.size(0))
        # noisy_label = noisy_label.view(real_data.size(0), -1)
        noisy_one_hot_v = gan_net.convert_int_to_one_hot_vector(noisy_label, params.num_classes).to(device)

        fake_data = g_model(noisy_input, real_one_hot_v)  # .detach()
        # fake_data = g_model(noisy_input, noisy_one_hot_v)  # .detach()

        # 2. Train Generator

        # fake_data = g_model(noisy_input, noisy_one_hot_v)   # not sure

        # Train G
        g_error = train_generator(g_optimizer, fake_data, real_data, mse_loss_fn)

        # # Log error
        stats = {}
        stats['g_error'] = get_stats(g_error, 'error')

        # Save Losses for plotting later
        D_losses.append(g_error.item())

        # Display Progress
        # Display Images

        if i % params.save_summary_steps == 0:
            proportions_batch = real_label.shape[0] / params.batch_size
            prop.append(proportions_batch)
            summ.append(stats)

    stats_mean = {metric: np.sum([x[metric] for x in summ] / np.sum(prop)) for metric in summ[0]}
    stats_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in stats_mean.items())
    logging.info("train metrics: " + stats_string)
    if (epoch + 1) % (0.01 * params.num_epochs) == 0:
        print("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        print(stats_string)

        test_samples = g_model(test_noise, test_one_hot_v).data.cpu()
        test_samples_reshaped = gan_net.vectors_to_samples(test_samples)  # ?
        test_titles = gan_net.labels_to_titles(test_labels)

        display_results.fill_figure(test_samples_reshaped, fig, epoch + 1, args.model_dir, labels=test_titles)

        real_samples_reshaped = gan_net.vectors_to_samples(real_data)  # ?
        real_titles = gan_net.labels_to_titles(real_label)

        display_results.fill_figure(real_samples_reshaped, fig, epoch + 1, args.model_dir, labels=real_titles,
                                    dtype='real')

    return test_samples, real_data, stats_mean['g_error']


def train_g(g_model, train_dataloader, g_optimizer, mse_loss_fn, params, model_dir):

    best_loss = np.inf

    fig = display_results.create_figure()

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        test_samples, real_samples, loss_mean = train(g_model, g_optimizer, mse_loss_fn, train_dataloader, params, epoch, fig)

        is_best = loss_mean <= best_loss

        if is_best:
            logging.info("- Found new best loss")
            print("Epoch {}/{}".format(epoch + 1, params.num_epochs))
            print("- Found new best loss")
            best_loss = loss_mean
            print("mean loss is {:05.3f}".format(loss_mean))
            loss_metric_dict = {'loss': loss_mean}

            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': g_model.state_dict(),
                                   'optim_dict': g_optimizer.state_dict()}, is_best=is_best, checkpoint=model_dir)

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_min_avg_loss_best_weights.json")
            utils.save_dict_to_json(loss_metric_dict, best_json_path)

            if test_samples is not None:
                np_test_samples = np.array(test_samples)
                np_test_samples = np.around(np_test_samples * 127.5 + 127.5).astype(int)
                np_test_out = (test_noise.cpu().numpy())
                np_test_labels = (test_labels.view(test_labels.shape[0], -1).cpu().numpy())

                data_path = os.path.join(model_dir, 'data')
                if not os.path.isdir(data_path):
                    os.mkdir(data_path)

                test_all_data = (np.concatenate((np_test_samples, np_test_out, np_test_labels), axis=1)).tolist()
                last_csv_path = os.path.join(data_path, "best_samples_epoch_{}.csv".format(epoch + 1))
                utils.save_incorrect_to_csv(test_all_data, last_csv_path)

        if test_samples is not None:

            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': g_model.state_dict(),
                                   'optim_dict': g_optimizer.state_dict()}, is_best=False, checkpoint=model_dir,
                                  ntype='g')

            np_test_samples = np.array(test_samples)
            np_test_samples = np.around(np_test_samples * 127.5 + 127.5).astype(int)
            np_test_out = (test_noise.cpu().numpy())
            np_test_labels = (test_labels.view(test_labels.shape[0], -1).cpu().numpy())

            data_path = os.path.join(model_dir, 'data')
            if not os.path.isdir(data_path):
                os.mkdir(data_path)

            test_all_data = (np.concatenate((np_test_samples, np_test_out, np_test_labels), axis=1)).tolist()
            last_csv_path = os.path.join(data_path, "samples_epoch_{}.csv".format(epoch+1))
            utils.save_incorrect_to_csv(test_all_data, last_csv_path)

    display_results.close_figure(fig)
    return


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
    if args.parent_dir and not torch.cuda.is_available():
        os.chdir(args.parent_dir)
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    # # Set the random seed for reproducible experiments
    # torch.manual_seed(230)
    # if params.cuda:
    #     torch.cuda.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("device {} will be used".format(device))

    logging.info("Loading the datasets...")

    # change here !!!! only single sample !!!!
    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['train', 'dev'], args.data_dir, params)
    train_dl = dataloaders['train']
    dev_dl = dataloaders['dev']

    logging.info("data was loaded from {}".format(args.data_dir))
    logging.info("- done.")

    # Define the model and optimizer
    generator = gan_net.GeneratorNet(params)

    generator.apply(gan_net.weights_init)

    if torch.cuda.is_available():
        generator.cuda()

    print(generator)
    logging.info("generator network structure is")
    logging.info("{}".format(generator))

    g_optimizer = optim.Adam(generator.parameters(), lr=params.g_learning_rate, betas=(params.beta1, params.beta2))

    # fetch loss functions
    # mse_loss_fn = gan_net.bce_loss_fn
    mse_loss_fn = gan_net.mse_loss_fn

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    num_test_samples = 20
    test_noise = gan_net.noise(num_test_samples, params.noise_dim)

    test_labels = list(range(num_test_samples))
    test_labels = [it % params.num_classes for it in test_labels]
    # test_labels = [0 for _ in range(num_test_samples)]
    test_labels = torch.Tensor(test_labels)
    test_labels = test_labels.type(torch.LongTensor)
    test_labels = test_labels.to(device)

    test_labels = test_labels.view(num_test_samples, -1)
    test_one_hot_v = gan_net.convert_int_to_one_hot_vector(test_labels, params.num_classes).to(device)

    D_losses = []
    G_losses = []
    D_preds = []
    G_preds = []
    accuracy_vals = []

    train_g(generator, train_dl, g_optimizer, mse_loss_fn, params, args.model_dir)

    # track results
    display_results.plot_graph(G_losses, D_losses, "Loss", args.model_dir)
    # display_results.plot_graph(G_preds, D_preds, "Predictions", args.model_dir)

    g_grads_graph = collect_network_statistics(generator)
    # print (not g_grads_graph)

    display_results.plot_graph(g_grads_graph, [], "Grads", args.model_dir)
