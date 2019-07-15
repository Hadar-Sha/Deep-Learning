"""Train the model"""

import argparse
import logging
import os
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
import re
# from tqdm import tqdm

# from utils_gan import Logger
import display_digit as display_results
import utils
import model_gan.gan_net as gan_net
import model_gan.one_label_data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default="C:/Users/H/Documents/Haifa Univ/Thesis/DL-Pytorch-data", help='path to experiments and data folder. not for Server')
parser.add_argument('--data_dir', default='data/black-white-small', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/gan_model/debug', help="Directory containing params.json")
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

    # Return error0
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


def get_stats(val, val_type):

    if isinstance(val, torch.autograd.Variable) and val_type == 'error':
        ret_val = val.data.cpu().numpy()

    elif isinstance(val, torch.autograd.Variable) and val_type == 'pred':
        ret_val = val.data.mean()

    else:
        print('invalid input')
        ret_val = None

    return ret_val


def train(d_model, g_model, d_optimizer, g_optimizer, loss_fn, dataloader, params, epoch, fig):  # , axes):
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
    test_samples = None
    prop = []
    summ = []
    stop_d_perc = 0.2
    num_of_batches = max(1, len(dataloader.dataset) // dataloader.batch_size)

    for i, (real_batch, real_label) in enumerate(dataloader):

        # 1. Train Discriminator
        real_data = Variable(real_batch)
        real_label = Variable(real_label)

        if params.cuda:
            real_data = real_data.cuda()
            real_label = real_label.cuda()

        if real_label.size(1) == 1:
            real_label = real_label.view(real_label.size(0))

        # Generate fake data
        noisy_input = gan_net.noise(real_data.size(0), params.noise_dim, params.noise_type)

        # real_one_hot_v = gan_net.convert_int_to_one_hot_vector(real_label, params.num_classes).to(device)

        real_label = real_label.view(real_label.size(0))

        # Generate fake data
        fake_data = g_model(noisy_input)

        # Train D
        if not params.stop_d or epoch <= (stop_d_perc * params.num_epochs):
            d_error, d_pred_real, d_pred_fake = train_discriminator(d_model, d_optimizer, real_data, fake_data.detach())  # do not remove .detach() here !!!!!
        # else: # if params.stop_d and epoch > (stop_d_perc * params.num_epochs):
            # do not train d

        # 2. Train Generator

        # Train G
        g_error = train_generator(d_model, g_optimizer, fake_data)
        # Log error
        stats = {}
        if not params.stop_d or epoch <= (stop_d_perc * params.num_epochs):
            stats['d_error'] = get_stats(d_error, 'error')
            stats['d_pred_real'] = get_stats(d_pred_real, 'pred')
            stats['d_pred_fake'] = get_stats(d_pred_fake, 'pred')

        stats['g_error'] = get_stats(g_error, 'error')

        if not params.stop_d or epoch <= (stop_d_perc * params.num_epochs):
            G_preds.append(d_pred_fake.data.mean())
            D_preds.append(d_pred_real.data.mean())

        if i % params.save_summary_steps == 0:
            if num_of_batches > 1:
                proportions_batch = real_label.shape[0] / params.batch_size
            else:
                proportions_batch = 1
            prop.append(proportions_batch)
            summ.append(stats)

        if ((i + 1) % max(1, round(0.1 * num_of_batches)) == 0) and (epoch == 0):
            # Display data Images
            real_samples_reshaped = gan_net.vectors_to_samples(real_data)  # ?
            real_titles = gan_net.labels_to_titles(real_label)

            display_results.fill_figure(real_samples_reshaped, fig, i + 1, args.model_dir, -1, 1,
                                        labels=real_titles,
                                        dtype='real')

    stats_mean = {metric: np.sum([x[metric] for x in summ] / np.sum(prop)) for metric in summ[0]}
    # Save Losses for plotting later
    if not params.stop_d or epoch <= (stop_d_perc * params.num_epochs):
        D_losses.append(d_error.item())
    G_losses.append(g_error.item())

    stats_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in stats_mean.items())
    logging.info("train metrics: " + stats_string)

    if ((epoch + 1) % (0.01 * params.num_epochs) == 0) or ((epoch + 1) <= min(10, (0.001 * params.num_epochs))):
        # Display Progress
        print("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        print(stats_string)

        # Display test Images
        test_samples = g_model(test_noise).data.cpu()
        test_samples_reshaped = gan_net.vectors_to_samples(test_samples)

        display_results.fill_figure(test_samples_reshaped, fig, epoch + 1, args.model_dir, -1, 1, labels=None)

    if not params.stop_d or epoch <= (stop_d_perc * params.num_epochs):
        return test_samples, stats_mean['d_error'] + stats_mean['g_error']
    else:
        return test_samples, stats_mean['g_error']


def train_gan(d_model, g_model, train_dataloader, dev_dataloader, d_optimizer, g_optimizer, loss_fn, params, model_dir,
                       restore_file=None):

    best_loss = np.inf
    dest_min = 0
    dest_max = 255
    curr_min = -1
    curr_max = 1

    fig = display_results.create_figure()

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        test_samples, loss_mean_sum = \
            train(d_model, g_model, d_optimizer, g_optimizer, loss_fn, train_dataloader, params, epoch, fig)

        is_best = loss_mean_sum <= best_loss

        g_grads_graph, _ = get_network_grads(g_model)
        d_grads_graph, _ = get_network_grads(d_model)
        g_vals_graph = collect_network_statistics(g_model)
        d_vals_graph = collect_network_statistics(d_model)

        grads_per_epoch_g.append(g_grads_graph)
        grads_per_epoch_d.append(d_grads_graph)
        vals_per_epoch_g.append(g_vals_graph)
        vals_per_epoch_d.append(d_vals_graph)

        if is_best:
            logging.info("- Found new best loss")
            print("Epoch {}/{}".format(epoch + 1, params.num_epochs))
            print("- Found new best loss")
            best_loss = loss_mean_sum
            print("mean loss is {:05.3f}".format(loss_mean_sum))
            loss_metric_dict = {'loss': loss_mean_sum}

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(model_dir, "metrics_dev_best_weights.json")
            utils.save_dict_to_json(loss_metric_dict, best_json_path, epoch + 1)

            # display_results.plot_graph(g_grads_graph, d_grads_graph, "Grads_Best", args.model_dir, epoch=epoch + 1)

            if test_samples is not None:
                np_test_samples = np.array(test_samples)
                np_test_samples = \
                    dest_min + (dest_max - dest_min) * (np_test_samples - curr_min) / (curr_max - curr_min)
                np_test_samples = np.around(np_test_samples).astype(int)
                np_test_out = (test_noise.cpu().numpy())

                data_path = os.path.join(model_dir, 'data')
                if not os.path.isdir(data_path):
                    os.mkdir(data_path)

                test_all_data = (np.concatenate((np_test_samples, np_test_out), axis=1)).tolist()
                last_csv_path = os.path.join(data_path, "best_samples_epoch_{}.csv".format(epoch + 1))
                utils.save_incorrect_to_csv(test_all_data, last_csv_path)

        if test_samples is not None:
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': d_model.state_dict(),
                                   'optim_dict': d_optimizer.state_dict()}, is_best=is_best, checkpoint=model_dir,
                                  ntype='d')

            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': g_model.state_dict(),
                                   'optim_dict': g_optimizer.state_dict()}, is_best=is_best, checkpoint=model_dir,
                                  ntype='g')

            np_test_samples = np.array(test_samples)
            np_test_samples = \
                dest_min + (dest_max - dest_min) * (np_test_samples - curr_min) / (curr_max - curr_min)
            np_test_samples = np.around(np_test_samples).astype(int)
            np_test_out = (test_noise.cpu().numpy())

            data_path = os.path.join(model_dir, 'data')
            if not os.path.isdir(data_path):
                os.mkdir(data_path)

            test_all_data = (np.concatenate((np_test_samples, np_test_out), axis=1)).tolist()
            last_csv_path = os.path.join(data_path, "samples_epoch_{}.csv".format(epoch + 1))
            utils.save_incorrect_to_csv(test_all_data, last_csv_path)

    display_results.close_figure(fig)
    return


def get_network_grads(net):
    weight_string = "weight"
    bias_string = "bias"
    output_gradients = []
    output_names = []

    # tmp_grad = {}

    parameters_names = list(net.state_dict().keys())
    j = 0
    for i in range(len(parameters_names)):
        par = parameters_names[i - j]
        is_rel_w = re.search(weight_string, par)
        is_rel_b = re.search(bias_string, par)
        if is_rel_w is None and is_rel_b is None:
            parameters_names.remove(par)
            j += 1
    # grads = torch.autograd.grad(loss_fn, parameters_names, retain_graph=True)

    for name, param in net.named_parameters():
        if name in parameters_names:
            # tmp_grad[name] = param.grad.norm()
            # output_gradients.append(param.grad.norm().item())
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
            # output_gradients.append(param.grad.data.cpu().numpy().tolist())
            output_names.append(name)

    # output_gradients.append(tmp_grad)
    return output_gradients, output_names


def collect_network_statistics(net):

    net_grads_graph = []

    for param_tensor in net.state_dict():

        if (net.state_dict()[param_tensor]).dtype != torch.float:
            continue

        all_net_vals = ((net.state_dict()[param_tensor]).cpu().numpy()).tolist()

        # if needed, flatten the list to get one nim and one max
        flat_net_grads = []
        if isinstance(all_net_vals, (list,)):

            for elem in all_net_vals:
                if isinstance(elem, (list,)) and isinstance(elem[0], (list,)):
                    for item in elem:
                        flat_net_grads.extend(item)
                elif isinstance(elem, (list,)):
                    flat_net_grads.extend(elem)
                else:
                    flat_net_grads.extend([elem])
        else:
            flat_net_grads = all_net_vals

        net_grads_graph.append([min(flat_net_grads), np.median(flat_net_grads), max(flat_net_grads)])

    return net_grads_graph


def plot_summary_graphs_layers(vals_per_epoch, n_type, v_type, im_path):
    vals_np = np.array(vals_per_epoch)
    for it in range(vals_np.shape[1]):
        val = vals_np[:, it].tolist()
        if it % 2:
            display_results.plot_graph(val, None, "{}_{}_layer_bias_{}".format(n_type, v_type, it), im_path)
        else:
            display_results.plot_graph(val, None, "{}_{}_layer_weight_{}".format(n_type, v_type, it), im_path)

    return


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

    if params.init_weights:
        discriminator.apply(gan_net.weights_init)
        generator.apply(gan_net.weights_init)

    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()

    print(discriminator)
    logging.info("discriminator network structure is")
    logging.info("{}".format(discriminator))
    print(generator)
    logging.info("generator network structure is")
    logging.info("{}".format(generator))

    # Optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=params.d_learning_rate, betas=(params.beta1, params.beta2))
    g_optimizer = optim.Adam(generator.parameters(), lr=params.g_learning_rate, betas=(params.beta1, params.beta2))

    # fetch loss function and metrics
    loss_fn = gan_net.loss_fn

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    num_test_samples = 20
    test_noise = gan_net.noise(num_test_samples, params.noise_dim, params.noise_type)

    # test_labels = list(range(num_test_samples))
    # test_labels = [it % params.num_classes for it in test_labels]
    # # test_labels = [0 for _ in range(num_test_samples)]
    # test_labels = torch.Tensor(test_labels)
    # test_labels = test_labels.type(torch.LongTensor)
    # test_labels = test_labels.to(device)
    #
    # test_labels = test_labels.view(num_test_samples, -1)
    # test_one_hot_v = gan_net.convert_int_to_one_hot_vector(test_labels, params.num_classes).to(device)

    D_losses = []
    G_losses = []
    D_preds = []
    G_preds = []

    grads_per_epoch_g = []
    grads_per_epoch_d = []
    vals_per_epoch_g = []
    vals_per_epoch_d = []

    train_gan(discriminator, generator, train_dl, dev_dl, d_optimizer, g_optimizer, loss_fn, params, args.model_dir,
              args.restore_file)

    # track results
    display_results.plot_graph(G_losses, D_losses, "Loss", args.model_dir)
    display_results.plot_graph(G_preds, D_preds, "Predictions", args.model_dir)

    plot_summary_graphs_layers(grads_per_epoch_g, 'G', 'Grads', args.model_dir)
    # grads_np_g = np.array(grads_per_epoch_g)
    # for i in range(grads_np_g.shape[1]):
    #     val_g = grads_np_g[:, i].tolist()
    #     if i % 2:
    #         display_results.plot_graph(val_g, None, "G_Grads_layer_bias_{}".format(i + 1), args.model_dir)
    #     else:
    #         display_results.plot_graph(val_g, None, "G_Grads_layer_weight_{}".format(i + 1), args.model_dir)

    plot_summary_graphs_layers(grads_per_epoch_d, 'D', 'Grads', args.model_dir)
    # grads_np_d = np.array(grads_per_epoch_d)
    # for i in range(grads_np_d.shape[1]):
    #     val_d = grads_np_d[:, i].tolist()
    #     if i % 2:
    #         display_results.plot_graph(None, val_d, "D_Grads_layer_bias_{}".format(i + 1), args.model_dir)
    #     else:
    #         display_results.plot_graph(None, val_d, "D_Grads_layer_weight_{}".format(i + 1), args.model_dir)

    plot_summary_graphs_layers(vals_per_epoch_g, 'G', 'Vals', args.model_dir)
    # vals_np_g = np.array(vals_per_epoch_g)
    # for i in range(vals_np_g.shape[1]):
    #     val_g = vals_np_g[:, i].tolist()
    #     if i % 2:
    #         display_results.plot_graph(val_g, None, "G_Vals_layer_bias_{}".format(i + 1), args.model_dir)
    #     else:
    #         display_results.plot_graph(val_g, None, "G_Vals_layer_weight_{}".format(i + 1), args.model_dir)

    plot_summary_graphs_layers(vals_per_epoch_d, 'D', 'Vals', args.model_dir)
    # grads_np_d = np.array(grads_per_epoch_d)
    # for i in range(grads_np_d.shape[1]):
    #     val_d = grads_np_d[:, i].tolist()
    #     if i % 2:
    #         display_results.plot_graph(None, val_d, "D_Vals_layer_bias_{}".format(i + 1), args.model_dir)
    #     else:
    #         display_results.plot_graph(None, val_d, "D_Vals_layer_weight_{}".format(i + 1), args.model_dir)

