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
# import model_gan.acgan_net as gan_net
import model_gan.acgan_hiding_net as gan_net
import model_gan.two_labels_data_loader as data_loader
# import model_gan.one_label_data_loader as data_loader
#import model_gan.single_sample_data_loader as data_loader


parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default="C:/Users/H/Documents/Haifa Univ/Thesis/DL-Pytorch-data", help='path to experiments and data folder. not for Server')
parser.add_argument('--data_dir', default='data/two-labels/single-class/hidden-easy-big-only-9-5', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/acgan_model/hiding_scheme/debug', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train_discriminator(d, optimizer, real_data, fake_data, real_labels, fake_labels, r_f_loss_fn, c_loss_fn, num_classes):
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_r_f_real, prediction_class = d(real_data)
    # Calculate error and backpropagate
    is_real_fake_error = r_f_loss_fn(prediction_r_f_real, gan_net.real_data_target(real_data.size(0)))
    class_error = c_loss_fn(prediction_class, real_labels, num_classes)

    total_real_error = is_real_fake_error + class_error
    total_real_error.backward()

    # compute the current classification accuracy
    class_accuracy_real = gan_net.compute_acc(prediction_class, real_labels)

    # 1.2 Train on Fake Data
    prediction_r_f_fake, prediction_class = d(fake_data)
    # Calculate error and backpropagate
    is_real_fake_error = r_f_loss_fn(prediction_r_f_fake, gan_net.fake_data_target(real_data.size(0)))
    class_error = c_loss_fn(prediction_class, fake_labels, num_classes)

    total_fake_error = is_real_fake_error + class_error
    total_fake_error.backward()

    # compute the current classification accuracy
    class_accuracy_fake = gan_net.compute_acc(prediction_class, fake_labels)
    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error0
    return total_real_error + total_fake_error, prediction_r_f_real, prediction_r_f_fake, class_accuracy_real, class_accuracy_fake


def train_generator(d, optimizer, fake_data, fake_labels, r_f_loss_fn, c_loss_fn, num_classes):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction_r_f, prediction_class = d(fake_data)
    # Calculate error and backpropagate

    # real_data_target: not a mistake - a tip from ganHacks
    is_real_fake_error = r_f_loss_fn(prediction_r_f, gan_net.real_data_target(prediction_r_f.size(0)))
    class_error = c_loss_fn(prediction_class, fake_labels, num_classes)

    total_generator_error = is_real_fake_error + class_error
    total_generator_error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return total_generator_error, prediction_r_f


def get_stats(val, val_type):

    if isinstance(val, torch.autograd.Variable) and val_type == 'error':
        ret_val = val.data.cpu().numpy()

    elif isinstance(val, torch.autograd.Variable) and val_type == 'pred':
        ret_val = val.data.mean()

    else:
        print('invalid input')
        ret_val = None

    return ret_val


def train(d_model, d_optimizer, g_model, g_optimizer, r_f_loss_fn, c_loss_fn, dataloader, params, epoch, fig):
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
    num_of_batches = max(1, len(dataloader.dataset)//dataloader.batch_size)

    for i, (real_batch, real_label) in enumerate(dataloader):

        # 1. Train Discriminator
        real_data = Variable(real_batch)
        real_label = Variable(real_label)

        if torch.cuda.is_available():
            real_data = real_data.cuda()
            real_label = real_label.cuda()

        # Generate fake data
        noisy_input = gan_net.noise(real_data.size(0), params.noise_dim, params.noise_type)
        # noisy_label = Variable(torch.randint(params.num_classes//2, (real_data.size(0),)))

        # temp_labels = list(range(real_data.size(0)))
        # temp_labels = [it % params.num_classes for it in temp_labels]

        temp_labels = []
        for j in range(len(possible_classes) - 1):
            temp_labels.extend([possible_classes[j] for _ in range(real_data.size(0) // len(possible_classes))])
        # last class will have amount of samples to complete to num_test_samples
        temp_labels.extend([possible_classes[len(possible_classes) - 1]
                            for _ in range(
                real_data.size(0) - (len(possible_classes) - 1) * (real_data.size(0) // len(possible_classes)))])

        # temp_labels.extend([5 for _ in range(real_data.size(0) // 4)])
        # temp_labels.extend([7 for _ in range(real_data.size(0) - 3 * (real_data.size(0) // 4))])

        noisy_label = torch.Tensor(temp_labels)
        noisy_label = Variable(noisy_label)

        # if len(noisy_label.shape) == 2:
        #     noisy_label = noisy_label.view(real_label.size(0), -1, 2)
        # else:
        #     noisy_label = noisy_label.view(real_label.size(0), -1)

        if real_label.size(1) == 1:
            real_label = real_label.view(real_label.size(0))

        # noisy_label = Variable(torch.randint(params.num_classes, (real_data.size(0),)))
        noisy_label = noisy_label.type(torch.LongTensor).to(device)

        real_label_before = torch.index_select(real_label, 1, torch.tensor([0], device=real_label.device))
        real_label_after = torch.index_select(real_label, 1, torch.tensor([1], device=real_label.device))
        real_bef_one_hot_v = gan_net.convert_int_to_one_hot_vector(real_label_before, params.num_classes).to(device)
        real_aft_one_hot_v = gan_net.convert_int_to_one_hot_vector(real_label_after, params.num_classes).to(device)
        real_one_hot_v = torch.cat((real_bef_one_hot_v, real_aft_one_hot_v), 1)
        # real_label = real_label.view(real_label.size(0))
        # noisy_label = noisy_label.view(real_data.size(0), -1)
        noisy_label_before = torch.index_select(noisy_label, 1, torch.tensor([0], device=noisy_label.device))
        noisy_label_after = torch.index_select(noisy_label, 1, torch.tensor([1], device=noisy_label.device))
        noisy_bef_one_hot_v = gan_net.convert_int_to_one_hot_vector(noisy_label_before, params.num_classes).to(device)
        noisy_aft_one_hot_v = gan_net.convert_int_to_one_hot_vector(noisy_label_after, params.num_classes).to(device)
        noisy_one_hot_v = torch.cat((noisy_bef_one_hot_v, noisy_aft_one_hot_v), 1)

        fake_data = g_model(noisy_input, noisy_one_hot_v)

        # Train D
        d_error, d_pred_real, d_pred_fake, class_accuracy_real, class_accuracy_fake = \
            train_discriminator(d_model, d_optimizer, real_data, fake_data.detach(), real_label, noisy_label, r_f_loss_fn, c_loss_fn, params.num_classes)  # do not remove .detach() here !!!!!

        # 2. Train Generator

        # Train G
        g_error, d_pred_fake_g = train_generator(d_model, g_optimizer, fake_data, noisy_label, r_f_loss_fn, c_loss_fn, params.num_classes)  # do not change. without .detach() here !!!!!

        # # Log error
        stats = {}
        stats['d_error'] = get_stats(d_error, 'error')
        stats['g_error'] = get_stats(g_error, 'error')
        stats['class_accuracy_real'] = torch.tensor(class_accuracy_real).numpy()
        stats['class_accuracy_fake'] = torch.tensor(class_accuracy_fake).numpy()
        stats['d_pred_real'] = get_stats(d_pred_real, 'pred')
        stats['d_pred_fake'] = get_stats(d_pred_fake, 'pred')
        stats['d_pred_fake_g'] = get_stats(d_pred_fake_g, 'pred')

        if i % params.save_summary_steps == 0:
            if num_of_batches > 1:
                proportions_batch = real_label.shape[0] / params.batch_size
            else:
                proportions_batch = 1
            prop.append(proportions_batch)
            summ.append(stats)

        if ((i + 1) % max(1, round(0.25*num_of_batches)) == 0) and (epoch == 0):
        # if ((i + 1) % max(1, round(0.1*num_of_batches)) == 0) and (epoch == 0):
            # Display data Images
            real_samples_reshaped = gan_net.vectors_to_samples(real_data)  # ?
            real_titles = gan_net.labels_to_titles(real_label)

            print('plotting batch #{} of input data'.format(i+1))
            display_results.fill_figure(real_samples_reshaped, fig, i + 1, args.model_dir, -1, 1, withgrayscale=True,
                                        labels=real_titles, dtype='real')

    stats_mean = {metric: np.sum([x[metric] for x in summ] / np.sum(prop)) for metric in summ[0]}
    # Save Losses for plotting later
    D_losses.append(d_error.item())
    G_preds.append(d_pred_fake.data.mean().item())
    D_preds.append(d_pred_real.data.mean().item())
    G_losses.append(g_error.item())
    real_accuracy_vals.append(class_accuracy_real)
    fake_accuracy_vals.append(class_accuracy_fake)

    stats_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in stats_mean.items())
    logging.info("train metrics: " + stats_string)
    if ((epoch + 1) % (0.01 * params.num_epochs) == 0) or ((epoch + 1) <= min(10, (0.001 * params.num_epochs))):
        # Display Progress
        print("Epoch {}/{}".format(epoch + 1, params.num_epochs))
        print(stats_string)

        # Display test Images
        test_samples = g_model(test_noise, test_one_hot_v).data.cpu()
        test_samples_reshaped = gan_net.vectors_to_samples(test_samples)  # ?
        test_titles = gan_net.labels_to_titles(test_labels)

        display_results.fill_figure(test_samples_reshaped, fig, epoch + 1, args.model_dir, -1, 1,
                                    withgrayscale=True, labels=test_titles)

    return test_samples, stats_mean['d_error'] + stats_mean['g_error']


def train_gan(d_model, g_model, train_dataloader, d_optimizer, g_optimizer, r_f_loss_fn, c_loss_fn, params, model_dir):

    best_loss = np.inf
    dest_min = 0
    dest_max = 255
    curr_min = -1
    curr_max = 1

    fig = display_results.create_figure()

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        test_samples, loss_mean_sum = train(d_model, d_optimizer, g_model, g_optimizer, r_f_loss_fn, c_loss_fn,
                                            train_dataloader, params, epoch, fig)
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

            if test_samples is not None:
                np_test_samples = np.array(test_samples)
                np_test_samples = \
                    dest_min + (dest_max - dest_min) * (np_test_samples - curr_min) / (curr_max - curr_min)
                np_test_samples = np.around(np_test_samples).astype(int)
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
            np_test_labels = (test_labels.view(test_labels.shape[0], -1).cpu().numpy())

            data_path = os.path.join(model_dir, 'data')
            if not os.path.isdir(data_path):
                os.mkdir(data_path)

            test_all_data = (np.concatenate((np_test_samples, np_test_out, np_test_labels), axis=1)).tolist()
            last_csv_path = os.path.join(data_path, "samples_epoch_{}.csv".format(epoch+1))
            utils.save_incorrect_to_csv(test_all_data, last_csv_path)

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
        layer_indx = it // 2
        if it % 2:  # odd row numbers store bias vals
            display_results.plot_graph(val, None, "{}_{}_layer_bias_{}".format(n_type, v_type, layer_indx), im_path)
            print('{}_{}_layer_bias_{} graph plotted'.format(n_type, v_type, layer_indx))
        else:  # even row numbers store weight vals
            display_results.plot_graph(val, None, "{}_{}_layer_weight_{}".format(n_type, v_type, layer_indx), im_path)
            print('{}_{}_layer_weight_{} graph plotted'.format(n_type, v_type, layer_indx))

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

    d_optimizer = optim.Adam(discriminator.parameters(), lr=params.d_learning_rate, betas=(params.beta1, params.beta2))
    g_optimizer = optim.Adam(generator.parameters(), lr=params.g_learning_rate, betas=(params.beta1, params.beta2))

    # fetch loss functions
    real_fake_loss_fn = gan_net.real_fake_loss_fn
    class_selection_loss_fn = gan_net.class_selection_loss_fn

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    num_test_samples = 20
    test_noise = gan_net.noise(num_test_samples, params.noise_dim, params.noise_type)

    # test_labels = list(range(num_test_samples))
    # test_labels = [it % params.num_classes for it in test_labels]
    possible_classes = [[9, 5]]
    # possible_classes = [0, 2, 3, 4, 5, 6, 7, 9]
    test_labels = []
    for i in range(len(possible_classes)-1):
        test_labels.extend([possible_classes[i] for _ in range(num_test_samples // len(possible_classes))])
    # last class will have amount of samples to complete to num_test_samples
    test_labels.extend([possible_classes[len(possible_classes)-1]
          for _ in range(num_test_samples - (len(possible_classes)-1) * (num_test_samples // len(possible_classes)))])

    # test_labels.extend([5 for _ in range(num_test_samples//4)])
    # test_labels.extend([7 for _ in range(num_test_samples - 3*(num_test_samples//4))])

    test_labels = torch.Tensor(test_labels)
    test_labels = test_labels.type(torch.LongTensor)
    test_labels = test_labels.to(device)

    # if len(test_labels.shape) == 2:
    #     test_labels = test_labels.view(num_test_samples, -1, 2)
    # else:
    #     test_labels = test_labels.view(num_test_samples, -1)
    test_label_before = torch.index_select(test_labels, 1, torch.tensor([0], device=test_labels.device))
    test_label_after = torch.index_select(test_labels, 1, torch.tensor([1], device=test_labels.device))
    test_bef_one_hot_v = gan_net.convert_int_to_one_hot_vector(test_label_before, params.num_classes).to(device)
    test_aft_one_hot_v = gan_net.convert_int_to_one_hot_vector(test_label_after, params.num_classes).to(device)
    test_one_hot_v = torch.cat((test_bef_one_hot_v, test_aft_one_hot_v), 1)
    # test_one_hot_v = gan_net.convert_int_to_one_hot_vector(test_labels, params.num_classes).to(device)

    D_losses = []
    G_losses = []
    D_preds = []
    G_preds = []
    real_accuracy_vals = []
    fake_accuracy_vals = []
    grads_per_epoch_g = []
    grads_per_epoch_d = []
    vals_per_epoch_g = []
    vals_per_epoch_d = []

    train_gan(discriminator, generator, train_dl, d_optimizer, g_optimizer, real_fake_loss_fn, class_selection_loss_fn, params, args.model_dir)

    # track results
    print('plotting graphs')
    display_results.plot_graph(G_losses, D_losses, "Loss", args.model_dir)
    print('loss graph plotted')
    display_results.plot_graph(G_preds, D_preds, "Predictions", args.model_dir)
    print('predictions graph plotted')
    display_results.plot_graph(real_accuracy_vals, fake_accuracy_vals, "Accuracy", args.model_dir)
    print('accuracy graph plotted')

    plot_summary_graphs_layers(grads_per_epoch_g, 'G', 'Grads', args.model_dir)
    plot_summary_graphs_layers(grads_per_epoch_d, 'D', 'Grads', args.model_dir)
    plot_summary_graphs_layers(vals_per_epoch_g, 'G', 'Vals', args.model_dir)
    plot_summary_graphs_layers(vals_per_epoch_d, 'D', 'Vals', args.model_dir)