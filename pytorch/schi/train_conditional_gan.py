"""Train the model"""

import argparse
import logging
import os

import torch
import torch.optim as optim
from torch.autograd import Variable
# from tqdm import tqdm

from utils_gan import Logger
import display_digit as display_results
import utils
import model_gan.conditional_gan_net as gan_net
# import model_gan.two_labels_data_loader as data_loader
import model_gan.one_label_data_loader as data_loader
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/data-with-grayscale-4000', help="Directory containing the dataset")
# data-two-labels-big # grayscale-logits # data/data-w-gray-only-2 data/data-with-grayscale-4000
parser.add_argument('--model_dir', default='experiments/cgan_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def train_discriminator(d, optimizer, real_data, fake_data, real_labels, fake_labels):
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = d(real_data, real_labels)
    # Calculate error and backpropagate
    error_real = loss_fn(prediction_real, gan_net.real_data_target(real_data.size(0)))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = d(fake_data, fake_labels)
    # Calculate error and backpropagate
    error_fake = loss_fn(prediction_fake, gan_net.fake_data_target(real_data.size(0)))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error0
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(d, optimizer, fake_data, fake_labels):
    # 2. Train Generator
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = d(fake_data, fake_labels)
    # Calculate error and backpropagate
    error = loss_fn(prediction, gan_net.real_data_target(prediction.size(0)))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error


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
        fig:
    """

    logger = Logger(model_name='CGAN', data_name='SCHI')

    # fig = display_results.create_figure()

    # Use tqdm for progress bar
    # with tqdm(total=len(dataloader)) as t:
    for i, (real_batch, real_label) in enumerate(dataloader):

        # 1. Train Discriminator
        real_data = Variable(real_batch)
        real_label = Variable(real_label)

        if torch.cuda.is_available():
            real_data = real_data.cuda()
            real_label = real_label.cuda()

        # Generate fake data
        noisy_input = gan_net.noise(real_data.size(0), params.noise_dim)
        # noisy_label = gan_net.create_random_labels(real_data.size(0))

        # noisy_label = Variable(2*torch.ones((real_data.size(0),), dtype=torch.int))
        noisy_label = Variable(torch.randint(params.num_classes, (real_data.size(0),)))
        noisy_label = noisy_label.type(torch.LongTensor)

        test_samples = None

        if not params.is_one_hot:
            if real_label.size(1) == 1:
                real_label = real_label.view(real_label.size(0))

            fake_data = g_model(noisy_input, noisy_label).detach()
            # Train D
            d_error, d_pred_real, d_pred_fake = train_discriminator(d_model, d_optimizer, real_data, fake_data,
                                                                    real_label, noisy_label)
            # 2. Train Generator
            fake_data = g_model(noisy_input, noisy_label)
            # Train G
            g_error = train_generator(d_model, g_optimizer, fake_data, noisy_label)

        else:
            real_one_hot_v = gan_net.convert_int_to_one_hot_vector(real_label, params.num_classes)

            noisy_label = noisy_label.view(real_data.size(0), -1)
            noisy_one_hot_v = gan_net.convert_int_to_one_hot_vector(noisy_label, params.num_classes)

            fake_data = g_model(noisy_input, noisy_one_hot_v).detach()

            # Train D
            d_error, d_pred_real, d_pred_fake = train_discriminator(d_model, d_optimizer, real_data, fake_data,
                                                                    real_one_hot_v, noisy_one_hot_v)

            # 2. Train Generator
            fake_data = g_model(noisy_input, noisy_one_hot_v)
            # Train G
            g_error = train_generator(d_model, g_optimizer, fake_data, noisy_one_hot_v)

        # Log error
        logger.log(d_error, g_error, epoch, i+1, i+1)  # num_batches)

        # Save Losses for plotting later
        G_losses.append(d_error.item())
        D_losses.append(g_error.item())

        G_preds.append(d_pred_fake.data.mean())
        D_preds.append(d_pred_real.data.mean())

    # Display Progress
    if (epoch+1) % (0.01*params.num_epochs) == 0:

        # Display Images
        if not params.is_one_hot:
            test_samples = g_model(test_noise, test_labels).data.cpu()
        else:
            test_samples = g_model(test_noise, test_one_hot_v).data.cpu()

        test_samples_reshaped = gan_net.vectors_to_samples(test_samples)  # ?

        # fig1, axes1 = display_results.create_grid(num_test_samples)
        # display_results.fill_grid(test_samples, fig1, axes1, epoch, i+1)

        display_results.fill_figure(test_samples_reshaped, fig, gan_net.labels_to_titles(test_labels))
        # Display status Logs
        logger.display_status(
            epoch+1, params.num_epochs, i+1, i+1,
            d_error, g_error, d_pred_real, d_pred_fake)

    return test_samples


def train_gan(d_model, g_model, train_dataloader, dev_dataloader, d_optimizer, g_optimizer, loss_fn, params, model_dir,
                       restore_file=None):

    fig = display_results.create_figure()
    # fig = None

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        test_samples = train(d_model, g_model, d_optimizer, g_optimizer, loss_fn, train_dataloader, params, epoch, fig)

        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': d_model.state_dict(),
                               'optim_dict': d_optimizer.state_dict()}, is_best=False, checkpoint=model_dir, ntype='d')

        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': g_model.state_dict(),
                               'optim_dict': g_optimizer.state_dict()}, is_best=False, checkpoint=model_dir, ntype='g')

        # a=1
        if test_samples is not None:
            np_test_samples = np.array(test_samples)
            np_test_samples = np.around(np_test_samples * 127.5 + 127.5).astype(int)
            np_test_out = (test_noise.numpy())  # .tolist()
            np_test_labels = (test_labels.view(test_labels.shape[0], -1).numpy())
            # print(np_test_samples.shape)
            # print(np_test_labels.shape)
            test_all_data = (np.concatenate((np_test_samples, np_test_out, np_test_labels), axis=1)).tolist()
            last_csv_path = os.path.join(model_dir, "samples_epoch_{}.csv".format(epoch+1))
            utils.save_incorrect_to_csv(test_all_data, last_csv_path)

    return


def collect_network_statistics(net):

    status_net = []
    net_grads_graph = []
    for param_tensor in net.state_dict():

        status_net.append([param_tensor,
                                       (net.state_dict()[param_tensor].norm()).item(),
                                       list(net.state_dict()[param_tensor].size())])

        all_net_grads = ((net.state_dict()[param_tensor]).numpy()).tolist()
        # if needed, flatten the list to get one nim and one max
        if isinstance(all_net_grads, (list,)):
            flat_net_grads = []
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
    discriminator = gan_net.DiscriminatorNet(params)
    generator = gan_net.GeneratorNet(params)
    if torch.cuda.is_available():
        gan_net.discriminator.cuda()
        gan_net.generator.cuda()

    discriminator.apply(gan_net.weights_init)
    generator.apply(gan_net.weights_init)

    print(discriminator)
    print(generator)

    # Optimizers
    d_optimizer = optim.Adam(discriminator.parameters(), lr=params.d_learning_rate, betas=(params.beta1, params.beta2))
    g_optimizer = optim.Adam(generator.parameters(), lr=params.g_learning_rate, betas=(params.beta1, params.beta2))
    # d_optimizer = optim.SGD(discriminator.parameters(), lr=params.learning_rate)
    # g_optimizer = optim.SGD(generator.parameters(), lr=params.learning_rate)

    # fetch loss function
    loss_fn = gan_net.loss_fn

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    num_test_samples = 20
    test_noise = gan_net.noise(num_test_samples, params.noise_dim)

    # test_labels = gan_net.create_random_labels(num_test_samples)
    # test_labels = Variable(torch.randint(params.num_classes, (num_test_samples,)))
    test_labels = list(range(num_test_samples))
    test_labels = [it % params.num_classes for it in test_labels]
    # test_labels = [2 for _ in range(num_test_samples)]
    test_labels = torch.Tensor(test_labels)
    test_labels = test_labels.type(torch.LongTensor)

    if params.is_one_hot:
        test_labels = test_labels.view(num_test_samples, -1)
        test_one_hot_v = gan_net.convert_int_to_one_hot_vector(test_labels, params.num_classes)

    D_losses = []
    G_losses = []
    D_preds = []
    G_preds = []

    train_gan(discriminator, generator, train_dl, dev_dl, d_optimizer, g_optimizer, loss_fn, params, args.model_dir,
              args.restore_file)

    # track results
    display_results.plot_graph(G_losses, D_losses, "Loss")
    display_results.plot_graph(G_preds, D_preds, "Predictions")

    # status_discriminator = []
    # status_generator = []
    # d_grads_graph = []
    # g_grads_graph = []

    d_grads_graph = collect_network_statistics(discriminator)
    g_grads_graph = collect_network_statistics(generator)

    # for param_tensor in discriminator.state_dict():
    #
    #     status_discriminator.append([param_tensor,
    #                                    (discriminator.state_dict()[param_tensor].norm()).item(),
    #                                    list(discriminator.state_dict()[param_tensor].size())])
    #
    #     all_d_grads = ((discriminator.state_dict()[param_tensor]).numpy()).tolist()
    #     # if needed, flatten the list to get one nim and one max
    #     if isinstance(all_d_grads, (list,)):
    #         flat_d_grads = []
    #         for elem in all_d_grads:
    #             if isinstance(elem, (list,)) and isinstance(elem[0], (list,)):
    #                 for item in elem:
    #                     flat_d_grads.extend(item)
    #             elif isinstance(elem, (list,)):
    #                 flat_d_grads.extend(elem)
    #             else:
    #                 flat_d_grads.extend([elem])
    #     else:
    #         flat_d_grads = all_d_grads
    #
    #     d_grads_graph.append([min(flat_d_grads), max(flat_d_grads)])
    #
    # for param_tensor in generator.state_dict():
    #     status_generator.append([param_tensor,
    #                                  (generator.state_dict()[param_tensor].norm()).item(),
    #                                  list(generator.state_dict()[param_tensor].size())])
    #     all_g_grads = ((generator.state_dict()[param_tensor]).numpy()).tolist()
    #     # if needed, flatten the list to get one nim and one max
    #
    #     if isinstance(all_d_grads, (list,)):
    #         flat_g_grads = []
    #         for elem in all_g_grads:
    #             if isinstance(elem, (list,)) and isinstance(elem[0], (list,)):
    #                 for item in elem:
    #                     flat_g_grads.extend(item)
    #             elif isinstance(elem, (list,)):
    #                 flat_g_grads.extend(elem)
    #             else:
    #                 flat_g_grads.extend([elem])
    #     else:
    #         flat_g_grads = all_g_grads
    #
    #     g_grads_graph.append([min(flat_g_grads), max(flat_g_grads)])

    display_results.plot_graph(g_grads_graph, d_grads_graph, "Grads")
