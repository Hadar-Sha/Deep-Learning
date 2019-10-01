import argparse
import logging
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# import utils
import net_to_shap as net
import one_label_data_loader_to_shap as one_labels_data_loader
import plot_digit_utils
from evaluate_to_shap import evaluate

import utils_shap


plt.ioff()
#
# RGB = 3

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default="toSync/Thesis/DL-Pytorch-data/to_shap", help='path to experiments and data folder. not for Server')
parser.add_argument('--data_dir', default='./data/grayscale-data', help="Directory containing the destination dataset")
parser.add_argument('--model_dir', default='experiments/debug', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best',
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--reduce_background_shap', default=0, type=int,
                    help="if showing relative shap values to background's shap values")
parser.add_argument('--change_test_shap', default=0, type=int, help='how to modify background')
parser.add_argument('--focused_ind', default=-1, type=int, help='specific index to show in plot')


# './experiment-data-with-gray-4000' # './grayscale-data'
def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def load_model(model_dir, restore_file):
    # reload weights from restore_file if specified
    if restore_file is not None and model_dir is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        print("Restoring parameters from {}".format(restore_path))
        load_checkpoint(restore_path, model, None)
    return


if __name__ == '__main__':

    print(os.getcwd())

    # Load the parameters from json file
    args = parser.parse_args()
    if args.parent_dir and not torch.cuda.is_available():
        past_to_drive = os.environ['OneDrive']
        os.chdir(os.path.join(past_to_drive, args.parent_dir))

    print(os.getcwd())
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils_shap.Params(json_path)

    # Set the logger
    utils_shap.set_logger(os.path.join(args.model_dir, 'analyze.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    dataloaders = one_labels_data_loader.fetch_dataloader(['train', 'test'], args.data_dir)
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']

    logging.info("data was loaded from {}".format(args.data_dir))
    logging.info("- done.")

    num_of_batches = max(1, len(train_dl.dataset) // train_dl.batch_size)
    logging.info("data-set size: {}".format(len(train_dl.dataset)))
    logging.info("number of batches: {}".format(num_of_batches))

    # Define the model and optimizer
    model = net.NeuralNet()

    load_model(args.model_dir, args.restore_file)

    # batch = next(iter(train_dl))
    # images, labels = batch

    test_batch = next(iter(test_dl))
    test_im, test_l = test_batch

    # size_of_batch = images.shape[0]
    # bg_len = size_of_batch

    out_layer_1, out_layer_2, out_layer_3, _, y_hat = model(test_im)

    layers_out_list = [out_layer_1, out_layer_2, out_layer_3, y_hat]

    num_classes = y_hat.shape[1]

    for ind in range(len(layers_out_list)):
        val_to_hist = layers_out_list[ind].cpu().detach().numpy()
        # nonzeros_vals_to_hist = val_to_hist[np.nonzero(val_to_hist)]
        dif_vals = np.zeros(val_to_hist.shape)
        stats = np.zeros((val_to_hist.shape[0], 4))

        for i in range(val_to_hist.shape[0]):
            dif_vals[i] = val_to_hist[i]-val_to_hist[5]
            stats[i] = (val_to_hist[i].min(), np.mean(val_to_hist[i]), np.median(val_to_hist[i]), val_to_hist[i].max())

        # plt.hist(val_to_hist[i])
        path = os.path.join(args.model_dir, 'out_layer_{}_dif_gray.csv'.format(ind))
        utils_shap.save_out_to_csv(dif_vals, path)

    labels = list(range(val_to_hist.shape[0]))
    labels = [str(v) for v in labels]

    plt.plot(stats)

    # fetch loss function and metrics
    loss_fn = net.loss_fn

    metrics = net.metrics
    incorrect = net.incorrect
    num_epochs = 10000

    test_metrics, incorrect_samples = evaluate(model, loss_fn, test_dl, metrics, incorrect, num_epochs - 1)

    # print(dif_list)
    # plt.hist(val_to_hist, label=labels)
    # for i in range(val_to_hist.shape[0]):
    #     plt.hist(val_to_hist[i]-val_to_hist[5])

    # if args.focused_ind in range(num_classes):
    #     class_vals_to_hist = val_to_hist[:, args.focused_ind]
    #     plt.hist(class_vals_to_hist)
    # plt.hist(val_to_hist)

    # print(type(out_lay_1))
    #
    # print(type(out_lay_2))
    # print(type(out_lay_3))
    # print(type(y_hat))

    # bg_samples_to_plot = plot_digit_utils.samples_to_images(images)
    # plot_digit_utils.plot_images(bg_samples_to_plot, 'background')

    test_samples_to_plot = plot_digit_utils.samples_to_images(test_im)
    plot_digit_utils.plot_images(test_samples_to_plot, 'test')
