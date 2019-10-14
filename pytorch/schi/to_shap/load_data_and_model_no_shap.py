import argparse
import logging
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# import utils
import net_to_fviz as net
import one_label_data_loader_to_fviz as one_labels_data_loader
import plot_digit_utils
from evaluate_to_fviz import evaluate

import utils_shap


plt.ioff()

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default="toSync/Thesis/DL-Pytorch-data/to_shap", help='path to experiments and data folder. not for Server')
parser.add_argument('--data_dir', default='./data/gest_laws/proximity/digit-5-const-lab-change-only-horizontal', help="Directory containing the dataset")  # './data/grayscale-data'
parser.add_argument('--model_dir', default='experiments/gest_laws/proximity/debug', help="Directory containing trained network")
parser.add_argument('--restore_file', default='best',
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--all_layers', default=0, type=int,
                    help="if saving all layers output")
parser.add_argument('--focused_ind', default=5, type=int, help='specific index to show in plot')   # default=-1
parser.add_argument('--num_colors', default=8, type=int, help='')  # default=1  11


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

    # Load the parameters from json file
    args = parser.parse_args()
    if args.parent_dir and not torch.cuda.is_available():
        past_to_drive = os.environ['OneDrive']
        os.chdir(os.path.join(past_to_drive, args.parent_dir))

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

    image_path = os.path.join(args.model_dir, 'graphs')
    if not os.path.isdir(image_path):
        os.mkdir(image_path)

    data_path = os.path.join(args.model_dir, 'data')
    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    # Define the model and optimizer
    model = net.NeuralNet()

    load_model(args.model_dir, args.restore_file)

    model.eval()  # important for dropout not to work in forward pass

    batch = next(iter(train_dl))
    images, labels = batch

    # test_batch = test_dl
    # test_im, test_l = test_batch

    # test_batch = next(iter(test_dl))
    # test_im, test_l = test_batch

    # out_layer_1, out_layer_2, out_layer_3, out_layer_4, _, y_hat = model(test_im)
    out_layer_1, out_layer_2, out_layer_3, out_layer_4, y_hat_log, y_hat = model(images)

    layers_out_list = [out_layer_1, out_layer_2, out_layer_3, out_layer_4, y_hat_log, y_hat]

    num_classes = y_hat.shape[1]
    num_conds = y_hat.shape[0] // args.num_colors
    print('num_conds is: {}'.format(num_conds))
    print('num_colors is: {}'.format(args.num_colors))
    print(y_hat.shape)
    logging.info("number of conditions: {}".format(num_conds))
    logging.info("number of colors: {}".format(args.num_colors))

    test_samples_to_plot = plot_digit_utils.samples_to_images(images)
    # test_samples_to_plot = plot_digit_utils.samples_to_images(test_im)
    path = os.path.join(args.model_dir, 'test_samples')
    if num_conds > 1:
        plot_digit_utils.plot_images(test_samples_to_plot, num_rows=num_conds, path=path)
    else:
        plot_digit_utils.plot_images(test_samples_to_plot, path=path)

    if args.all_layers:
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

    if args.focused_ind in range(num_classes):

        softmax_y_hat = y_hat[:, args.focused_ind].cpu().detach().numpy()
        log_softmax_y_hat = y_hat_log[:, args.focused_ind].cpu().detach().numpy()
        last_linear_out = out_layer_4[:, args.focused_ind].cpu().detach().numpy()
        labels_list = ['softmax', 'log_softmax', 'last_linear_out']
        layers_list = [softmax_y_hat, log_softmax_y_hat, last_linear_out]
        min_y_axis_list = [round(v.min(), -1 * int(round(np.log10(np.std(v))))) for v in layers_list]
        max_y_axis_list = [round(v.max(), -1 * int(round(np.log10(np.std(v))))) for v in layers_list]
        scale_list = [-1 * int(round(np.log10(np.std(v)))) for v in layers_list]

        all_data_all_conds = [[] for _ in range(len(layers_list))]
        if args.num_colors > 1 and num_conds > 1:
            for i in range(num_conds):
                for j in range(len(layers_list)):
                    label_s = labels_list[j]
                    layer_v = layers_list[j]
                    data_v = layer_v[i*args.num_colors:(i+1)*args.num_colors]

                    all_data_all_conds[j].append(data_v)
                    # all_data_all_conds[j].extend(data_v)

                    temp_list = [v for v in range(args.num_colors*num_conds)]
                    x_idx = temp_list[i*args.num_colors:(i+1)*args.num_colors]
                    # print(x_idx)
                    filename = 'cond_{}_all_colors_ '.format(i) + label_s
                    path_v_im = os.path.join(image_path, filename)
                    path_v_dat = os.path.join(data_path, filename)

                    f = plt.figure()

                    plt.title(filename)
                    plt.xlabel("color")
                    plt.ylabel("last layer output class {}".format(args.focused_ind))
                    plt.plot(x_idx, data_v, label=label_s, marker='.')
                    plt.xticks(np.arange(i*args.num_colors, (i+1)*args.num_colors, 1))
                    # plt.xticks(np.arange(0, args.num_colors, 1))
                    plt.yticks(np.arange(min_y_axis_list[j], max_y_axis_list[j], 10**-scale_list[j]))
                    plt.tight_layout()

                    plt.savefig(path_v_im)
                    plt.close(f)

                    # if i == 0:
                    # print('saving data i= {}, j= {}'.format(i, j))
                    # print(data_v.shape)
                    # utils_shap.save_out_to_csv(data_v, path_v_dat + '.csv')

            print(np.array(all_data_all_conds).shape)
            for j in range(len(layers_list)):
                label_s = labels_list[j]
                filename = 'all_conds_all_colors_' + label_s
                path_v_im = os.path.join(image_path, filename)
                path_v_dat = os.path.join(data_path, filename)
                utils_shap.save_out_to_csv(all_data_all_conds[j], path_v_dat + '.csv')

                f = plt.figure()

                for i in range(num_conds):
                    plt.plot(np.arange(0, args.num_colors, 1), all_data_all_conds[j][i], label='cond_{}'.format(i), marker='.')

                plt.title(filename)
                plt.xlabel("color")
                plt.ylabel("last layer output class {}".format(args.focused_ind))
                # plt.xticks(np.arange(i * args.num_colors, (i + 1) * args.num_colors, 1))
                plt.xticks(np.arange(0, args.num_colors, 1))
                plt.yticks(np.arange(min_y_axis_list[j], max_y_axis_list[j], 10 ** -scale_list[j]))
                plt.legend()
                plt.tight_layout()

                plt.savefig(path_v_im)
                plt.close(f)

            for_graphs_list = [[] for _ in range(len(layers_list))]
            # conds_list = ['all_dark', 'all_bright', 'bright_dark', 'dark_bright']

            for i in range(args.num_colors):  # num_conds
                for j in range(len(layers_list)):
                    label_s = labels_list[j]
                    layer_v = layers_list[j]
                    data_v = layer_v[i::args.num_colors]

                    for_graphs_list[j].append(data_v)
                    # for_graphs_list[j].extend(data_v)
                    # print(for_graphs_list[j])

                    temp_list = [v for v in range(args.num_colors*num_conds)]
                    x_idx = temp_list[i::args.num_colors]
                    # print(x_idx)
                    filename = 'color_{}_all_conds_'.format(i) + label_s

                    path_v_im = os.path.join(image_path, filename)
                    path_v_dat = os.path.join(data_path, filename)

                    f = plt.figure()
                    plt.title(filename)
                    plt.xlabel("condition")
                    plt.ylabel("last layer output class {}".format(args.focused_ind))
                    plt.plot(x_idx, data_v, label=label_s, marker='.')
                    plt.xticks(np.arange(i, args.num_colors*num_conds, args.num_colors))
                    # plt.xticks(np.arange(0, num_conds, 1))
                    plt.yticks(
                        np.arange(min_y_axis_list[j], max_y_axis_list[j], 10 ** -scale_list[j]))
                    plt.tight_layout()

                    plt.savefig(path_v_im)
                    plt.close(f)

                    # if i == 0:
                    # print('saving data i= {}, j= {}'.format(i, j))
                    # print(data_v.shape)
                    # utils_shap.save_out_to_csv(data_v, path_v_dat + '.csv')

            print(np.array(for_graphs_list).shape)
            for j in range(len(layers_list)):
                label_s = labels_list[j]
                filename = 'all_colors_all_conds_' + label_s

                path_v_im = os.path.join(image_path, filename)
                path_v_dat = os.path.join(data_path, filename)
                utils_shap.save_out_to_csv(for_graphs_list[j], path_v_dat + '.csv')

                f = plt.figure()

                for i in range(args.num_colors):
                    plt.plot(np.arange(0, num_conds, 1), for_graphs_list[j][i], label='color_{}'.format(i), marker='*')

                plt.title(filename)
                plt.xlabel("condition")
                plt.ylabel("last layer output class {}".format(args.focused_ind))
                # for i in range(len(for_graphs_list[j])):
                #     plt.plot(for_graphs_list[j][i])
                # plt.xticks(np.arange(i, args.num_colors * num_conds, args.num_colors))
                plt.xticks(np.arange(0, num_conds, 1))
                plt.yticks(
                    np.arange(min_y_axis_list[j], max_y_axis_list[j], 10 ** -scale_list[j]))
                plt.legend()
                plt.tight_layout()
                # plt.show()

                plt.savefig(path_v_im)
                plt.close(f)

        elif num_conds > 1:
            for j in range(len(layers_list)):
                label_s = labels_list[j]
                layer_v = layers_list[j]
                data_v = layer_v
                filename = 'all_conds_single_color_' + label_s
                # path_v = os.path.join(args.model_dir, filename)
                path_v_im = os.path.join(image_path, filename)
                path_v_dat = os.path.join(data_path, filename)

                f = plt.figure()
                plt.title(filename)
                plt.xlabel("condition")
                plt.ylabel("last layer output class {}".format(args.focused_ind))
                plt.plot(data_v, label=label_s, marker='.')
                plt.xticks(np.arange(0, num_conds, 1))
                plt.yticks(
                    np.arange(min_y_axis_list[j], max_y_axis_list[j], 10 ** -scale_list[j]))
                plt.tight_layout()
                plt.savefig(path_v_im)
                plt.close(f)
                utils_shap.save_out_to_csv(data_v, path_v_dat + '.csv')

        else:
            for j in range(len(layers_list)):
                label_s = labels_list[j]
                layer_v = layers_list[j]
                data_v = layer_v
                filename = 'all_colors_single_cond_' + label_s
                # path_v = os.path.join(args.model_dir, filename)
                path_v_im = os.path.join(image_path, filename)
                path_v_dat = os.path.join(data_path, filename)

                f = plt.figure()
                plt.title(filename)
                plt.xlabel("color")
                plt.ylabel("last layer output class {}".format(args.focused_ind))
                plt.plot(data_v, label=label_s, marker='.')
                plt.xticks(np.arange(0, args.num_colors, 1))
                # plt.xticks(np.arange(0, num_conds, 1))
                plt.yticks(
                    np.arange(min_y_axis_list[j], max_y_axis_list[j], 10 ** -scale_list[j]))
                plt.tight_layout()
                plt.savefig(path_v_im)
                plt.close(f)
                utils_shap.save_out_to_csv(data_v, path_v_dat + '.csv')

    # max_vals_list = []
    # # compare only network's output
    # for i in range(len(y_hat)):

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


