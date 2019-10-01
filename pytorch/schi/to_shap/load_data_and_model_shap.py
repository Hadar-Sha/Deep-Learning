import argparse
import os
import torch
import numpy as np
import shap
import matplotlib
import matplotlib.pyplot as plt

# import utils
import net_to_shap as net
import one_label_data_loader_to_shap as one_labels_data_loader
from evaluate_to_shap import evaluate
import plot_digit_utils
# from train import train

im_w = 200
im_h = 300
RGB = 3

plt.ioff()

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default="toSync/Thesis/DL-Pytorch-data/to_shap", help='path to experiments and data folder. not for Server')
parser.add_argument('--data_dir', default='./data/grayscale-data', help="Directory containing the destination dataset")
parser.add_argument('--model_dir', default='', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best',
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--reduce_background_shap', default=0, type=int,
                    help="if showing relative shap values to background's shap values")
parser.add_argument('--change_test_shap', default=0, type=int, help='how to modify background')
parser.add_argument('--focused_ind', default=-1, type=int, help='specific index to show in plot')
parser.add_argument('--grad_exp', default=0, type=int, help='')
parser.add_argument('--plot_colored', default=0, type=int, help='')
parser.add_argument('--all_classes', default=0, type=int, help='')


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
        load_checkpoint(restore_path, model, None)  # optimizer)
    return


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    if args.parent_dir and not torch.cuda.is_available():
        past_to_drive = os.environ['OneDrive']
        os.chdir(os.path.join(past_to_drive, args.parent_dir))

    # Create the input data pipeline
    print("Loading the datasets...")

    # fetch dataloaders
    dataloaders = one_labels_data_loader.fetch_dataloader(['train', 'test'], args.data_dir)
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']

    print("data was loaded from {}".format(args.data_dir))
    print("- done.")

    # Define the model and optimizer
    model = net.NeuralNet()

    print(model)

    load_model(args.model_dir, args.restore_file)

    batch = next(iter(train_dl))
    images, labels = batch

    test_batch = next(iter(test_dl))
    test_im, test_l = test_batch

    size_of_batch = images.shape[0]
    bg_len = size_of_batch

    if args.all_classes:
        # test_len = 20
        test_len = min(round(0.1 * size_of_batch), 10)
        test_idx = []
        # test_idx = [[] for _ in range(test_len)]
        chosen = [False for _ in range(test_len)]

        i = 0
        while i < test_len:
            for j in range(size_of_batch):
                if i >= test_len:
                    break

                if labels[j].item() <= test_len and chosen[labels[j].item()] is False:
                    test_idx.append(j)
                    chosen[labels[j].item()] = True
                    i += 1

        background_idx = list(set([v for v in range(size_of_batch)])-set(test_idx))
        background = images[background_idx]
        test_images_samples = images[test_idx]

    else:
        background = images[:bg_len]
        test_images_samples = test_im[0: 2]

    e = shap.DeepExplainer(model, background)
    shap_values_samples = e.shap_values(test_images_samples)

    min_shap_val = round(np.array(shap_values_samples).min(), 3)
    max_shap_val = round(np.array(shap_values_samples).max(), 3)
    abs_min_max_val = max(abs(min_shap_val), abs(max_shap_val))

    print('min shap val is: {}'.format(min_shap_val))
    print('max shap val is: {}'.format(max_shap_val))

    # output is in shape [batch_size,24] changing to illustrate as an image
    shap_values = []
    shap_values = [[[] for _ in range(len(shap_values_samples))] for _ in range(RGB)]

    # creating figure for converting data to image
    fig = plt.figure(figsize=(im_w, im_h), dpi=1)

    for j in range(len(shap_values_samples)):
        reshaped = plot_digit_utils.vectors_to_samples(torch.tensor(shap_values_samples[j]))
        reshaped_np = np.array(reshaped)
        reshaped_splitd = [reshaped_np[:, :, i].tolist() for i in range(RGB)]
        max_bg_val = np.array(reshaped_splitd)[:, :, 7].max()

        for ind in range(RGB):
            for k in range(len(reshaped_splitd[0])):

                if args.reduce_background_shap:
                    bg_val = reshaped_splitd[ind][k][7]
                else:
                    bg_val = 0
                reshaped_splitd[ind][k] = [it - bg_val for it in reshaped_splitd[ind][k]]
                tensor_image = plot_digit_utils.create_digit_image(reshaped_splitd[ind][k], fig, min_shap_val - bg_val,
                                                      max_shap_val - bg_val, normalized_color=1)  # normalized_color=0)  #

                image_s = tensor_image.numpy()
                shap_values[ind][j].append(image_s)

    test_images = [torch.empty(len(test_images_samples), 1, im_h, im_w) for _ in range(RGB)]

    reshaped_test = plot_digit_utils.vectors_to_samples(test_images_samples.clone().detach())

    # added by Hadar!!!!!
    reshaped_test_np = np.array(reshaped_test)
    reshaped_test_splitd = [reshaped_test_np[:, :, i].tolist() for i in range(RGB)]
    if args.change_test_shap > 0:
        if args.change_test_shap == 1:
            reshaped_test_splitd = plot_digit_utils.grayscale_to_bright(reshaped_test_splitd)
        elif args.change_test_shap == 2:
            reshaped_test_splitd = plot_digit_utils.grayscale_to_white_bg(reshaped_test_splitd)
        elif args.change_test_shap == 3:
            reshaped_test_splitd = plot_digit_utils.grayscale_to_binary(reshaped_test_splitd)

    for id in range(RGB):
        for k in range(len(reshaped_test_splitd[0])):
            test_images[id][k] = plot_digit_utils.create_digit_image(reshaped_test_splitd[id][k], fig, min_shap_val, max_shap_val)

    # closing temp figure
    plt.close(fig)

    shap_numpy_splitd = [[np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values[i]] for i in range(RGB)]
    shap_numpy_stackd = list(np.concatenate((shap_numpy_splitd[0], shap_numpy_splitd[1], shap_numpy_splitd[2]), axis=2))

    test_numpy = [np.swapaxes(np.swapaxes(test_images[i].numpy(), 1, -1), 1, 2) for i in range(RGB)]
    test_numpy_stackd = np.concatenate((test_numpy[0], test_numpy[1], test_numpy[2]), axis=1)

    # fetch loss function and metrics
    loss_fn = net.loss_fn

    metrics = net.metrics
    incorrect = net.incorrect
    num_epochs = 10000

    test_metrics, incorrect_samples = evaluate(model, loss_fn, test_dl, metrics, incorrect, num_epochs - 1)

    to_view_bg_images = plot_digit_utils.samples_to_images(images)
    to_view_test_im = plot_digit_utils.samples_to_images(test_im)

    plot_digit_utils.plot_images(to_view_bg_images, 'background')
    plot_digit_utils.plot_images(to_view_test_im, 'test')

    num_classes = len(shap_numpy_stackd)

    if args.focused_ind in range(num_classes):
        print(args.focused_ind)

        shap_temp_for_print = np.array(shap_values_samples[args.focused_ind])
        print('min shap val of single color sample is: {0:.3f}'.format(shap_temp_for_print[0].min()))
        print('max shap val of single color sample is: {0:.3f}'.format(shap_temp_for_print[0].max()))
        print('min shap val of two colors sample is: {0:.3f}'.format(shap_temp_for_print[1].min()))
        print('max shap val of two colors sample is: {0:.3f}'.format(shap_temp_for_print[1].max()))

        print("single color sample's bg's shap values is {}".format(shap_temp_for_print[0, 21:]))
        print("two colors sample's bg's shap values is {}".format(shap_temp_for_print[1, 21:]))

        shap.image_plot(shap_numpy_stackd[args.focused_ind], test_numpy_stackd)
    else:
        labels_for_plot = np.matmul(np.ones([len(shap_numpy_stackd[0]), 1], dtype=int),
                                    np.arange(num_classes).reshape([1, num_classes]))

        shap.image_plot(shap_numpy_stackd, test_numpy_stackd, labels_for_plot)

        # white_temp = np.ones(test_numpy_stackd.shape)
        # black_temp = np.zeros(test_numpy_stackd.shape)

        # shap.image_plot(shap_numpy_stackd, black_temp, labels_for_plot)
        # shap.image_plot(shap_numpy_stackd, white_temp, labels_for_plot)

    if args.grad_exp:
        e = shap.GradientExplainer(model, model.fc4)
        shap_values, indexes = e.shap_values(test_images_samples)

        # plot the explanations
        shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]

        shap.image_plot(shap_values, test_images_samples, labels_for_plot)

    if args.plot_colored:
        unioned_test_numpy = np.concatenate((test_numpy[0], test_numpy[1], test_numpy[2]), axis=3)

        for i in range(len(unioned_test_numpy)):
            plt.imshow(unioned_test_numpy[i], cmap=plt.get_cmap('gray'))
            plt.show()

        for i in range(len(test_numpy_stackd)):
            np_im = np.array(test_numpy_stackd[i])
            np_im = np_im.reshape(np_im.shape[:2])
            plt.imshow(np_im, cmap=plt.get_cmap('gray'))
            plt.show()

