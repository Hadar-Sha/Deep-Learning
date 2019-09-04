import argparse
import os
import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from torchvision.transforms import functional as F

# import utils
import net_to_shap as net
import one_label_data_loader_to_shap as one_labels_data_loader
# from evaluate_to_shap import evaluate
# from train import train

width = 1
height = 0.2
im_w = 200
im_h = 300
RGB = 3

plt.ioff()

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./experiment-data-with-gray-4000', help="Directory containing the destination dataset")
parser.add_argument('--model_dir', default='', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best',
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--reduce_background_shap', default=True, action='store_false',
                    help="if showing relative shap values to background's shap values")


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


def vectors_to_samples(vectors):
    vectors = vectors.reshape(vectors.size()[0], -1, 3)
    vectors = vectors.cpu().numpy()
    vectors = vectors.tolist()
    return vectors


def create_background(color, center=[width, 1.5*width]):

    points = np.zeros([4, 2], dtype=float)
    points[0] = [center[0]-width, center[1]-1.5*width]
    points[1] = [center[0]-width, center[1]+1.5*width]
    points[2] = [center[0]+width, center[1]+1.5*width]
    points[3] = [center[0]+width, center[1]-1.5*width]

    background = Polygon(points, True, facecolor=color)
    return background


def create_segment(segment_center, vertical_or_horizontal, color):
    points = np.zeros([6, 2], dtype=float)
    if not vertical_or_horizontal:  # segment is horizontal
        points[0, 0] = segment_center[0] - width / 2
        points[0, 1] = segment_center[1] - height / 2

        points[1, 0] = segment_center[0] + width / 2
        points[1, 1] = segment_center[1] - height / 2

        points[2, 0] = segment_center[0] + width / 2 + height / 2
        points[2, 1] = segment_center[1]

        points[3, 0] = segment_center[0] + width / 2
        points[3, 1] = segment_center[1] + height / 2

        points[4, 0] = segment_center[0] - width / 2
        points[4, 1] = segment_center[1] + height / 2

        points[5, 0] = segment_center[0] - width / 2 - height / 2
        points[5, 1] = segment_center[1]

    else:  # segment is vertical
        points[0, 0] = segment_center[0] + height / 2
        points[0, 1] = segment_center[1] - width / 2

        points[1, 0] = segment_center[0] + height / 2
        points[1, 1] = segment_center[1] + width / 2

        points[2, 0] = segment_center[0]
        points[2, 1] = segment_center[1] + width / 2 + height / 2

        points[3, 0] = segment_center[0] - height / 2
        points[3, 1] = segment_center[1] + width / 2

        points[4, 0] = segment_center[0] - height / 2
        points[4, 1] = segment_center[1] - width / 2

        points[5, 0] = segment_center[0]
        points[5, 1] = segment_center[1] - width / 2 - height / 2

    segment = Polygon(points, True, facecolor=color)
    return segment


def grayscale_to_binary(colors):
    # np_colors = np.array(colors)
    for i in range(RGB):
        for j in range(len(colors[0])):
            colors[i][j] = [1. if colors[i][j][it] == colors[i][j][7] else 0. for it in range(len(colors[0][0]))]

    return colors


def grayscale_to_white_bg(colors):
    # np_colors = np.array(colors)
    for i in range(RGB):
        for j in range(len(colors[0])):
            max_v = np.array(colors[i][j]).max()
            min_v = np.array(colors[i][j]).min()
            colors[i][j] = [1. if colors[i][j][it] == colors[i][j][7] else 1.-(max_v-min_v)
                            for it in range(len(colors[0][0]))]

    return colors


def grayscale_to_bright(colors):
    # np_colors = np.array(colors)
    for i in range(RGB):
        for j in range(len(colors[0])):
            max_v = np.array(colors[i][j]).max()
            min_v = np.array(colors[i][j]).min()
            colors[i][j] = [1. if colors[i][j][it] == max_v else 1.-(max_v-min_v)
                            for it in range(len(colors[0][0]))]

    return colors


def create_digit_image(colors, fig, curr_min_val=0, curr_max_val=1):

    fig.clear()
    width = 1
    height = 0.2
    normalized = False
    is_grayscale = False

    numpy_colors = np.array(colors)
    if len(numpy_colors.shape) == 1:
        numpy_colors = numpy_colors.reshape(numpy_colors.shape[0], 1) * np.ones([1, 3])
        is_grayscale = True

    # curr_min_val = numpy_colors.min()
    # curr_max_val = numpy_colors.max()

    # convert to [0,1] to draw
    if numpy_colors.min() < 0 or numpy_colors.max() > 1:
        if (curr_max_val - curr_min_val) <= 0:
            print('wrong min and max input values')
            return
        else:
            numpy_colors = np.round(((numpy_colors - curr_min_val) / (curr_max_val - curr_min_val)), 3)
            normalized = True

    colors = numpy_colors.tolist()

    plt.subplots_adjust(0, 0, 1, 1)
    myaxis = fig.add_subplot()
    patches = []

    myaxis.axis('off')
    myaxis.set_xlim([0, 2 * width])
    myaxis.set_ylim([0, 3 * width])
    myaxis.set_aspect('equal', 'box')

    segments_centers = []
    center = [width, 1.5*width]
    bg_patch = create_background(colors[7])
    patches.append(bg_patch)
    myaxis.add_patch(bg_patch)

    segments_centers.append([center[0], center[1] + width + height])
    segments_centers.append([center[0], center[1]])
    segments_centers.append([center[0], center[1] - width - height])
    segments_centers.append([center[0] - width / 2 - height / 2, center[1] + width / 2 + height / 2])
    segments_centers.append([center[0] + width / 2 + height / 2, center[1] + width / 2 + height / 2])
    segments_centers.append([center[0] - width / 2 - height / 2, center[1] - width / 2 - height / 2])
    segments_centers.append([center[0] + width / 2 + height / 2, center[1] - width / 2 - height / 2])

    vertical_horizon = [0, 0, 0, 1, 1, 1, 1]
    for i in range(len(segments_centers)):
        polygon = create_segment(segments_centers[i], vertical_horizon[i], colors[i])
        patches.append(polygon)
        myaxis.add_patch(polygon)

    fig.canvas.draw()
    fig.canvas.toolbar.pack_forget()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data_c = data/255.

    if is_grayscale:
        data_c = data_c[:, :, 0]

    if normalized:
        data_back = np.round(curr_min_val + data_c * (curr_max_val - curr_min_val), 3)
    else:
        data_back = np.round(data_c, 3)

    data_tensor = F.to_tensor(data_back)

    return data_tensor


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()

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

    load_model(args.model_dir, args.restore_file)

    batch = next(iter(test_dl))
    images, labels = batch

    size_of_batch = images.shape[0]
    bg_len = round(0.9 * size_of_batch)
    test_len = 20
    # test_len = min(round(0.1 * size_of_batch), 10)
    test_idx = []
    # test_idx = [[] for _ in range(test_len)]
    chosen = [False for _ in range(test_len)]

    # i = 0
    # while i < test_len:
    #     for j in range(size_of_batch):
    #         if i >= test_len:
    #             break
    #
    #         if labels[j].item() <= test_len and chosen[labels[j].item()] is False:
    #             test_idx.append(j)
    #             # test_idx[labels[j].item()].append(j)
    #             chosen[labels[j].item()] = True
    #             i += 1
    #
    # background_idx = list(set([v for v in range(size_of_batch)])-set(test_idx))
    # background = images[background_idx]
    # test_images_samples = images[test_idx]
    background = images[:bg_len]
    test_images_samples = images[bg_len: min(bg_len + 2, size_of_batch)]

    # test_images_samples = images[bg_len: min(bg_len + 4, size_of_batch)]

    e = shap.DeepExplainer(model, background)
    shap_values_samples = e.shap_values(test_images_samples)
    min_shap_val = round(np.array(shap_values_samples).min(), 3)
    max_shap_val = round(np.array(shap_values_samples).max(), 3)

    # output is in shape [batch_size,24] changing to illustrate as an image
    shap_values = []
    shap_values = [[[] for _ in range(len(shap_values_samples))] for _ in range(RGB)]

    # creating figure for converting data to image
    fig = plt.figure(figsize=(im_w, im_h), dpi=1)

    for j in range(len(shap_values_samples)):
        reshaped = vectors_to_samples(torch.tensor(shap_values_samples[j]))
        reshaped_np = np.array(reshaped)
        reshaped_splitd = [reshaped_np[:, :, i].tolist() for i in range(RGB)]
        max_bg_val = np.array(reshaped_splitd)[:, :, 7].max()

        for ind in range(RGB):
            for k in range(len(reshaped_splitd[0])):
                # change code here !!! for each reshaped_splitd[ind][k] (shape = 8) change values to be
                # newval = val- bgval (reshaped_splitd[ind][k][7])

                if args.reduce_background_shap:
                    bg_val = reshaped_splitd[ind][k][7]
                    reshaped_splitd[ind][k] = [it - bg_val for it in reshaped_splitd[ind][k]]
                else:
                    bg_val = 0
                tensor_image = create_digit_image(reshaped_splitd[ind][k], fig, min_shap_val - bg_val,
                                                      max_shap_val - bg_val)

                image_s = tensor_image.numpy()
                shap_values[ind][j].append(image_s)

    test_images = [torch.empty(len(test_images_samples), 1, im_h, im_w) for _ in range(RGB)]
    reshaped_test = vectors_to_samples(test_images_samples.clone().detach())

    ts_images = torch.empty(len(test_images_samples), RGB, im_h, im_w)
    for i in range(len(reshaped_test)):
        ts_images[i] = create_digit_image(reshaped_test[i], fig)

    reshaped_test_np = np.array(reshaped_test)
    reshaped_test_splitd = [reshaped_test_np[:, :, i].tolist() for i in range(RGB)]
    reshaped_test_splitd = grayscale_to_bright(reshaped_test_splitd)  # added by Hadar!!!!!
    # reshaped_test_splitd = grayscale_to_white_bg(reshaped_test_splitd)  # added by Hadar!!!!!
    # reshaped_test_splitd = grayscale_to_binary(reshaped_test_splitd)  # added by Hadar!!!!!

    for id in range(RGB):
        for k in range(len(reshaped_test_splitd[0])):
            test_images[id][k] = create_digit_image(reshaped_test_splitd[id][k], fig, min_shap_val, max_shap_val)

    # closing temp figure
    plt.close(fig)

    shap_numpy_splitd = [[np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values[i]] for i in range(RGB)]
    shap_numpy_stackd = list(np.concatenate((shap_numpy_splitd[0], shap_numpy_splitd[1], shap_numpy_splitd[2]), axis=2))

    test_numpy = [np.swapaxes(np.swapaxes(test_images[i].numpy(), 1, -1), 1, 2) for i in range(RGB)]
    test_numpy_stackd = np.concatenate((test_numpy[0], test_numpy[1], test_numpy[2]), axis=1)

    a = 1
    ts_images = np.swapaxes(np.swapaxes(ts_images.numpy(), 1, -1), 1, 2)

    # plt.ion()
    plt.show()
    for i in range(len(ts_images)):
        plt.figure(i+1)
        plt.imshow(ts_images[i])
        # plt.show()
        plt.draw()

    # print(np.array(shap_numpy_stackd).shape)

    # labels.shape[0] == shap_values[0].shape[0]
    # print(np.matmul(np.ones([2,1]),np.arange(len(shap_numpy_stackd)).reshape([1, len(shap_numpy_stackd)])).shape[0])
    # print(shap_numpy_stackd[0].shape[0])
    #
    # # labels.shape[1] == len(shap_values)
    # print(np.matmul(np.ones([2,1]),np.arange(len(shap_numpy_stackd)).reshape([1, len(shap_numpy_stackd)])).shape[1])
    # print(len(shap_numpy_stackd))
    labels_for_plot = np.matmul(np.ones([len(shap_numpy_stackd[0]), 1], dtype=int),
                                np.arange(len(shap_numpy_stackd)).reshape([1, len(shap_numpy_stackd)]))

    shap.image_plot(shap_numpy_stackd, test_numpy_stackd, labels_for_plot)

    # unioned_test_numpy = np.concatenate((test_numpy[0], test_numpy[1], test_numpy[2]), axis=3)

    # for i in range(len(unioned_test_numpy)):
    #     # plt.imshow(unioned_test_numpy[i])
    #     # plt.show()
    #     plt.imshow(unioned_test_numpy[i], cmap=plt.get_cmap('gray'))
    #     plt.show()
    #
    # for i in range(len(test_numpy_stackd)):
    #     np_im = np.array(test_numpy_stackd[i])
    #     np_im = np_im.reshape(np_im.shape[:2])
    #     # np_im = np.concatenate((np_im, np_im, np_im), axis=2)
    #     plt.imshow(np_im, cmap=plt.get_cmap('gray'))
    #     plt.show()

    # plot the feature attributions
    # for i in range(RGB):
    #
    #     shap.image_plot(shap_numpy_splitd[i], test_numpy[i])

    # shap.image_plot(shap_numpy, -test_numpy)

