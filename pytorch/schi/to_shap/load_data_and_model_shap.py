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

plt.ioff()

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./experiment-data-with-gray-4000', help="Directory containing the destination dataset")
parser.add_argument('--model_dir', default='', help="Directory containing params.json")
parser.add_argument('--restore_file', default='best',
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


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


def create_digit_image(colors, fig):  # , curr_min_val=0, curr_max_val=1):

    fig.clear()
    width = 1
    height = 0.2
    normalized = False
    is_grayscale = False

    numpy_colors = np.array(colors)
    if len(numpy_colors.shape) == 1:
        numpy_colors = numpy_colors.reshape(numpy_colors.shape[0], 1) * np.ones([1, 3])
        is_grayscale = True

    curr_min_val = numpy_colors.min()
    curr_max_val = numpy_colors.max()

    # convert to [0,1] to draw
    if numpy_colors.min() < 0 or numpy_colors.max() > 1:
        if (curr_max_val - curr_min_val) <= 0:
            print('wrong min and max input values')
            return
        else:
            numpy_colors = np.round(((numpy_colors - curr_min_val) / (curr_max_val - curr_min_val)), 3)
            normalized = True

    colors = numpy_colors.tolist()
    # data_tensor = torch.empty(3, 300, 200, dtype=torch.float)

    # fig_temp, myaxis = plt.subplots(figsize=(im_w, im_h), dpi=1)
    # fig_temp.clear()
    # fig_temp.subplots_adjust(0, 0, 1, 1)
    plt.subplots_adjust(0, 0, 1, 1)
    myaxis = fig.add_subplot()
    # myaxis = fig.axes
    patches = []

    myaxis.axis('off')
    myaxis.set_xlim([0, 2 * width])
    myaxis.set_ylim([0, 3 * width])
    myaxis.set_aspect('equal', 'box')
    # fig_temp.tight_layout()

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
    # fig_temp.canvas.draw()

    # fig_temp.canvas.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    # data = np.fromstring(fig_temp.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    data_c = data/255.

    if is_grayscale:
        data_c = data_c[:,:,0]

    if normalized:
        data_back = np.round(curr_min_val + data_c * (curr_max_val - curr_min_val), 3)  # .astype(int)
    else:
        data_back = np.round(data_c, 3)

    data_tensor = F.to_tensor(data_back)

    # fig_temp.clf()
    # plt.show()
    # plt.pause(1)
    # plt.close('all')
    # plt.close(fig_temp)
    # print(data_tensor.shape)

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

    # # get training samples and labels
    # data = train_dl.dataset.images
    # labels = train_dl.dataset.digit_labels

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
    test_images_samples = images[bg_len: min(bg_len + 4, size_of_batch)]

    e = shap.DeepExplainer(model, background)
    shap_values_samples = e.shap_values(test_images_samples)
    # shap_values_samples_red

    # output is in shape [batch_size,24] changing to illustrate as an image
    shap_values = []
    shap_values = [[[] for _ in range(len(shap_values_samples))] for _ in range(3)]
    # shap_values = [[] for _ in range(len(shap_values_samples))]

    # creating figure for converting data to image
    fig = plt.figure(figsize=(im_w, im_h), dpi=1)

    print(len(shap_values_samples))
    for j in range(len(shap_values_samples)):
        reshaped = vectors_to_samples(torch.tensor(shap_values_samples[j]))
        reshaped_np = np.array(reshaped)
        reshaped_splitd = [reshaped_np[:, :, i].tolist() for i in range(3)]
        # reshaped_red = reshaped_np[:, :, 0].tolist()

        for ind in range(3):
            for k in range(len(reshaped_splitd[0])):
                tensor_image = create_digit_image(reshaped_splitd[ind][k], fig)
                image = tensor_image.numpy()
                shap_values[ind][j].append(image)

    # test_images = torch.empty(len(test_images_samples), 8, 3)
    test_images = [torch.empty(len(test_images_samples), 1, im_h, im_w) for _ in range(3)]  #, dtype=torch.int)
    # test_images = torch.empty(len(test_images_samples), 3, im_h, im_w)  #, dtype=torch.int)
    reshaped_test = vectors_to_samples(test_images_samples.clone().detach())
    reshaped_test_np = np.array(reshaped_test)
    reshaped_test_splitd = [reshaped_test_np[:, :, i].tolist() for i in range(3)]

    for id in range(3):
        for k in range(len(reshaped_test_splitd[0])):
            test_images[id][k] = create_digit_image(reshaped_test_splitd[id][k], fig)

    # closing temp figure
    plt.close(fig)

    # shap_numpy_blue = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values[2]]
    shap_numpy_splitd = [[np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values[i]] for i in range(3)]
    shap_numpy_stackd = list(np.concatenate((shap_numpy_splitd[0], shap_numpy_splitd[1], shap_numpy_splitd[2]), axis=2))

    test_numpy = [np.swapaxes(np.swapaxes(test_images[i].numpy(), 1, -1), 1, 2) for i in range(3)]
    test_numpy_stackd = np.concatenate((test_numpy[0], test_numpy[1], test_numpy[2]), axis=1)
    # test_numpy_stackd = test_numpy_stackd[0]

    shap.image_plot(shap_numpy_stackd, test_numpy_stackd)

    # # plot the feature attributions
    # for i in range(3):
    #
    #     shap.image_plot(shap_numpy_splitd[i], test_numpy[i])
    # shap.image_plot(shap_numpy_red, test_numpy)

    # shap.image_plot(shap_numpy, -test_numpy)

    # # fetch loss function and metrics
    # loss_fn = net.loss_fn
    #
    # metrics = net.metrics
    # incorrect = net.incorrect
    # num_epochs = 10000
    #
    # test_metrics, incorrect_samples = evaluate(model, loss_fn, test_dl, metrics, incorrect, num_epochs - 1)

