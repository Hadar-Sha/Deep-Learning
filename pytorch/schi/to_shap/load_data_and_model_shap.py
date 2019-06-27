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
from evaluate_to_shap import evaluate
# from train import train

width = 1
height = 0.2
im_w = 20
im_h = 30

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./experiment-data-with-gray-4000', help="Directory containing the destination dataset")
parser.add_argument('--model_dir', default='', help="Directory containing params.json")
parser.add_argument('--restore_file', default='fully_connected_best',
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
    checkpoint = torch.load(checkpoint)
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


def create_digit_image(colors, curr_min_val=0, curr_max_val=1):

    width = 1
    height = 0.2

    numpy_colors = np.array(colors)

    # convert to [0,1] to draw
    if numpy_colors.min() < 0 or numpy_colors.max() > 1:
        if (curr_max_val - curr_min_val) <= 0:
            print('wrong min and max input values')
            return
        else:
            numpy_colors = np.round(((numpy_colors - curr_min_val) / (curr_max_val - curr_min_val)), 3)

    colors = numpy_colors.tolist()

    # if len(numpy_colors.shape) < 2 or len(numpy_colors.shape) > 3:
    #     print('wrong input dimention. should be [_,8,3]')
    # elif len(numpy_colors.shape) == 2:
    #     colors = [colors]
    # # elif len(numpy_colors.shape)==3:
    # #   do noting
    # data_tensor = torch.empty(numpy_colors.shape[0], 3, 300, 200, dtype=torch.float)
    #
    # for j in range(numpy_colors.shape[0]):

    fig_temp, myaxis = plt.subplots(figsize=(im_w, im_h), dpi=1)
    fig_temp.subplots_adjust(0, 0, 1, 1)
    patches = []

    myaxis.axis('off')
    myaxis.set_xlim([0, 2 * width])
    myaxis.set_ylim([0, 3 * width])
    myaxis.set_aspect('equal', 'box')

    segments_centers = []
    center = [width, 1.5*width]
    bg_patch = create_background(colors[7])
    # bg_patch = create_background(colors[j][7])
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
        # polygon = create_segment(segments_centers[i], vertical_horizon[i], colors[j][i])
        patches.append(polygon)
        myaxis.add_patch(polygon)

    fig_temp.canvas.draw()

    data = np.fromstring(fig_temp.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig_temp.canvas.get_width_height()[::-1] + (3,))
    data_c = data/255.
    # data_c = data.reshape(fig_temp.canvas.get_width_height()[::-1] + (3,))

    data_back = np.round(curr_min_val + data_c *(curr_max_val - curr_min_val), 3).astype(int)
    # data_back = np.round(curr_min_val + data_c *(curr_max_val - curr_min_val), 3)

    data_tensor = F.to_tensor(data_back)
    # data_tensor = F.to_tensor(data)
    # data_tensor[j] = F.to_tensor(data)

    plt.close(fig_temp)

    return data_tensor
    # return data


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()

    # Create the input data pipeline
    print("Loading the datasets...")

    # fetch dataloaders
    dataloaders = one_labels_data_loader.fetch_dataloader(['train', 'test'], args.data_dir)
    train_dl = dataloaders['train']
    test_dl = dataloaders['test']

    # get training samples and labels
    data = train_dl.dataset.images
    labels = train_dl.dataset.digit_labels

    print("data was loaded from {}".format(args.data_dir))
    print("- done.")

    # Define the model and optimizer
    model = net.NeuralNet()

    load_model(args.model_dir, args.restore_file)

    batch = next(iter(test_dl))
    images, _ = batch
    # images_samples, _ = batch

    # images_reshaped = vectors_to_samples(images_samples)
    #
    # # bg and test_images should be tensors
    # images = create_digit_image(images_reshaped, curr_min_val=0, curr_max_val=255)
    # # images = [create_digit_image(b_v, curr_min_val=0, curr_max_val=255) for b_v in images_reshaped]
    # # images = torch.tensor(images)

    size_of_batch = images.shape[0]
    bg_len = round(0.9 * size_of_batch)
    background = images[:bg_len]
    test_images_samples = images[bg_len: min(bg_len + 10, size_of_batch)]

    # input should be in shape [batch_size,3 (num_colors), im_h, im_w]
    # currently in shape [batch_size, im_h, im_w,3 (num_colors)]
    e = shap.DeepExplainer(model, background)
    shap_values_samples = e.shap_values(test_images_samples)

    # output is in shape [batch_size,24] changing to illustrate as an image
    shap_values = []
    shap_values = [[] for _ in range(len(shap_values_samples))]
    # shap_vals_reshaped = []
    # shap_vals_reshaped = torch.empty(len(shap_values_samples), len(shap_values_samples[0]), 3, 300, 200, dtype=torch.float)

    for j in range(len(shap_values_samples)):
        reshaped = vectors_to_samples(torch.tensor(shap_values_samples[j]))
        for k in range(len(reshaped)):
            tensor_image = create_digit_image(reshaped[k], curr_min_val=-255, curr_max_val=255)
            image = tensor_image.numpy()  # .tolist()
            shap_values[j].append(image)
            # shap_values.append(create_digit_image(reshaped[k], curr_min_val=-255, curr_max_val=255))
        # shap_vals_reshaped.append(vectors_to_samples(torch.tensor(shap_values_samples[j])))     # continue here !!!!

    test_images = torch.empty(len(test_images_samples), 3, im_h, im_w, dtype=torch.int)
    # test_images = []
    # for j in range(len(test_images_samples)):
    reshaped = vectors_to_samples(torch.tensor(test_images_samples))

    # continue here !!!!
    for k in range(len(reshaped)):
        test_images[k] = create_digit_image(reshaped[k], curr_min_val=-255, curr_max_val=255)
        # test_images.append(create_digit_image(reshaped[k], curr_min_val=-255, curr_max_val=255))
    # shap_vals_reshaped = vectors_to_samples(torch.tensor(v)) for v in shap_values_samples    # continue here !!!!
    # shap_values = create_digit_image(shap_vals_reshaped, curr_min_val=0, curr_max_val=255)

    # test_images_reshaped = vectors_to_samples(test_images_samples)
    # test_images = create_digit_image(test_images_reshaped, curr_min_val=0, curr_max_val=255)

    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    # test_numpy = np.swapaxes(np.swapaxes(test_images, 1, -1), 1, 2)
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

    # plot the feature attributions
    shap.image_plot(shap_numpy, -test_numpy)

    # fetch loss function and metrics
    loss_fn = net.loss_fn

    metrics = net.metrics
    incorrect = net.incorrect
    num_epochs = 10000

    test_metrics, incorrect_samples = evaluate(model, loss_fn, test_dl, metrics, incorrect, num_epochs - 1)

