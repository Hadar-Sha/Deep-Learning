import torch
import numpy as np
from torchvision.transforms import functional as F

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

width = 1
height = 0.2
im_w = 200
im_h = 300
RGB = 3

plt.ioff()


def vectors_to_samples(vectors):
    vectors = vectors.reshape(vectors.size()[0], -1, 3)
    vectors = vectors.cpu().numpy()
    vectors = vectors.tolist()
    return vectors


def grayscale_to_binary(colors):
    for i in range(RGB):
        for j in range(len(colors[0])):
            colors[i][j] = [1. if colors[i][j][it] == colors[i][j][7] else 0. for it in range(len(colors[0][0]))]

    return colors


def grayscale_to_white_bg(colors):
    for i in range(RGB):
        for j in range(len(colors[0])):
            max_v = np.array(colors[i][j]).max()
            min_v = np.array(colors[i][j]).min()
            colors[i][j] = [1. if colors[i][j][it] == colors[i][j][7] else 1.-(max_v-min_v)
                            for it in range(len(colors[0][0]))]

    return colors


def grayscale_to_bright(colors):
    for i in range(RGB):
        for j in range(len(colors[0])):
            max_v = np.array(colors[i][j]).max()
            min_v = np.array(colors[i][j]).min()
            colors[i][j] = [1. if colors[i][j][it] == max_v else 1.-(max_v-min_v)
                            for it in range(len(colors[0][0]))]

    return colors


def create_background(color, center=[width, 1.5*width], area=1):

    area = max(1, area)  # to make sure we avoid division by 0 later
    points = np.zeros([4, 2], dtype=float)
    points[0] = [center[0]-width, center[1]-1.5*width]
    points[1] = [center[0]-width, center[1]+1.5*width]
    points[2] = [center[0]+width, center[1]+1.5*width]
    points[3] = [center[0]+width, center[1]-1.5*width]

    if area > 1:
        color = [it / area for it in color]

    background = Polygon(points, True, facecolor=color)
    return background


def create_segment(segment_center, vertical_or_horizontal, color, area=1):
    points = np.zeros([6, 2], dtype=float)
    area = max(1, area)  # to make sure we avoid division by 0 later

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

    if area > 1:
        color = [it / area for it in color]

    segment = Polygon(points, True, facecolor=color)
    return segment


def create_digit_image(colors, fig, curr_min_val=0, curr_max_val=1, normalized_color=0):

    fig.clear()

    normalized = False
    is_grayscale = False

    # to avoid division with big number we normalize by total image area
    segment_area = 0.22*0.25*(im_w*im_w) / (im_w*im_h)
    background_area = ((im_w*im_h) - (7 * segment_area)) / (im_w*im_h)

    numpy_colors = np.array(colors)
    if len(numpy_colors.shape) == 1:
        numpy_colors = numpy_colors.reshape(numpy_colors.shape[0], 1) * np.ones([1, 3])
        is_grayscale = True

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
    if normalized_color:
        in_area = segment_area
        bg_area = background_area
    else:
        in_area = 1
        bg_area = 1

    bg_patch = create_background(colors[7], area=bg_area)
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

        polygon = create_segment(segments_centers[i], vertical_horizon[i], colors[i], area=in_area)
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


def samples_to_images(samples, fig=None):

    flag_close = False
    if fig is None:
        flag_close = True
        fig = plt.figure(figsize=(im_w, im_h), dpi=1)

    reshaped_test = vectors_to_samples(samples.clone().detach())
    ts_images = torch.empty(len(samples), RGB, im_h, im_w)
    for i in range(len(reshaped_test)):
        ts_images[i] = create_digit_image(reshaped_test[i], fig)

    ts_images = np.swapaxes(np.swapaxes(ts_images.numpy(), 1, -1), 1, 2)

    if flag_close:
        plt.close(fig)

    return ts_images


def plot_images(images, title=None, path=None):

    num_of_samples = len(images)
    num_rows = max(1, int(np.floor(np.sqrt(num_of_samples))))
    axes = np.zeros((num_rows, int(np.ceil(num_of_samples / num_rows)))).tolist()

    fig = plt.figure()
    if title is not None and isinstance(title, (str,)):
        fig.suptitle(title)

    for i in range(num_of_samples):
        row, col = np.unravel_index(i, (num_rows, int(np.ceil(num_of_samples / num_rows))))
        axes[row][col] = fig.add_subplot(num_rows, int(np.ceil(num_of_samples / num_rows)), i + 1)
        axes[row][col].axis('off')
        axes[row][col].imshow(images[i])

    plt.show()
    if path is not None:
        plt.savefig(path)
    plt.close(fig)
    return
