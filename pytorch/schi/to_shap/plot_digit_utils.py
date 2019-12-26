import torch
import math
import os
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


def create_background(color, center=[width, 1.5*width], area=1, text=None):

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


def create_segment(segment_center, vertical_or_horizontal, color, area=1, text=None):
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


def convert_to_after_filer_grayscale(np_colors, scale_green=0.02):
    max_gray = 1 + scale_green
    np_grays = (np_colors[:, 0]+np_colors[:, 1] * scale_green)/max_gray
    np_grays = np.reshape(np_grays, (np_grays.shape[0], 1))
    grayscale_np_colors = np_grays * np.ones((1, 3))
    return grayscale_np_colors


def display_digit(colors, myaxis, curr_min_val=0, curr_max_val=1, withgrayscale=False, annot=False):

    vertical_horizon = [0, 0, 0, 1, 1, 1, 1]

    numpy_colors = np.array(colors)

    # convert to [0,1] to draw
    if numpy_colors.min() < 0 or numpy_colors.max() > 1:
        if (curr_max_val - curr_min_val) <= 0:
            print('wrong min and max input values')
            return
        else:
            numpy_colors = (numpy_colors - curr_min_val) / (curr_max_val - curr_min_val)

    colors = numpy_colors.tolist()
    if withgrayscale:
        gray_colors = convert_to_after_filer_grayscale(numpy_colors).tolist()

    patches = []

    if withgrayscale:
        myaxis.set_xlim([0, 4 * width])
        myaxis.set_ylim([0, 6 * width])
    else:
        myaxis.set_xlim([0, 2 * width])
        myaxis.set_ylim([0, 3*width])

    myaxis.set_aspect('equal', 'box')
    myaxis.axis('off')

    if withgrayscale:
        segments_gray_centers = []
        gray_center = [3 * width, 1.5 * width]
        bg_patch = create_background(gray_colors[7], gray_center)
        patches.append(bg_patch)
        myaxis.add_patch(bg_patch)

        segments_gray_centers.append([gray_center[0], gray_center[1] + width + height])
        segments_gray_centers.append([gray_center[0], gray_center[1]])
        segments_gray_centers.append([gray_center[0], gray_center[1] - width - height])
        segments_gray_centers.append([gray_center[0] - width / 2 - height / 2, gray_center[1] + width / 2 + height / 2])
        segments_gray_centers.append([gray_center[0] + width / 2 + height / 2, gray_center[1] + width / 2 + height / 2])
        segments_gray_centers.append([gray_center[0] - width / 2 - height / 2, gray_center[1] - width / 2 - height / 2])
        segments_gray_centers.append([gray_center[0] + width / 2 + height / 2, gray_center[1] - width / 2 - height / 2])

        for i in range(len(segments_gray_centers)):
            polygon = create_segment(segments_gray_centers[i], vertical_horizon[i], gray_colors[i])
            patches.append(polygon)
            myaxis.add_patch(polygon)

    # else:
    segments_centers = []
    center = [width, 1.5*width]
    bg_patch = create_background(colors[7])
    patches.append(bg_patch)
    myaxis.add_patch(bg_patch)
    if annot is True:
        myaxis.annotate('background segment (8)',  # this is the text
                        # center,
                        (0, width*2.65),  # this is the point to label
                        color='k',
                        weight='bold',
                        fontsize=10*width,
                        textcoords="offset points",  # how to position the text
                        xytext=(0, 20),  # distance from text to points (x,y)
                        ha='left')  # horizontal alignment can be left, right or center

    segments_centers.append([center[0], center[1] + width + height])
    segments_centers.append([center[0], center[1]])
    segments_centers.append([center[0], center[1] - width - height])
    segments_centers.append([center[0] - width / 2 - height / 2, center[1] + width / 2 + height / 2])
    segments_centers.append([center[0] + width / 2 + height / 2, center[1] + width / 2 + height / 2])
    segments_centers.append([center[0] - width / 2 - height / 2, center[1] - width / 2 - height / 2])
    segments_centers.append([center[0] + width / 2 + height / 2, center[1] - width / 2 - height / 2])

    for i in range(len(segments_centers)):
        polygon = create_segment(segments_centers[i], vertical_horizon[i], colors[i])
        patches.append(polygon)
        myaxis.add_patch(polygon)
        if annot is True:
            myaxis.annotate(str(i+1),  # this is the text
                            segments_centers[i],
                            # (0, width * 2.65),  # this is the point to label
                            color='k',
                            weight='bold',
                            fontsize=10 * width,
                            textcoords="offset points",  # how to position the text
                            xytext=(0, -width),  # distance from text to points (x,y)
                            ha='center')  # horizontal alignment can be left, right or center

    return


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


def plot_images(images, num_rows=None, title=None, path=None):

    fig = plt.figure(figsize=(20, 30))
    num_of_samples = len(images)
    if num_rows is None or isinstance(num_rows, (int,)) is False:
        num_rows = max(1, int(np.floor(np.sqrt(num_of_samples))))
    axes = np.zeros((num_rows, int(np.ceil(num_of_samples / num_rows)))).tolist()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    if title is not None and isinstance(title, (str,)):
        fig.suptitle(title)

    for i in range(num_of_samples):
        row, col = np.unravel_index(i, (num_rows, int(np.ceil(num_of_samples / num_rows))))
        axes[row][col] = fig.add_subplot(num_rows, int(np.ceil(num_of_samples / num_rows)), i + 1)
        axes[row][col].axis('off')
        axes[row][col].imshow(images[i])

    # plt.draw()
    # plt.show()
    if path is not None:
        plt.pause(0.01)
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)
    plt.close(fig)
    return


def create_figure():
    fig = plt.figure(figsize=(20, 30))
    return fig


def close_figure(figure):
    plt.close(figure)
    return


def feed_digits_to_figure(samples, fig, image_path, curr_min_val, curr_max_val, dtype, labels=None, withgrayscale=False,
                          num_of_rows=None):
    fig.clear()

    num_of_samples = len(samples)
    if num_of_rows is None:
        num_of_rows = max(1, math.floor(math.sqrt(num_of_samples)))

    axes = np.zeros((num_of_rows, math.ceil(num_of_samples / num_of_rows))).tolist()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(num_of_samples):
        row, col = np.unravel_index(i, (num_of_rows, math.ceil(num_of_samples / num_of_rows)))
        axes[row][col] = fig.add_subplot(num_of_rows, math.ceil(num_of_samples / num_of_rows), i + 1)

        if labels is not None:
            digit_val = str(labels[i])
            axes[row][col].set_title(digit_val)

        temp = np.array(samples[i])
        if temp.min() >= curr_min_val and temp.max() <= curr_max_val:
            display_digit(samples[i], axes[row][col], curr_min_val, curr_max_val, withgrayscale)
        else:
            axes[row][col].axis('off')

    # save graph
    # path = os.path.join(image_path, 'images')
    # if not os.path.isdir(path):
    #     os.mkdir(path)

    impath = os.path.join(image_path, 'test_samples.png')

    fig.savefig(impath, bbox_inches='tight')
    return


if __name__ == '__main__':
    color_one = [
        [0.62, 0, 0.5],
        [0.84, 0.06, 0.1],
        [0.84, 0.06, 0.1],
        [0.86, 0.047, 0.73],
        [0.62, 0.95, 0.5],
        [0.86, 0.047, 0.73],
        [0.62, 0, 0.5],
        # background
        [0.823, 0.803, 0]]

    color_two = [
        [0.5, 0.2, 0],
        [0, 0, 0.5],
        [0.5, 0.2, 0],
        [0.5, 0.2, 0],
        [0.5, 0.2, 0],
        [1, 1, 1],
        [0.5, 0.2, 0],
        # background
        [1, 1, 1]]

    color_hide = [
        [0.961, 0.416, 0.988],
        [0.961, 0.416, 0.988],
        [0.961, 0.416, 0.988],
        [0.961, 0.416, 0.988],
        [0.961, 0.71, 0.141],
        [0.859, 0.318, 0.886],
        [0.961, 0.416, 0.988],
        [0.859, 0.71, 0.141]
    ]
    gray = [[0.5], [0], [0.5], [0.5], [0.5], [1], [0.5]]
    colors = color_one

    fig = plt.figure()

    # ax = fig.add_subplot()
    # display_digit(color_hide, ax)
    # fig.savefig('./hiding-sample', bbox_inches='tight')
    # # plt.tight_layout()
    # # plt.show()
    # #
    #
    # color_hide_r = [[v[0], 0, 0] for v in color_hide]
    #
    # fig.clear()
    # # fig = plt.figure()
    # ax = fig.add_subplot()
    # display_digit(color_hide_r, ax)
    # fig.savefig('./hiding-sample-r', bbox_inches='tight')
    # # plt.tight_layout()
    # # plt.show()
    #
    # color_hide_g = [[0, v[1], 0] for v in color_hide]
    #
    # fig.clear()
    # # fig = plt.figure()
    # ax = fig.add_subplot()
    # display_digit(color_hide_g, ax)
    # fig.savefig('./hiding-sample-g', bbox_inches='tight')
    # # plt.tight_layout()
    # # plt.show()
    #
    # color_hide_b = [[0, 0, v[2]] for v in color_hide]
    #
    # fig.clear()
    # # fig = plt.figure()
    # ax = fig.add_subplot()
    # display_digit(color_hide_b, ax)
    # fig.savefig('./hiding-sample-b', bbox_inches='tight')
    # # plt.tight_layout()
    # # plt.show()

    color_hide_with_filt = convert_to_after_filer_grayscale(np.array(color_hide))

    fig.clear()
    # fig = plt.figure()
    ax = fig.add_subplot()
    display_digit(color_hide_with_filt, ax)
    fig.savefig('./hiding-with-filter', bbox_inches='tight')

    color_hide_g_filter_pass = [[0, 0.02*v[1], 0] for v in color_hide]

    fig.clear()
    # fig = plt.figure()
    ax = fig.add_subplot()
    display_digit(color_hide_g_filter_pass, ax)
    fig.savefig('./hiding-sample-g-filter-pass', bbox_inches='tight')