import numpy as np
from matplotlib.patches import Polygon
from matplotlib import image
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import argparse
import os
from matplotlib import animation
from torchvision.transforms import functional as F
import torch

width = 1
height = 0.2

plt.ioff()


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


def convert_to_after_filer_grayscale(np_colors, scale_green=0.02):
    max_gray = 1 + scale_green
    np_grays = (np_colors[:, 0]+np_colors[:, 1] * scale_green)/max_gray
    np_grays = np.reshape(np_grays, (np_grays.shape[0], 1))
    grayscale_np_colors = np_grays * np.ones((1, 3))
    return grayscale_np_colors


def display_digit(colors, myaxis, curr_min_val=0, curr_max_val=1, withgrayscale=False):

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

    return


def create_digit_image(colors, curr_min_val=0, curr_max_val=1):

    numpy_colors = np.array(colors)

    # convert to [0,1] to draw
    if numpy_colors.min() < 0 or numpy_colors.max() > 1:
        if (curr_max_val - curr_min_val) <= 0:
            print('wrong min and max input values')
            return
        else:
            numpy_colors = round(((numpy_colors - curr_min_val) / (curr_max_val - curr_min_val)), 3)

    colors = numpy_colors.tolist()
    if len(numpy_colors.shape) < 2 or len(numpy_colors.shape) > 3:
        print('wrong input dimention. should be [_,8,3]')
    elif len(numpy_colors.shape) == 2:
        colors = [colors]
    # elif len(numpy_colors.shape)==3:
    #   do noting
    data_tensor = torch.empty(numpy_colors.shape[0], 3, 300, 200, dtype=torch.float)

    for j in range(numpy_colors.shape[0]):

        fig_temp, myaxis = plt.subplots(figsize=(200, 300), dpi=1)
        fig_temp.subplots_adjust(0, 0, 1, 1)
        myaxis.axis('off')
        myaxis.set_xlim([0, 2 * width])
        myaxis.set_ylim([0, 3 * width])
        myaxis.set_aspect('equal', 'box')

        patches = []

        segments_centers = []
        center = [width, 1.5*width]
        bg_patch = create_background(colors[j][7])
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
            polygon = create_segment(segments_centers[i], vertical_horizon[i], colors[j][i])
            patches.append(polygon)
            myaxis.add_patch(polygon)

        fig_temp.canvas.draw()
        # plt.show()

        data = np.fromstring(fig_temp.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig_temp.canvas.get_width_height()[::-1] + (3,))

        # fig_temp.savefig('test_{}.png'.format(j), dpi=1)
        data_tensor[j] = F.to_tensor(data)

        plt.close(fig_temp)

    return data_tensor


def create_figure():
    fig = plt.figure(figsize=(20, 30))
    return fig


def close_figure(figure):
    plt.close(figure)
    return


def feed_digits_to_figure(_, samples, fig, epoch, image_path, curr_min_val, curr_max_val, labels, dtype, withgrayscale):
    fig.clear()
    if dtype is not None:
        fig.suptitle('batch #{}'.format(epoch))
    else:
        fig.suptitle('epoch #{}'.format(epoch))

    num_of_samples = len(samples)
    num_of_rows = max(1, math.floor(math.sqrt(num_of_samples)))

    axes = np.zeros((num_of_rows, math.ceil(num_of_samples / num_of_rows))).tolist()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i in range(num_of_samples):
        row, col = np.unravel_index(i, (num_of_rows, math.ceil(num_of_samples / num_of_rows)))
        axes[row][col] = fig.add_subplot(num_of_rows, math.ceil(num_of_samples / num_of_rows), i + 1)

        if labels is not None:
            digit_val = str(labels[i])
            axes[row][col].set_title(digit_val)
        display_digit(samples[i], axes[row][col], curr_min_val, curr_max_val, withgrayscale)

    # save graph
    path = os.path.join(image_path, 'images')
    if not os.path.isdir(path):
        os.mkdir(path)

    if dtype is not None:
        impath = os.path.join(path, '{}_samples_batch_#{}.png'.format(dtype, epoch))
    else:
        impath = os.path.join(path, 'test_samples_epoch_#{}.png'.format(epoch))
    plt.savefig(impath, bbox_inches='tight')
    return


def fill_figure(samples, fig, epoch, image_path, curr_min_val, curr_max_val, withgrayscale=False, dtype=None, labels=None):

    im_ani = animation.FuncAnimation(fig, feed_digits_to_figure, frames=None,
                         fargs=(samples, fig, epoch, image_path, curr_min_val, curr_max_val, labels, dtype, withgrayscale), interval=2, repeat=False, blit=False)

    plt.draw()
    plt.pause(0.01)


def plot_graph(losses_one, losses_two, gtype, image_path, epoch=None):
    # plt.close('all')
    fig1 = plt.figure()

    if gtype == "Loss":
        plt.title("Generator and Discriminator Loss During Training")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
    elif gtype == "General Loss":
        plt.title("Loss During Training")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
    elif gtype == "VAE Loss":
        plt.title("BCE and KL Loss During Training")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
    elif gtype == "Predictions":
        plt.title("Generator and Discriminator predictions During Training")
        plt.xlabel("iterations")
        plt.ylabel("Predictions")
    elif gtype == "Grads_Best" and epoch is not None:
        plt.title("min and max gradients with best metrics epoch {}".format(epoch))
        plt.xlabel("layers")
        plt.ylabel("Grads")
    elif gtype == "Grads":
        plt.title("min and max gradients During Training")
        plt.xlabel("layers")
        plt.ylabel("Grads")

    if (not losses_one) is False and (not losses_two) is False:
    # if losses_one is not None and (not losses_one) is False and losses_two is not None and (not losses_two) is False:
        if gtype == "VAE Loss":
            bce_h, = plt.plot(losses_one, label="BCE")
            kl_h, = plt.plot(losses_two, label="KL")
            plt.legend(handles=[bce_h, kl_h], fontsize="small")
        elif gtype == "Grads_Best" or gtype == "Grads":
            plt.plot(losses_one)
            plt.plot(losses_two)
        else:
            g_h = plt.plot(losses_one, label="G")
            d_h = plt.plot(losses_two, label="D")
            # plt.legend(handles=[g_h, d_h])
            plt.legend([g_h, d_h], ['G', 'D'], fontsize="small")
    elif (not losses_one) is False:
    # elif losses_one is not None and (not losses_one) is False:
        if gtype == "VAE Loss":
            bce_h, = plt.plot(losses_one, label="BCE")
            plt.legend(handles=[bce_h])
        elif gtype == "Grads_Best" or gtype == "Grads":
            plt.plot(losses_one)
        else:
            g_h, = plt.plot(losses_one, label="G")
            plt.legend(handles=[g_h])
        # plt.plot(losses_one)
    elif (not losses_two) is False:
    # elif losses_two is not None and (not losses_two) is False:
        if gtype == "VAE Loss":
            kl_h, = plt.plot(losses_two, label="KL")
            plt.legend(handles=[kl_h])
        elif gtype == "Grads_Best" or gtype == "Grads":
            plt.plot(losses_two)
        else:
            d_h, = plt.plot(losses_two, label="D")
            plt.legend(handles=[d_h])
        # plt.plot(losses_two)
    else:
        print('no data was provided')
        return

    #save graph
    path = os.path.join(image_path, 'images')
    if not os.path.isdir(path):
        os.mkdir(path)
    impath = os.path.join(path, '{}_graph.png'.format(gtype))
    plt.savefig(impath)
    plt.pause(1)
    plt.close(fig1)

    return


if __name__ == '__main__':

    # colors = np.array([231,231,231,173,173,173,231,231,231,231,231,231,231,231,231,231,231,231,231,231,231,173,173,173])
    # colors = colors/255.
    #
    # colors = np.reshape(colors, (8, 3))
    #
    # colors = colors.tolist()

    color_one = [
        [0.5, 0.2, 0],
        [0, 0, 0.5],
        [0.5, 0.2, 0],
        [0.5, 0.2, 0],
        [0.5, 0.2, 0],
        [0, 0, 0],
        [0.5, 0.2, 0],
        # background
        [0, 0, 0]]

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

    colors = [color_one, color_two]
    # colors = []
    # colors.append([0.5, 0.2, 0])
    # colors.append([0, 0, 0.5])
    # colors.append([0.5, 0.2, 0])
    # colors.append([0.5, 0.2, 0])
    # colors.append([0.5, 0.2, 0])
    # colors.append([0, 0, 0])
    # colors.append([0.5, 0.2, 0])
    # # background
    # colors.append([0, 0, 0])

    print(colors)
    # fig, ax = plt.subplots()

    data_t = create_digit_image(colors)

    print(data_t.min())
    print(data_t.max())
    # display_digit(colors, ax, True)
#     #
#     plt.show()
#     # plt.close('all')
#
#     # create_digit_image(colors)


