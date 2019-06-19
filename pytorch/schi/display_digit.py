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


def convert_to_after_filer_grayscale(np_colors):
    scale_green = 0.02
    max_gray = 1 + scale_green
    np_grays = (np_colors[:,0]+np_colors[:,1]*scale_green)/max_gray
    np_grays = np.reshape(np_grays, (np_grays.shape[0],1))
    grayscale_np_colors = np_grays*np.ones((1,3))
    return grayscale_np_colors


def display_digit(colors, myaxis, withgrayscale=False):

    vertical_horizon = [0, 0, 0, 1, 1, 1, 1]

    numpy_colors = np.array(colors)

    if numpy_colors.min() < 0 or numpy_colors.max() > 1:
        if (numpy_colors.max() - numpy_colors.min()) > 0:
            numpy_colors = (numpy_colors + 1) / 2
            # numpy_colors = (numpy_colors - numpy_colors.min()) / (numpy_colors.max() - numpy_colors.min())
        else:
            return

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
        gray_center = [3*width, 1.5 * width]
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


def create_digit_image(colors):

    fig_temp, myaxis = plt.subplots(figsize=(20, 30), dpi=10)  # figsize=(17, 27), dpi=10

    numpy_colors = np.array(colors)

    if numpy_colors.min() < 0 or numpy_colors.max() > 1:
        numpy_colors = (numpy_colors - numpy_colors.min()) / (numpy_colors.max() - numpy_colors.min())

    colors = numpy_colors.tolist()

    patches = []
    myaxis.set_xlim([0, 2 * width])
    myaxis.set_ylim([0, 3 * width])
    myaxis.set_aspect('equal', 'box')
    myaxis.axis('off')

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

    fig_temp.canvas.draw()

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(5)
    # plt.savefig('./temp.png', bbox_inches='tight')

    data = np.fromstring(fig_temp.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig_temp.canvas.get_width_height()[::-1] + (3,))

    plt.close('all')
    data_tensor = F.to_tensor(data)

    return data_tensor


def create_figure():
    fig = plt.figure(figsize=(20, 30))
    return fig


def close_figure(figure):
    plt.close(figure)
    return


def feed_digits_to_figure(_, samples, fig, epoch, image_path, labels, dtype, withgrayscale):
    fig.clear()
    if dtype is not None:
        fig.suptitle('batch #{}'.format(epoch))
    else:
        fig.suptitle('epoch #{}'.format(epoch))

    num_of_samples = len(samples)
    num_of_rows = max(1, math.floor(0.2*num_of_samples))
    axes = np.zeros((num_of_rows, math.ceil(num_of_samples / num_of_rows))).tolist()

    for i in range(num_of_samples):
        row, col = np.unravel_index(i, (num_of_rows, math.ceil(num_of_samples / num_of_rows)))
        axes[row][col] = fig.add_subplot(num_of_rows, math.ceil(num_of_samples / num_of_rows), i + 1)

        if labels is not None:
            digit_val = str(labels[i])
            axes[row][col].set_title(digit_val)
        display_digit(samples[i], axes[row][col], withgrayscale)

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


def fill_figure(samples, fig, epoch, image_path, withgrayscale=False, dtype=None, labels=None):

    im_ani = animation.FuncAnimation(fig, feed_digits_to_figure, frames=None,
                         fargs=(samples, fig, epoch, image_path, labels, dtype, withgrayscale), interval=2, repeat=False, blit=False)

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

    if losses_one is not None and (not losses_one) is False and losses_two is not None and (not losses_two) is False:
        if gtype == "Loss":
            plt.plot(losses_one, label="G")
            plt.plot(losses_two, label="D")
        elif gtype == "VAE Loss":
            plt.plot(losses_one, label="BCE")
            plt.plot(losses_two, label="KL")
    elif losses_one is not None and (not losses_one) is False:
        plt.plot(losses_one)
    elif losses_two is not None and (not losses_two) is False:
        plt.plot(losses_two)
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


# def create_grid(num_of_samples):
#     fig, axes = plt.subplots(4, math.ceil(num_of_samples / 4), subplot_kw=dict(), clear=True)
#     return fig, axes
#
#
# def fill_grid(samples, fig, axes, epoch, n_batch, save=True):
#     num_of_samples = len(samples)
#     args = parser.parse_args()
#
#     for i in range(num_of_samples):
#         row, col = np.unravel_index(i, (4, math.ceil(num_of_samples/4)))
#         display_digit(samples[i], axes[row, col])
#
#     if save:
#         path = os.path.join(args.model_dir, 'images')
#         if not os.path.isdir(path):
#             os.mkdir(path)
#         fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(path, '', epoch, n_batch))
#
#     plt.draw()
#     plt.show()
#     plt.pause(0.001)
#     plt.close(fig)
#
#     return


if __name__ == '__main__':
    colors = np.array([231,231,231,173,173,173,231,231,231,231,231,231,231,231,231,231,231,231,231,231,231,173,173,173])
    colors = colors/255.

    colors = np.reshape(colors, (8, 3))

    colors = colors.tolist()
#
#     colors = []
#     colors.append([0.5, 0, 0])
#     colors.append([0, 0, 0.5])
#     colors.append([0.5, 0, 0])
#     colors.append([0.5, 0, 0])
#     colors.append([0.5, 0, 0])
#     colors.append([0, 0, 0])
#     colors.append([0.5, 0, 0])
#     # background
#     colors.append([0, 0, 0])
#
    print(colors)
    fig, ax = plt.subplots()
    #
    display_digit(colors, ax, True)
#     #
    plt.show()
#     # plt.close('all')
#
#     # create_digit_image(colors)


