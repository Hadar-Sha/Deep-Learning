import numpy as np
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import argparse
import os
from matplotlib import animation

width = 1
height = 0.2

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/cgan_model', help="Directory containing params.json")

plt.ioff()


def create_background(color):
    points = np.zeros([4, 2], dtype=float)
    points[0] = [0, 0]
    points[1] = [0, 3*width]
    points[2] = [2*width, 3*width]
    points[3] = [2*width, 0]

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


def display_digit(colors, myaxis):

    # flat_colors = [item for sublist in colors for item in sublist]

    numpy_colors = np.array(colors)
    # numpy_colors = np.array([np.array(xi) for xi in colors])

    if numpy_colors.min() < 0 or numpy_colors.max() > 1:
        numpy_colors = (numpy_colors - numpy_colors.min()) / (numpy_colors.max() - numpy_colors.min())

    colors = numpy_colors.tolist()

    patches = []
    myaxis.set_xlim([0, 2 * width])
    myaxis.set_ylim([0, 3*width])
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

    return


def create_figure():
    fig = plt.figure()
    return fig


def close_figure(figure):
    plt.close(figure)
    return


def feed_digits_to_figure(_, samples, fig, epoch, labels):
    args = parser.parse_args()
    fig.clear()
    fig.suptitle('epoch #{}'.format(epoch))

    num_of_samples = len(samples)
    axes = np.zeros((4, 5)).tolist()

    for i in range(num_of_samples):
        row, col = np.unravel_index(i, (4, math.ceil(num_of_samples / 4)))
        axes[row][col] = fig.add_subplot(4, math.ceil(num_of_samples / 4), i + 1)

        if labels is not None:
            digit_val = str(labels[i])
            axes[row][col].set_title(digit_val)
        display_digit(samples[i], axes[row][col])

    # save graph
    path = os.path.join(args.model_dir, 'images')
    if not os.path.isdir(path):
        os.mkdir(path)
    impath = os.path.join(path, 'test_samples_epoch_#{}.png'.format(epoch))
    plt.savefig(impath)
    return


def fill_figure(samples, fig, epoch, labels=None):

    im_ani = animation.FuncAnimation(fig, feed_digits_to_figure, frames=None,
                         fargs=(samples, fig, epoch, labels), interval=2, repeat=False, blit=False)

    plt.draw()
    plt.pause(0.01)


def plot_graph(g_losses, d_losses, gtype):
    plt.close('all')
    args = parser.parse_args()
    fig1 = plt.figure()
    print(fig1)

    if gtype == "Loss":
        plt.title("Generator and Discriminator Loss During Training")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
    elif gtype == "Predictions":
        plt.title("Generator and Discriminator predictions During Training")
        plt.xlabel("iterations")
        plt.ylabel("Predictions")
    else:
        plt.title("Generator and Discriminator min and max gradients During Training")
        plt.xlabel("layers")
        plt.ylabel("Grads")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")

    #save graph
    path = os.path.join(args.model_dir, 'images')
    if not os.path.isdir(path):
        os.mkdir(path)
    impath = os.path.join(path, '{}_graph.png'.format(gtype))
    plt.savefig(impath)
    plt.pause(1)
    plt.close(fig1)

    return


def create_grid(num_of_samples):
    fig, axes = plt.subplots(4, math.ceil(num_of_samples / 4), subplot_kw=dict(), clear=True)
    return fig, axes


def fill_grid(samples, fig, axes, epoch, n_batch, save=True):
    num_of_samples = len(samples)
    args = parser.parse_args()

    for i in range(num_of_samples):
        row, col = np.unravel_index(i, (4, math.ceil(num_of_samples/4)))
        display_digit(samples[i], axes[row, col])

    if save:
        path = os.path.join(args.model_dir, 'images')
        if not os.path.isdir(path):
            os.mkdir(path)
        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(path, '', epoch, n_batch))

    plt.draw()
    plt.show()
    plt.pause(0.001)
    plt.close(fig)

    return


if __name__ == '__main__':

    colors = []
    colors.append([1, 0, 0])
    colors.append([1, 0, 0])
    colors.append([1, 0, 0])
    colors.append([1, 0, 0])
    colors.append([1, 0, 0])
    colors.append([0, 0, 0])
    colors.append([1, 0, 0])
    # background
    colors.append([0, 0, 0])

    print(colors)
    fig, ax = plt.subplots()

    display_digit(colors, ax)

    plt.show()

