import numpy as np
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import math
import argparse
import os

width = 1
height = 0.2

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/cgan_model', help="Directory containing params.json")


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


    # segments_centers.append([center[0], center[1] - width - height])
    # segments_centers.append([center[0], center[1]])
    # segments_centers.append([center[0], center[1] + width + height])
#     segments_centers.append([center[0] - width / 2 - height / 2, center[1] - width / 2 - height / 2])
#     segments_centers.append([center[0] + width / 2 + height / 2, center[1] - width / 2 - height / 2])
#     segments_centers.append([center[0] - width / 2 - height / 2, center[1] + width / 2 + height / 2])
#     segments_centers.append([center[0] + width / 2 + height / 2, center[1] + width / 2 + height / 2])

    vertical_horizon = [0, 0, 0, 1, 1, 1, 1]
    for i in range(len(segments_centers)):
        polygon = create_segment(segments_centers[i], vertical_horizon[i], colors[i])
        patches.append(polygon)
        myaxis.add_patch(polygon)

    return


def create_figure():
    fig = plt.figure()
    return fig


def fill_figure(samples, fig, labels=None):  # , save=True):
    # args = parser.parse_args()
    fig.clear()

    num_of_samples = len(samples)
    axes = np.zeros((4, 5)).tolist()

    for i in range(num_of_samples):
        row, col = np.unravel_index(i, (4, math.ceil(num_of_samples/4)))
        axes[row][col] = fig.add_subplot(4, math.ceil(num_of_samples / 4), i + 1)

        if labels is not None:
            # temp = labels[i]
            digit_val = str(labels[i])
            axes[row][col].set_title(digit_val)
        display_digit(samples[i], axes[row][col])

    plt.draw()
    # if save:
    #     path = os.path.join(args.model_dir, 'images')
    #     if not os.path.isdir(path):
    #         os.mkdir(path)
    #         fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(path, '', epoch, n_batch))
    # plt.show()
    plt.pause(0.001)

    # for i in range(4):
    #     for j in range(4):
    #         v = axes[i][j].get_title()
    #         print(v)
    #         axes[i][j].set_title('')
    #         v = axes[i][j].get_title()
    #         print(v)
    #        # axes[i][j].cla()

    # fig.clear()


def plot_graph(g_losses, d_losses, gtype):
    plt.figure(figsize=(10, 5))
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

    plt.legend()
    plt.show()
    return


def create_grid(num_of_samples):
    fig, axes = plt.subplots(4, math.ceil(num_of_samples / 4), subplot_kw=dict(), clear=True)
    # fig.set_visible(False)
    return fig, axes


def fill_grid(samples, fig, axes, epoch, n_batch, save=True):
    num_of_samples = len(samples)
    args = parser.parse_args()

    # plt.draw()


    for i in range(num_of_samples):
        row, col = np.unravel_index(i, (4, math.ceil(num_of_samples/4)))
        display_digit(samples[i], axes[row, col])

    # fig.set_visible(False)
    if save:
        path = os.path.join(args.model_dir, 'images')
        if not os.path.isdir(path):
            os.mkdir(path)
        fig.savefig('{}/{}_epoch_{}_batch_{}.png'.format(path, '', epoch, n_batch))

    plt.draw()
    plt.show()
    plt.pause(0.001)
    plt.close(fig)

    # plt.pause(0.001)
    # plt.show()

    return


if __name__ == '__main__':
    # create_grid(16)
    # colors = [[1, 0, 0]]*3
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
#     patches = []
    fig, ax = plt.subplots()
#     # ax.set_xlim([0, 2*width])
#     # ax.set_ylim([0, 3*width])
#     # ax.set_aspect('equal', 'box')
#     # ax.axis('equal')
#     # ax.axis([0, 3*width, 0, 3*width])
#
    display_digit(colors, ax)

    plt.show()


#     width = 100
#     height = 20

# colors1 = np.array([1, 0.5, 0])
# #
# # # for i in range(3):
# polygon = Polygon([[0.25, 0.75], [0.75, 0.75], [0.75, 0.25], [0.25, 0.25]], True, facecolor=colors1)
# patches.append(polygon)
#
# colors = np.random.rand(len(patches))
# print(colors)
# # colors1 = np.array([1, 0, 0])
# print(colors1)
# # colors = [100,20,50,70]
# p = PatchCollection(patches, alpha=1)
# p.set_array(colors1)
# # p.set_facecolor(colors1)
# # p.set_edgecolor(colors1)
# # # p.set_array(colors1)
# ax.add_patch(polygon)
# # ax.add_collection(p)
# # fig.colorbar(p, ax=ax)
#
# plt.show()
