import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection, Collection
import matplotlib.pyplot as plt
import math
import argparse


#
# N = 4

width = 1
height = 0.2

# parser = argparse.ArgumentParser()
# parser.add_argument('--colorsfile', default='', help="file containing colors to draw")


def create_background(color):  # , center):
    # continue here !!!!!
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
    patches = []
    # fig, ax = plt.subplots()
    # myaxis.clear()
    myaxis.set_xlim([0, 2 * width])
    myaxis.set_ylim([0, 3*width])
    myaxis.set_aspect('equal', 'box')
    myaxis.axis('off')

    segments_centers = []
    center = [width, 1.5*width]
    # center = [0.5, 0.5]
    bg_patch = create_background(colors[7])  # , center)
    patches.append(bg_patch)
    myaxis.add_patch(bg_patch)

    segments_centers.append([center[0], center[1] - width - height])
    segments_centers.append([center[0], center[1]])
    segments_centers.append([center[0], center[1] + width + height])
    segments_centers.append([center[0] - width / 2 - height / 2, center[1] - width / 2 - height / 2])
    segments_centers.append([center[0] + width / 2 + height / 2, center[1] - width / 2 - height / 2])
    segments_centers.append([center[0] - width / 2 - height / 2, center[1] + width / 2 + height / 2])
    segments_centers.append([center[0] + width / 2 + height / 2, center[1] + width / 2 + height / 2])

    vertical_horizon = [0, 0, 0, 1, 1, 1, 1]
    for i in range(len(segments_centers)):
        polygon = create_segment(segments_centers[i], vertical_horizon[i], colors[i])
        patches.append(polygon)
        myaxis.add_patch(polygon)

    # plt.show()
    # plt.draw()
    # plt.pause(0.02)
    # plt.show()
    return


def create_figure():
    fig = plt.figure()
    return fig


def fill_figure(samples, fig):

    num_of_samples = len(samples)
    axes = np.zeros((4, 4)).tolist()

    for i in range(num_of_samples):
        # axes.append(fig.add_subplot(4, math.ceil(num_of_samples / 4), i+1))
        # display_digit(samples[i], axes[i])
        row, col = np.unravel_index(i, (4, math.ceil(num_of_samples/4)))
        axes[row][col] = fig.add_subplot(4, math.ceil(num_of_samples / 4), i + 1)
        display_digit(samples[i], axes[row][col])

    plt.draw()
    # plt.show()
    plt.pause(0.001)
    fig.clear()


def create_grid(num_of_samples):
    fig, axes = plt.subplots(4, math.ceil(num_of_samples / 4), subplot_kw=dict(), clear=True)
    # plt.show()
    return fig, axes


def fill_grid(samples, axes):
    num_of_samples = len(samples)
    # axes = axes.ravel()

    # fig, axes = plt.subplots(1, num_of_samples, subplot_kw=dict(polar=False))
    # fig, axes = plt.subplots(4, round(num_of_samples/4), subplot_kw=dict(polar=False), clear=True)

    for i in range(num_of_samples):
        row, col = np.unravel_index(i, (4, math.ceil(num_of_samples/4)))
        display_digit(samples[i], axes[row, col])

        plt.draw()
        plt.pause(0.02)
        plt.show()

    # fig, axes = plt.subplots(4, round(num_of_samples/4), subplot_kw=dict(polar=True))
    # axes.set_xlim([0, 2 * width])
    # axes.set_ylim([0, 3 * width])
    # axes.set_aspect('equal', 'box')
    return


if __name__ == '__main__':
    create_grid(16)
#     colors = [[1, 0, 0]]*7
#     colors.append([0, 0, 1])
#     patches = []
#     # fig, ax = plt.subplots()
#     # ax.set_xlim([0, 2*width])
#     # ax.set_ylim([0, 3*width])
#     # ax.set_aspect('equal', 'box')
#     # ax.axis('equal')
#     # ax.axis([0, 3*width, 0, 3*width])
#
#     display_digit(colors)

    # plt.show()


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
