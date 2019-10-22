import numpy as np
import matplotlib
from matplotlib.patches import Circle, Wedge, Polygon
# from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt


def ilustrate_after_filter(colors, scale_green=0.02):
    max_gray = 1 + scale_green
    np_grays = (colors[:, 0] + colors[:, 1] * scale_green) / max_gray
    np_grays = np.reshape(np_grays, (np_grays.shape[0], 1))
    grayscale_np_colors = np_grays * np.ones((1, 3))
    return grayscale_np_colors


fig = plt.figure()
myaxis = fig.add_subplot()
myaxis.axis('off')
# fig, ax = plt.subplots()

N = 5
x = np.arange(0, 2, 0.01)  # np.random.rand(N)
leng = np.shape(x)[0]
# M = np.shape(x)
size = np.random.randint(0, 15, 10)
# y = np.cos(2 * np.pi * x)
# print(y)
alpha = 0.8  # 1

for i in range(10):
    y = np.cos(x * size[i])
    # y = np.cos(2 * np.pi * x * size[i])
    plt.plot(x, y, color=np.array([1, 0.1+0.1*np.random.rand(), np.random.rand(), alpha]))

for i in range(10):
    y = np.sin(x * size[i])
    # y = np.cos(2 * np.pi * x * size[i])
    plt.plot(x, y, color=np.array([1, 0.1+0.1*np.random.rand(), np.random.rand(), alpha]))

for i in range(10):
    y = np.sin(x * size[i]) + np.cos(x * size[i])
    # y = np.cos(2 * np.pi * x * size[i])
    plt.plot(x, y, color=np.array([1, 0.1+0.1*np.random.rand(), np.random.rand(), alpha]))


# x = np.arange(0.25, 0.9, 0.1)  # np.random.rand(N)
# # print(x)
# y = 0.8*np.random.rand(N)+0.1
# radii = 0.1*np.random.rand(N)+0.1
patches = []
# # colors = np.zeros(0,3)
# circles_cols = []
# for x1, y1, r in zip(x, y, radii):
#     rand_col = np.array([0, 0, 0.5 + 0.5*np.random.rand(), 0.4])
#     # rand_col = np.random.rand(4)
#     # rand_col[:1] = 0
#     # rand_col[3] = 0.4
#     # print(rand_col)
#     circles_cols.append(rand_col)
#
#     circle = Circle((x1, y1), r, facecolor=rand_col)
#     patches.append(circle)
#     myaxis.add_patch(circle)

x1 = np.arange(0.2, 2, 0.4)
# print(x1.shape)
for i in range(5):
# for i in range(N-2):
    rand_col = np.array([0.1*np.random.rand(), 0, np.random.rand(), alpha])
    # tringle_vals = np.array([x1, x1+0.5+0.2*np.random.rand(5)]).transpose()
    # polygon = Polygon(tringle_vals, True, facecolor=rand_col)
    polygon = Polygon(2*np.random.rand(3, 2), True, facecolor=rand_col)
    patches.append(polygon)
    myaxis.add_patch(polygon)
# colors = np.random.rand(len(patches))  # , 3)
# print(colors)
# p = PatchCollection(patches, alpha=0.8)

# p.set_array(np.array(colors))
# ax.add_collection(p)

plt.show()