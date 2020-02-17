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

# gray_fig = plt.figure()
# myaxis_gray = gray_fig.add_subplot()
# myaxis_gray.axis('off')
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
    y = np.cos(x * i)
    # print(i)
    # print(size[i])
    # y = np.cos(x * size[i])
    # y = np.cos(2 * np.pi * x * size[i])
    plt.plot(x, y, color=np.array([1, 0.1+0.1*np.random.rand(), np.random.rand(), alpha]))

for i in range(10):
    y = np.sin(x * i)
    # y = np.sin(x * size[i])
    # y = np.cos(2 * np.pi * x * size[i])
    plt.plot(x, y, color=np.array([1, 0.1+0.1*np.random.rand(), np.random.rand(), alpha]))

for i in range(10):
    y = np.sin(x * i) + np.cos(x * i)
    # y = np.sin(x * size[i]) + np.cos(x * size[i])
    # y = np.cos(2 * np.pi * x * size[i])
    plt.plot(x, y, color=np.array([1, 0.1+0.1*np.random.rand(), np.random.rand(), alpha]))


# x = np.arange(0.25, 0.9, 0.1)  # np.random.rand(N)
# # print(x)
# y = 0.8*np.random.rand(N)+0.1
# radii = 0.1*np.random.rand(N)+0.1
patches = []
gray_patches = []
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
for i in range(10):
# for i in range(N-2):
    rand_col = np.array([0.1*np.random.rand(), 0.8*np.random.rand(), 0.5+0.5*np.random.rand(), alpha])

    # gray_col = np.array([rand_col[0], 0.2*rand_col[1], 0, alpha])

    # tringle_vals = np.array([x1, x1+0.5+0.2*np.random.rand(5)]).transpose()
    # polygon = Polygon(tringle_vals, True, facecolor=rand_col)
    rand_vals_x = 2*np.random.rand(3, 1)  # 0.5+ np.random.rand(3, 2)
    rand_vals_y = -1+2*np.random.rand(3, 1)
    rand_vals = np.concatenate((rand_vals_x, rand_vals_y),axis=1)
    # print(rand_vals.shape)
    polygon = Polygon(rand_vals, True, facecolor=rand_col)
    patches.append(polygon)
    myaxis.add_patch(polygon)
# colors = np.random.rand(len(patches))  # , 3)
# print(colors)
# p = PatchCollection(patches, alpha=0.8)

# p.set_array(np.array(colors))
# ax.add_collection(p)


# plt.show()
fig.tight_layout(pad=0)
fig.canvas.draw()
image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
image_from_plot = image_from_plot/255.

print(image_from_plot.shape)

# print(image_from_plot[:,:,0].shape)
# print(image_from_plot[:,:,0,np.newaxis].shape)
image_gray = (image_from_plot[:,:,0]+0.2*image_from_plot[:,:,1])/1.2
# image_gray = np.rint(image_gray)
print(image_gray.shape)
image_gray = image_gray[:,:,np.newaxis]
print(image_gray.shape)
print(image_gray.min())
print(image_gray.max())
image_gray = np.concatenate((image_gray, image_gray, image_gray), axis=2)

plt.show()

gray_fig = plt.figure()
myaxis_gray = gray_fig.add_subplot()
myaxis_gray.axis('off')

im = myaxis_gray.imshow(image_gray)
gray_fig.tight_layout(pad=0)
plt.show()