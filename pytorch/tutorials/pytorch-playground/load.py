import torch
from torch.autograd import Variable
from utee import selector
import matplotlib.pyplot as plt
import numpy as np
import os

import svhn

plt.ioff()


def collect_network_statistics(net):

    net_grads_graph = []

    for param_tensor in net.state_dict():

        if (net.state_dict()[param_tensor]).dtype != torch.float:
            continue

        all_net_vals = ((net.state_dict()[param_tensor]).cpu().numpy()).tolist()

        # if needed, flatten the list to get one nim and one max
        flat_net_grads = []
        if isinstance(all_net_vals, (list,)):

            for elem in all_net_vals:
                if isinstance(elem, (list,)) and isinstance(elem[0], (list,)):
                    for item in elem:
                        flat_net_grads.extend(item)
                elif isinstance(elem, (list,)):
                    flat_net_grads.extend(elem)
                else:
                    flat_net_grads.extend([elem])
        else:
            flat_net_grads = all_net_vals

        net_grads_graph.append([min(flat_net_grads), np.median(flat_net_grads), max(flat_net_grads)])

    return net_grads_graph


def plot_graph(losses_one, losses_two, gtype, image_path, epoch=None):

    fig1 = plt.figure()

    # default value for title and x and y axis
    plt.xlabel("iterations")
    plt.ylabel("values")
    if isinstance(gtype, str):
        plt.title(gtype)
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
        # plt.ylim(0, 1)
        plt.yticks(np.arange(0, 1.1, step=0.2))
    elif gtype == "Grads_Best" and epoch is not None:
        plt.title("min and max gradients with best metrics epoch {}".format(epoch))
        plt.xlabel("layers")
        plt.ylabel("Grads")
    elif gtype == "Grads":
        plt.title("min and max gradients During Training")
        plt.xlabel("layers")
        plt.ylabel("Grads")
    elif gtype == "Accuracy":
        plt.title("Generator and Discriminator Accuracy During Training")
        plt.xlabel("iterations")
        plt.ylabel("Accuracy")
        plt.yticks(np.arange(0, 1.1, step=0.1))

    if (not losses_one) is False and (not losses_two) is False:
        if gtype == "VAE Loss":
            plt.plot(losses_one, label="BCE")
            plt.plot(losses_two, label="KL")
            plt.legend()
        elif gtype == "Grads_Best" or gtype == "Grads":
            plt.plot(losses_one)
            plt.plot(losses_two)
        else:
            plt.plot(losses_one, label="G")
            plt.plot(losses_two, label="D")
            plt.legend()
    elif (not losses_one) is False:
        if gtype == "VAE Loss":
            plt.plot(losses_one, label="BCE")
            plt.legend()
        elif gtype == "Grads_Best" or gtype == "Grads":
            plt.plot(losses_one)
        else:
            plt.plot(losses_one, label="G")
            plt.legend()
    elif (not losses_two) is False:
        if gtype == "VAE Loss":
            plt.plot(losses_two, label="KL")
            plt.legend()
        elif gtype == "Grads_Best" or gtype == "Grads":
            plt.plot(losses_two)
        else:
            plt.plot(losses_two, label="D")
            plt.legend()
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
    fig1.clf()
    # plt.clf()
    plt.close(fig1)

    return


is_cuda = torch.cuda.is_available()
path = os.getcwd()
print(path)

model_raw, ds_fetcher, is_imagenet = selector.select('svhn', cuda=is_cuda, model_root=path)
ds_val = ds_fetcher(batch_size=10, train=False, val=True, data_root=path)
for idx, (data, target) in enumerate(ds_val):
    data = Variable(torch.FloatTensor(data))
    if is_cuda:
        data = Variable(torch.FloatTensor(data)).cuda()
    output = model_raw(data)

print(idx)
print(output.shape)

net_vals = collect_network_statistics(model_raw)
# plot_summary_graphs_layers(net_vals, 'D', 'Grads', './')
plot_graph(net_vals, None, 'D', './', None)

# save_checkpoint({'epoch': 0,
#                       'state_dict': model_raw.state_dict(),
#                       'optim_dict': model_raw.state_dict()}, is_best=False, checkpoint='./')