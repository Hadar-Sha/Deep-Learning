"""Train the model"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

import math
import utils
import model_vae.vae_net as vae_net
import model_vae.one_label_data_loader as data_loader
import display_digit as display_results
# from evaluate import evaluate


parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', default='data/with-grayscale/data-with-grayscale-4000', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/vae_model', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


def load_model(model_dir, restore_file):
    # reload weights from restore_file if specified
    if restore_file is not None and model_dir is not None:
        restore_path = os.path.join(model_dir, restore_file + '.pth.tar')
        print("\n\nRestoring parameters from {}".format(restore_path))
        # logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, None)  # optimizer)
    return


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    model = vae_net.VAENeuralNet(params).cuda() if params.cuda else vae_net.VAENeuralNet(params)

    load_model(args.model_dir, args.restore_file)

    print(model)

    status_before_transfer = []
    i = 0
    for param_tensor in model.state_dict():
        status_before_transfer.append([param_tensor, i])
        # status_before_transfer.append([param_tensor,
        #            (model.state_dict()[param_tensor].norm()).item(), list(model.state_dict()[param_tensor].size())])
        # status_before_transfer.append(((model.state_dict()[param_tensor]).cpu().numpy()).tolist())

        i += 1
    print(status_before_transfer)