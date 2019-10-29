
import argparse
import logging
import os
import numpy as np
import torch
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default="toSync/Thesis/DL-Pytorch-data", help='path to experiments and data folder. not for Server')
parser.add_argument('--data_dir', default='data/two-labels/single-class/hidden-easy-big-only-9-5', help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/acgan_model/hiding_scheme/debug', help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    if args.parent_dir and not torch.cuda.is_available():
        past_to_drive = os.environ['OneDrive']
        os.chdir(os.path.join(past_to_drive, args.parent_dir))

