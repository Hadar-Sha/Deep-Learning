"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys
import torch
import numpy as np

import utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir_cpu', default="C:/Users/H/Documents/Haifa Univ/Thesis/DL-Pytorch-data",
                    help='path to experiments and data folder in CPU only. not for Server')
parser.add_argument('--parent_dir', default='experiments/base_model_weighted_schi_dist/syn-color/three_layers',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='data/color-syn-one-color-big', help="Directory containing the dataset")
# parser.add_argument('--early_stop', type=bool, default=True, help="Optional, do early stop")


# 'experiments/learning_rate',

def launch_training_job(parent_dir, data_dir, early_stop, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # print(os.getcwd())
    os.chdir(curr_dir)
    # Launch training with this config
    # adding compatibility to do early stop when searching for hyperparams
    cmd = "{python} train.py --model_dir={model_dir} --data_dir {data_dir} --early_stop {early_stop}".format(python=PYTHON, model_dir=model_dir,
                                                                                   data_dir=data_dir, early_stop=early_stop)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    curr_dir = os.getcwd()
    args = parser.parse_args()
    if args.base_dir_cpu and not torch.cuda.is_available():
        # args.parent_dir = os.path.join(args.base_dir_cpu, args.parent_dir)
        # args.data_dir = os.path.join(args.base_dir_cpu, args.data_dir)
        os.chdir(args.base_dir_cpu)

    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch over one parameter

    learning_rate = None
    dropout_rate = None
    hidden_sizes = list(range(100, 301, 50))
    num_epochs = [1000]
    num_epochs.extend([5000 * 2 ** i for i in range(4)])

    for hidden_size in hidden_sizes:
        for num_epoch in num_epochs:
            job_name = ""
            # Modify the relevant parameter in params
            if learning_rate is not None:
                params.learning_rate = learning_rate
                job_name += "learning_rate_{}_".format(learning_rate)
            if hidden_size is not None:
                params.hidden_size = hidden_size
                job_name += "hidden_size_{}_".format(hidden_size)
            if dropout_rate is not None:
                params.dropout_rate = dropout_rate
                job_name += "dropout_{}_".format(dropout_rate)
            if num_epoch is not None:
                params.num_epochs = num_epoch
                job_name += "num_epochs_{}_".format(num_epoch)

            # Launch job (name has to be unique)
            # job_name = "learning_rate_{}_hidden_size_{}_dropout_{}_num_epochs_{}"\
            #     .format(learning_rate, hidden_size, dropout_rate, num_epochs)
            if not job_name is False:
                launch_training_job(args.parent_dir, args.data_dir, params.early_stop, job_name, params)
            else:
                print("no hyperparams chosen")
