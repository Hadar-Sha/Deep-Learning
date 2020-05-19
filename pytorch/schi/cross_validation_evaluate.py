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
parser.add_argument('--parent_model_dir', default='experiments/base_model_weighted_schi_dist/syn-color/three_layers',
                    help='Directory containing params.json')
parser.add_argument('--parent_data_dir', default='data/with-grayscale/k-fold-cross-val',
                    help='Parent directory containing k folds of same data')
parser.add_argument('--k_fold', type=int, default=1, help="")
# parser.add_argument('--early_stop', type=bool, default=True, help="Optional, do early stop")


# 'experiments/learning_rate',

def launch_testing_job(parent_dir, data_dir, job_name):  # , params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    #
    # # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    params.cuda = torch.cuda.is_available()  # use GPU is available
    # params.save(json_path)

    # print(os.getcwd())
    os.chdir(curr_dir)
    # Launch training with this config
    # adding compatibility to do early stop when searching for hyperparams
    cmd = "{python} evaluate.py --model_dir={model_dir} --data_dir {data_dir}".format(python=PYTHON, model_dir=model_dir,
                                                                                   data_dir=data_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    curr_dir = os.getcwd()
    args = parser.parse_args()
    if args.base_dir_cpu and not torch.cuda.is_available():
        os.chdir(args.base_dir_cpu)

    json_path = os.path.join(args.parent_model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch over one parameter
    for k in range(1, args.k_fold + 1):
        job_name = "{}-fold".format(k)
        data_dir = os.path.join(args.parent_data_dir, "{}-fold".format(k))
        # data_dir = "data/with-grayscale/k-fold-cross-val/{}-fold".format(k)
        # model_dir = "experiments/k-fold-cross/{}-fold".format(k)
        # parent_dir = "experiments/k-fold-cross"

        # Launch job (name has to be unique)
        # job_name = "learning_rate_{}_hidden_size_{}_dropout_{}_num_epochs_{}"\
        #     .format(learning_rate, hidden_size, dropout_rate, num_epochs)
        if job_name is not False:
            launch_testing_job(args.parent_model_dir, data_dir, job_name)  # , params)
        else:
            print("no hyperparams chosen")
