"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys
import numpy as np

import utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/acgan_model/debug',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='data/color-syn-one-color-big', help="Directory containing the dataset")


def launch_training_job(parent_dir, data_dir, job_name, params):
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

    # Launch training with this config
    # adding compatibility to do early stop when searching for hyperparams
    cmd = "{python} train_aux_classifier_gan.py --model_dir={model_dir} --data_dir {data_dir}".format(python=PYTHON, model_dir=model_dir,
                                                                                   data_dir=data_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch over one parameter
    # learning_rates = [1e-3, 1e-2]  # 1e-4
    hidden_sizes = list(range(100, 501, 100))
    z_noise_dims = list(range(200, 401, 100))
    # dropout_rates = np.round(np.arange(0.25, 1, 0.25), 2).tolist()
    # noise_types = ['normal', 'uniform', 'binary']
    num_epochs = 1000
    # learning_rate = 1e-2

    for hidden_size in hidden_sizes:
        for noise_dim in z_noise_dims:
            # Modify the relevant parameter in params
            # params.learning_rate = learning_rate
            params.hidden_size = hidden_size
            # params.noise_type = noise_type
            params.num_epochs = num_epochs
            params.noise_dim = noise_dim

            # Launch job (name has to be unique)
            job_name = "hidden_size_{}_noise_dim_{}"\
                .format(hidden_size, noise_dim)
            launch_training_job(args.parent_dir, args.data_dir, job_name, params)
