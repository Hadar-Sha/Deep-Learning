"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys
import numpy as np

import utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/learning_rate',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='data/64x64_SIGNS', help="Directory containing the dataset")
parser.add_argument('--early_stop', type=bool, default=True, help="Optional, do early stop")


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

    # Launch training with this config
    # adding compatibility to do early stop when searching for hyperparams
    cmd = "{python} train.py --model_dir={model_dir} --data_dir {data_dir} --early_stop {early_stop}".format(python=PYTHON, model_dir=model_dir,
                                                                                   data_dir=data_dir, early_stop=early_stop)
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
    hidden_sizes = list(range(30, 130, 10))  # list(range(34, 64, 2))  # list(range(24, 124, 2))
    dropout_rates = np.round(np.arange(0.25, 1, 0.25), 2).tolist()  # np.round(np.arange(0.25, 0.51, 0.05), 2).tolist() # 0
    num_epochs = 10000
    learning_rate = 1e-2

    for dropout_rate in dropout_rates:
        for hidden_size in hidden_sizes:
            # Modify the relevant parameter in params
            params.learning_rate = learning_rate
            params.hidden_size = hidden_size
            params.dropout_rate = dropout_rate
            params.num_epochs = num_epochs

            # Launch job (name has to be unique)
            job_name = "learning_rate_{}_hidden_size_{}_dropout_{}_num_epochs_{}"\
                .format(learning_rate, hidden_size, dropout_rate, num_epochs)
            launch_training_job(args.parent_dir, args.data_dir, args.early_stop, job_name, params)
