"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys
import random

import utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/vae_model/params',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='data/data-with-grayscale-4000', help="Directory containing the dataset")


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
    cmd = "{python} train_vae.py --model_dir={model_dir} --data_dir {data_dir}".format(python=PYTHON, model_dir=model_dir,
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
    learning_rates = [1e-4, 1e-3, 1e-2]
    batch_sizes = [2**6, 2**7, 2**8, 2**9]
    num_epochs = [10**2, 10**3, 10**4, 5*(10**4)]

    # try later
    # hidden_size, output_size

    rand_lr = [learning_rates[random.randrange(len(learning_rates))]
                  for item in range(15)]

    rand_bs = [batch_sizes[random.randrange(len(batch_sizes))]
                  for item in range(15)]

    rand_ne = [num_epochs[random.randrange(len(num_epochs))]
                  for item in range(15)]

    chosen_vals = []
    for i in range(15):
        chosen_vals.append([rand_lr[i], rand_bs[i], rand_ne[i]])

    for val in chosen_vals:
        # print(val[0])
        # print(val[1])
        # print(val[2])

        #Modify the relevant parameter in params
        params.learning_rate = val[0]
        params.batch_size = val[1]
        params.num_epochs = val[2]

        # Launch job (name has to be unique)
        job_name = "learning_rate_{}_batch_size_{}_num_epochs_{}".format(val[0], val[1], val[2])
        launch_training_job(args.parent_dir, args.data_dir, job_name, params)