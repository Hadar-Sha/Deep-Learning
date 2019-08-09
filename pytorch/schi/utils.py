import json
import logging
import os
import shutil
import csv
import types
import numpy as np

import torch


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


class WeightedAverage():
    """A simple class that maintains the weighted running average of a quantity


    """

    def __init__(self):
        # self.steps = 0
        self.total = 0
        self.prop = 0

    def update(self, val, prop):
        self.total += val*prop
        self.prop += prop
        # self.steps += 1

    def __call__(self):
        return self.total / float(self.prop)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # # Logging to console
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(logging.Formatter('%(message)s'))
        # logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path, epoch=None):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
        epoch:
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        # d = {k: "{:.3f}".format(v) for k, v in d.items()}
        d = {k: round(float(v), 3) for k, v in d.items()}
        if epoch is not None:
            d.update({'epoch': epoch})
        json.dump(d, f, indent=4)
        # if epoch is not None:
        #     e = {'epoch': epoch}
        #     json.dump(e, f, indent=4)


def save_checkpoint(state, is_best, checkpoint, ntype=None, best_type=None):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
        ntype:
        best_type:
    """
    if ntype and best_type:
        filepath = os.path.join(checkpoint, ntype + '_' + best_type + '_' + 'last.pth.tar')
    elif ntype:
        filepath = os.path.join(checkpoint, ntype + '_' + 'last.pth.tar')
    else:
        filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    # else:
        # print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        if ntype and best_type:
            shutil.copyfile(filepath, os.path.join(checkpoint, ntype + '_' + best_type + '_' + 'best.pth.tar'))
        elif ntype:
            shutil.copyfile(filepath, os.path.join(checkpoint, ntype + '_' + 'best.pth.tar'))
        else:
            shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def save_weights_biases(dataname, data, checkpoint):
    filepath = os.path.join(checkpoint, 'wb.csv')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)

    with open(filepath, "a", newline='') as myfile:
        myfile.write(dataname)
        myfile.write('\n')
        csvwr = csv.writer(myfile)
        list_data = data.tolist()

        row = list_data[0]
        if isinstance(row, list):
            str_data = [["{:05.5f}".format(row[i]) for i in range(len(row))] for row in list_data]
            for row in str_data:
                csvwr.writerow(row)
        else:
            csvwr.writerow(list_data)

    return


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def save_incorrect_to_csv(samples, csv_path):
    with open(csv_path, 'w', newline='') as csvfile:
        mywriter = csv.writer(csvfile, delimiter=',')
        for item in samples:
            if len(item) > 0:
                mywriter.writerow(item)

    return
