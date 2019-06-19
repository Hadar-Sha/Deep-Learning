import random
import os
import pandas as pd
import torch
import numpy as np
import glob

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html


class SchiDigitDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, num_samples, transform=None):
        # for reading data labels. reading the samples will be in get item

        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.num_classes = 10

        temp_mat = pd.read_csv(csv_file, header=None)
        self.mat = temp_mat.values

        # self.mat = self.mat[0:self.mat.shape[0]:round(0.1*self.mat.shape[0]), :] # 10 samples
        # self.mat = self.mat[0:50000:2500, :]
        # self.mat = self.mat[0:50000:500, :]
        # self.mat = self.mat[0:50000:250, :]

        if num_samples == 0:
            self.mat = self.mat[round(0.8*self.mat.shape[0]):round(0.9*self.mat.shape[0]), :]  # temp get only samples with label = 8
            # self.mat = self.mat[round(0.9*self.mat.shape[0]):, :]  # temp get only samples with label = 9

        elif num_samples > 0:
            interval = 1/self.num_classes
            # interval = 1/num_samples
            rand_ind = np.zeros(num_samples, dtype=int)
            for i in range(self.num_classes):
                seq = list(range(i * round(interval*self.mat.shape[0]), (i + 1) * round(interval*self.mat.shape[0])))
                random.seed(1)
                rand_ind[2 * i:2 * i + 2] = random.sample(seq, round(num_samples/self.num_classes))
                # low = i * round(interval*self.mat.shape[0])
                # high = (i + 1) * round(interval*self.mat.shape[0])
                # # for j in range(round(num_samples/self.num_classes)):
                # rand_ind[round(num_samples/self.num_classes) * i: round(num_samples/self.num_classes) * (i + 1)] = \
                #         np.random.randint(low, high, size=round(num_samples/self.num_classes))

            self.mat = self.mat[rand_ind, :]

        temp_images = self.mat[:, :24]
        self.images = temp_images.astype(float)

        digit_labels = self.mat[:, 24]
        self.digit_labels = digit_labels.reshape(len(digit_labels), 1)
        # self.digit_labels = digit_labels

        self.transform = transform

    def __len__(self):
        # return 1
        return len(self.digit_labels)

    def __getitem__(self, idx):
        # image = self.images
        # label = self.digit_labels
        image = self.images[idx, :]
        label = self.digit_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class Normalize(object):  # , low_bound, up_bound):

    def __call__(self, sample):
        # normalize to [-1,1]
        # sample = sample / 255.
        sample = (sample - 127.5)/127.5
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        tensorimage = torch.from_numpy(sample)
        tensorimage = tensorimage.type(torch.FloatTensor)

        return tensorimage


def fetch_dataloader(types, data_dir, params):
    """
    Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'dev', 'test']:

        if split in types:
            newpath = os.path.join(data_dir, "{}".format(split))

            my_list = os.listdir(newpath)
            file = my_list[0]
            path = os.path.join(newpath, file)

            # added to sample limited samples only
            num_samples = params.num_samples

            # Normalize() was added, wasn't in discriminator net
            # prevent shuffling in dev or test
            if split == 'train':
                dl = DataLoader(dataset=SchiDigitDataset(csv_file=path, num_samples=num_samples, transform=transforms.Compose(
                    [Normalize(), ToTensor()])), batch_size=params.batch_size, shuffle=True)

            else:
                dl = DataLoader(dataset=SchiDigitDataset(csv_file=path, num_samples=num_samples, transform=transforms.Compose(
                    [Normalize(), ToTensor()])), batch_size=params.batch_size, shuffle=False)

            dataloaders[split] = dl

    return dataloaders
