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

    def __init__(self, csv_file, transform=None):
        # for reading data labels. reading the samples will be in get item

        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        temp_mat = pd.read_csv(csv_file, header=None)
        self.mat = temp_mat.values

        temp_images = self.mat[:, :24]
        self.images = temp_images.astype(float)

        # self.images = np.reshape(temp_images, (temp_images.shape[0], temp_images.shape[1]//3, 3, 1))
        # self.images = np.reshape(temp_images, (temp_images.shape[0], temp_images.shape[1], 1))
        # self.images = temp_images.astype(int)
        # self.images = np.float32(temp_images)

        digit_labels = self.mat[:, 24]
        self.digit_labels = digit_labels.reshape(len(digit_labels), 1)

        self.transform = transform

    def __len__(self):
        return len(self.digit_labels)

    def __getitem__(self, idx):
        image = self.images[idx, :]
        label = self.digit_labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class Normalize(object):
    # print("in normalize")

    def __call__(self, sample):
        sample = (sample - 127.5)/127.5
        return sample
        # return sample.sub_(127.5).div_(127.5)
        # image, label = SchiDigitDataset.__getitem__()
        # image, label = sample['image'], sample['label']
        # image = np.divide(image, 255.)
        # return image, label


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        tensorimage = torch.from_numpy(sample)
        tensorimage = tensorimage.type(torch.FloatTensor)

        # image = self.images[idx]
        # image, label = sample['image'], sample['label']
        # tensorimage = torch.from_numpy(image)
        # tensorimage = tensorimage.type(torch.FloatTensor)
        #
        return tensorimage
        # return


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

            # Normalize() was added, wasn't in discriminator net
            # prevent shuffling in dev or test
            if split == 'train':
                dl = DataLoader(dataset=SchiDigitDataset(csv_file=path, transform=transforms.Compose(
                # [transforms.ToTensor(), transforms.Normalize((127.5, 127.5, 127.5),
                    # (127.5, 127.5, 127.5))])), batch_size=params.batch_size, shuffle=True)
                [Normalize(), ToTensor()])), batch_size=params.batch_size, shuffle=True)
                # [transforms.Normalize(0, 255), ToTensor()])), batch_size=params.batch_size, shuffle=True)

            else:
                dl = DataLoader(dataset=SchiDigitDataset(csv_file=path, transform=transforms.Compose(
                # [transforms.ToTensor(), transforms.Normalize((127.5, 127.5, 127.5),
                    # (127.5, 127.5, 127.5))])), batch_size=params.batch_size, shuffle=False)
                [Normalize(), ToTensor()])), batch_size=params.batch_size, shuffle=False)

            dataloaders[split] = dl

    return dataloaders
