import random
import os
import pandas as pd
import torch
import numpy as np
import glob

# from PIL import Image
# from torch.utils.data import Dataset #, DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

training_data_path = './data/train/exp-data-train.csv'
dev_data_path = './data/dev/exp-data-dev.csv'
test_data_path = './data/test/exp-data-test.csv'

# # borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# # and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# # define a training image loader that specifies transforms on images. See documentation for more details.
# train_transformer = transforms.Compose([
#     transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
#     transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
#     transforms.ToTensor()])  # transform it into a torch tensor
#
# # loader for evaluation, no horizontal flip
# eval_transformer = transforms.Compose([
#     transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
#     transforms.ToTensor()])  # transform it into a torch tensor


# class SIGNSDataset(Dataset):
#     """
#     A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
#     """
#     def __init__(self, data_dir, transform):
#         """
#         Store the filenames of the jpgs to use. Specifies transforms to apply on images.
#
#         Args:
#             data_dir: (string) directory containing the dataset
#             transform: (torchvision.transforms) transformation to apply on image
#         """
#         self.filenames = os.listdir(data_dir)
#         self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg')]
#
#         self.labels = [int(os.path.split(filename)[-1][0]) for filename in self.filenames]
#         self.transform = transform
#
#     def __len__(self):
#         # return size of dataset
#         return len(self.filenames)
#
#     def __getitem__(self, idx):
#         """
#         Fetch index idx image and labels from dataset. Perform transforms on image.
#
#         Args:
#             idx: (int) index in [0, 1, ..., size_of_dataset-1]
#
#         Returns:
#             image: (Tensor) transformed image
#             label: (int) corresponding label of image
#         """
#         image = Image.open(self.filenames[idx])  # PIL image
#         image = self.transform(image)
#         return image, self.labels[idx]

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
        self.images = np.float32(temp_images)
        # self.images = self.mat[:, :24]

        # self.images = self.images.astype(float)

        # self.images =
        # print(self.images)
        # print(self.images.dtype)

        digit_labels = self.mat[:, 24]
        self.digit_labels = digit_labels.reshape(len(digit_labels), 1)

        self.transform = transform

    def __len__(self):
        return len(self.digit_labels)

    def __getitem__(self, idx):
        image = self.images[idx, :]
        label = self.digit_labels[idx]
        # sample = {'image': image, 'label': label}

        # if self.transform:
        #     sample = self.transform(sample)

        # return sample
        return image, label

class Normalize(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image_normalized = np.divide(image,255.)
        # return {'image': image_normalized,
        #         'label': label}

        return image_normalized, label


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # print(image)
        # print('hello')
        tensorimage = torch.from_numpy(image)
        # print(tensorimage)
        tensorimage = tensorimage.type(torch.FloatTensor)

        return tensorimage
        # return {'image': tensorimage,
        #         'label': label}


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

    # digit_train_dataset = SchiDigitDataset(csv_file=training_data_path,
    #                                        transform=transforms.Compose([ToTensor()]))
    #
    # digit_dev_dataset = SchiDigitDataset(csv_file=dev_data_path,
    #                                       transform=transforms.Compose([ToTensor()]))
    #
    # # Dataset Loader (Input Pipline)
    # train_loader = torch.utils.data.DataLoader(dataset=digit_train_dataset,
    #                                            batch_size=params.batch_size,
    #                                            shuffle=True)
    #
    # dataloaders['train'] = train_loader
    # dev_loader = torch.utils.data.DataLoader(dataset=digit_dev_dataset,
    #                                           batch_size=params.batch_size,
    #                                           shuffle=False)
    #
    # dataloaders['dev'] = dev_loader
    # test_loader = torch.utils.data.DataLoader(dataset=digit_dev_dataset,
    #                                           batch_size=params.batch_size,
    #                                           shuffle=False)
    # dataloaders['test'] = test_loader

    #
    for split in ['train', 'dev', 'test']:
        # for split in ['train', 'val', 'test']:
        if split in types:
            # currdir = os.getcwd()
            newpath = os.path.join(data_dir, "{}".format(split))
            # os.chdir(newpath)

            myList = os.listdir(newpath)
            file = myList[0]
            print(file)
            path = os.path.join(newpath, file)

            # path = os.path.join(data_dir, "".format(split), "{}_signs".format(split))

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(dataset=SchiDigitDataset(csv_file=path, transform=transforms.Compose([ToTensor()])),
                                batch_size=params.batch_size, shuffle=True)
                # dl = DataLoader(SIGNSDataset(path, train_transformer), batch_size=params.batch_size, shuffle=True,
                #                         num_workers=params.num_workers,
                #                         pin_memory=params.cuda)
            else:
                dl = DataLoader(dataset=SchiDigitDataset(csv_file=path, transform=transforms.Compose([ToTensor()])),
                                batch_size=params.batch_size, shuffle=False)
                # dl = DataLoader(SIGNSDataset(path, eval_transformer), batch_size=params.batch_size, shuffle=False,
                #                 num_workers=params.num_workers,
                #                 pin_memory=params.cuda)

            dataloaders[split] = dl

    return dataloaders
