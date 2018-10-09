
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
import os
import argparse
import sys


# Hyper Parameters 
input_size = 24
hidden_size = 98 #100 #128  # 120
num_classes = 10 #11    # should be 11 with experiment data ( no digit is the extra option)
num_epochs = 40000 #30000 #10000  # was 2000
batch_size = 100
learning_rate = 0.01 #0.005 #0.01  #0.015  # 0.005
momentum_val= 0.1 #0.9999


training_data_path = './data/train/exp-data-train.csv'
dev_data_path = './data/dev/exp-data-dev.csv'

# training_data_path='../data/data-deep-train.csv'
# test_data_path='../data/data-deep-test.csv'

# parser = argparse.ArgumentParser()
# parser.add_argument(
#   '--learning_rate',
#   type=float,
#   default=learning_rate,
#   help='Initial learning rate.'
# )
# parser.add_argument(
#   '--num_epochs',
#   type=int,
#   default=num_epochs,
#   help='Number of steps to run trainer.'
# )
# parser.add_argument(
#   '--hidden_size',
#   type=int,
#   default=hidden_size,
#   help='Number of units in hidden layer 1.'
# )
# # parser.add_argument(
# #   '--hidden2',
# #   type=int,
# #   default=12,
# #   help='Number of units in hidden layer 2.'
# # )
# parser.add_argument(
#   '--batch_size',
#   type=int,
#   default=batch_size,
#   help='Batch size.  Must divide evenly into the dataset sizes.'
# )
#
# FLAGS, unparsed = parser.parse_known_args()

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
        self.images = self.mat[:, :24]

        digit_labels = self.mat[:, 24]
        self.digit_labels = digit_labels.reshape(len(digit_labels), 1)

        self.transform = transform

    def __len__(self):
        return len(self.digit_labels)

    def __getitem__(self, idx):
        image= self.images[idx, :]
        label = self.digit_labels[idx]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Normalize(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image_normalized = np.divide(image,255.)
        return {'image': image_normalized,
                'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        tensorimage = torch.from_numpy(image)
        tensorimage = tensorimage.type(torch.FloatTensor)

        return {'image': tensorimage,
                'label': label}

# if __name__ == '__main__':
#     main(argv=[sys.argv[0]] + unparsed)
#
#
# def main(_):
#     # Training the Model
#     for epoch in range(num_epochs):
#         train(epoch)
#         test(epoch)
#
#     return


print(os.getcwd())

digit_train_dataset = SchiDigitDataset(csv_file=training_data_path,
                                       transform=transforms.Compose([ToTensor()]))

digit_test_dataset = SchiDigitDataset(csv_file=dev_data_path,
                                       transform=transforms.Compose([ToTensor()]))

# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=digit_train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=digit_test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
# #"taste" the data
# it = iter(train_loader)
# print (it)
# im,_=it.next()
# torchvision.utils.save_image(im,'./data/example.png')


# Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):#, isTrain):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # (hidden_size, int(hidden_size*0.75))
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)  # (int(hidden_size*0.75), hidden_size//2)
        self.fc4 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):

        prob = 0.25

        # first layer
        out = F.relu(self.fc1(x))
        out = F.dropout(out, prob, training=self.training)

        # second layer
        out = F.relu(self.fc2(out))
        out = F.dropout(out, prob, training=self.training)

        # third layer
        out = F.relu(self.fc3(out))
        out = F.dropout(out, prob, training=self.training)

        out = self.fc4(out)

        return out


net = NeuralNet(input_size, num_classes)
print(net)

# Loss and Optimizer
# Softmax is internally computed.
criterion = nn.CrossEntropyLoss(size_average=True, reduce=True)
# criterion = nn.BCEWithLogitsLoss(size_average=True, reduce=True)
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate )# , momentum=momentum_val)

print('number of parameters: ', sum(param.numel() for param in net.parameters()))


def train(ep):
    net.train()      # set model in training mode (need this because of dropout)
    for batch_idx, sample_batched in enumerate(train_loader):
        images = Variable(sample_batched['image'])
        labels = Variable(sample_batched['label'])
        labels = labels.view(labels.size(0))
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if ep % 100 == 0 or ep == num_epochs - 1:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                ep, batch_idx * len(images), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(ep):
    net.eval()
    correct = 0
    total = 0

    for test_sample in test_loader:
        images_test = Variable(test_sample['image'])
        labels_test = test_sample['label']
        labels_test = labels_test.view(labels_test.size(0))

        outputs_test = net(images_test)

        _, predicted = torch.max(outputs_test.data, 1)

        total += labels_test.size(0)
        correct += (predicted == labels_test).sum()

    if ep % 100 == 0 or ep == num_epochs-1 :
        print('Accuracy of the model on the {} test images: {:.2f} %' .format(total, (100 * correct / total)))


# Training the Model
for epoch in range(num_epochs):
    train(epoch)
    test(epoch)


# # Save the Model
# torch.save(model.state_dict(), 'model.pkl')


