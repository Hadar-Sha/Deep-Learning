
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
import csv

# for output
max_accuracy = 0
loss_out = 100
epoch_out = 0
samples_out = []
classifier_ans_out = []
correct_ans_out = []
mat_out = []
numpy_mat_out = np.zeros((26,), dtype=int)

# Hyper Parameters 
input_size = 24
hidden_size = 98 #100 #128  # 120
num_classes = 10 #11    # should be 11 with experiment data ( no digit is the extra option)
num_epochs = 40000 #30000 #10000 #60000  #50000 # was 2000 100
batch_size = 100
learning_rate = 0.02 #0.01  #0.015  # 0.005
momentum_val = 0.1 #0.9999
# log_interval = 10

# input_size = 784
# hidden_size = 64
# num_classes = 10
# num_epochs = 10
# batch_size = 100
# learning_rate = 0.01

training_data_path = '../data/exp-only/exp-data-train.csv'
test_data_path = '../data/exp-only/exp-data-test.csv'

output_path = '../data/exp-only/logfile_'+'l_r- {:.3f}'.format(learning_rate)+' num_ep- {}'.format(num_epochs) + '.txt'
csv_out_path = '../data/exp-only/logfile_'+'l_r- {:.3f}'.format(learning_rate)+' num_ep- {}'.format(num_epochs) + '.csv'
# training_data_path='../data/data-deep-train.csv'
# test_data_path='../data/data-deep-test.csv'


class SchiDigitDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, transform=None):
        # for reading data labels. reading the samples will be in get item

        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        temp_mat = pd.read_csv(csv_file, header=None)
        self.mat = temp_mat.values

        # print (self.digit_labels.shape)
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
        image_normalized= np.divide(image,255.)
        return {'image': image_normalized,
                'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        tensorimage = torch.from_numpy(image)
        tensorimage = tensorimage.type(torch.FloatTensor)

        # tensorlabel= torch.from_numpy(label)

        # temp = tensorlabel.size()
        # print(label.shape)

        # tensorlabel= tensorlabel.view(temp)

        # print(tensorlabel.size())
        # tensorlabel= tensorlabel.type(torch.FloatTensor)

        return {'image': tensorimage,
                'label': label}

        # return {'image': tensorimage,
        #         'label': tensorlabel}


#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3015,))])

digit_train_dataset = SchiDigitDataset(csv_file=training_data_path,
                                       transform=transforms.Compose([ToTensor()]))

digit_test_dataset = SchiDigitDataset(csv_file=test_data_path,
                                       transform=transforms.Compose([ToTensor()]))

# digit_train_dataset = SchiDigitDataset(csv_file='../data/data-deep-train.csv',
#                                        transform=transforms.Compose([Normalize(),ToTensor()]))
#
# digit_test_dataset = SchiDigitDataset(csv_file='../data/data-deep-test.csv',
#                                        transform=transforms.Compose([Normalize(),ToTensor()]))

# for i in range(len(digit_train_dataset)):
#     sample = digit_train_dataset[i]

    # print(i, sample['image'].shape, sample['label'].shape)

    # print(i, sample['image'].size(), sample['label'].size())

    # ax = plt.subplot(1, 4, i + 1)
    # plt.tight_layout()
    # ax.set_title('Sample #{}'.format(i))
    # ax.axis('off')
    # show_landmarks(**sample)
    #
    # if i == 3:
    #     plt.show()
    #     break

# digits_train_in = pd.read_csv('../data/data-deep-train.csv',header = None)
# digits_train_tensor = digits_train_in.as_matrix()
# digits_train= torch.from_numpy(digits_train_tensor)
#
# digits_test_in = pd.read_csv('../data/data-deep-test.csv',header = None)
# digits_test_tensor = digits_test_in.as_matrix()
# digits_test= torch.from_numpy(digits_test_tensor)



# train_dataset = digits_train
# test_dataset = digits_test



# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=digit_train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=digit_test_dataset, batch_size=1,
                                          # batch_size=batch_size,
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
        # self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # (hidden_size, int(hidden_size*0.75))
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)  # (int(hidden_size*0.75), hidden_size//2)
        self.fc4 = nn.Linear(hidden_size // 2, num_classes)
        # if isTrain == 1:
        #     dropout_prob = 0.25
        # else:
        #     dropout_prob = 0

        # self.dropout = nn.Dropout(dropout_prob)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        prob= 0.25
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, 0.25, training=self.training)
        # x = F.relu(self.fc2(x))
        # x = F.dropout(x, 0.25, training=self.training)
        # x = self.fc3(x)
        # return F.log_softmax(x)

        # first layer
        out = F.relu(self.fc1(x))
        out = F.dropout(out, prob, training=self.training)

        # second layer
        out = F.relu(self.fc2(out))
        out = F.dropout(out, prob, training=self.training)

        # third layer
        out = F.relu(self.fc3(out))
        out = F.dropout(out, prob, training=self.training)

        # out = self.fc1(x)
        # out = self.relu(out)
        # out = self.dropout(out)

        # out = self.fc2(out)
        # out = self.relu(out)
        # out = self.dropout(out)

        out = self.fc4(out)
        # out = F.log_softmax(out, dim=1)

        # out = self.sigmoid(out)
        return out


net = NeuralNet(input_size, num_classes)
# train_net = NeuralNet(input_size, num_classes, 1)
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
        # print(outputs.size())
        # print(outputs.type())
        # print(labels.size())
        # print(labels.type())

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if ep % 100 == 0 or ep == num_epochs - 1:
        # if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                ep, batch_idx * len(images), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

        # # if (i + 1) % 11 == 0:
        # if ((batch_idx + 1) % 11 == 0 and (epoch+1) % 100 ==0 ) :
        #
        #     print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.6f'
        #            % (epoch+1, num_epochs, batch_idx+1, len(digit_train_dataset)//batch_size, loss.item()))


def test(ep):
    net.eval()
    correct = 0
    total = 0
    acc = 0



    incorrect = []
    global max_accuracy
    global loss_out
    global epoch_out
    # global samples_out
    # global classifier_ans_out
    # global correct_ans_out
    global mat_out
    global numpy_mat_out

    # current_samples_out = []
    # current_classifier_ans_out = []
    # current_correct_ans_out = []

    current_mat_out = []
    current_numpy_mat_out = np.zeros((26,), dtype=int)

    # incorrect_classification = 0

    for test_sample in test_loader:
        images_test = Variable(test_sample['image'])
        labels_test = test_sample['label']
        labels_test = labels_test.view(labels_test.size(0))

        outputs_test = net(images_test)

        _, predicted = torch.max(outputs_test.data, 1)

        total += labels_test.size(0)
        correct += (predicted == labels_test).sum()

        # find incorrect indexes
        current_incorrect_bin = (predicted != labels_test).numpy()
        current_incorrect_indexes = np.nonzero(current_incorrect_bin)

        # incorrect.append(current_incorrect.tolist())

        # find compatible incorrect samples and save them in a list
        samples_numpy = images_test.numpy()
        incorrect_samples = samples_numpy[current_incorrect_indexes]
        incorrect_samples = incorrect_samples.astype(int)
        # current_samples_out.append(incorrect_samples.tolist())

        labels_pred_numpy = predicted.numpy()
        incorrect_labels = labels_pred_numpy[current_incorrect_indexes]
        incorrect_labels = incorrect_labels.astype(int)
        # current_classifier_ans_out.append(incorrect_labels.tolist())

        labels_actual_numpy= labels_test.numpy()
        true_labels = labels_actual_numpy[current_incorrect_indexes]
        true_labels = true_labels.astype(int)
        # current_correct_ans_out.append(true_labels.tolist())

        temp_labels = np.column_stack((incorrect_labels, true_labels))
        current_numpy_mat_out = np.concatenate((incorrect_samples, temp_labels), axis=1)
        length = len(current_numpy_mat_out.tolist())
        if length > 0:
            current_mat_out.extend(current_numpy_mat_out.tolist())
        # current_mat_out.append(current_numpy_mat_out.tolist())

    if ep % 100 == 0 or ep == num_epochs-1 :
        print('Accuracy of the model on the {} test images: {} %' .format(total, (100 * correct / total)))
        acc = 100 * correct / total

    if acc > max_accuracy:
        max_accuracy = acc.item()
        epoch_out = ep
        # loss_out= loss.item()

        # samples_out = current_samples_out
        # classifier_ans_out = current_classifier_ans_out
        # correct_ans_out = current_correct_ans_out
        mat_out = current_mat_out


    return


# incorrect_answers = []
# incorrect_samples = []

# Training the Model
for epoch in range(num_epochs):
    train(epoch)
    test(epoch)


print(max_accuracy)
print(epoch_out)
# print(loss_out.item())

with open(output_path,'w') as file:
    file.write('parameters: \n')
    parameters_one = 'hidden size: {} \t # epochs: {} \n'.format(hidden_size,num_epochs)
    parameters_two = 'batch size: {} \t learning rate: {:.3f} \n'.format(batch_size, learning_rate)
    file.write(parameters_one)
    file.write(parameters_two)
    file.write('results: \n')
    acc_result = 'best accuracy {} % \t, achieved in epoch # {} \n'.format(max_accuracy, epoch_out)
    file.write(acc_result)

    file.write('incorrect samples | incorrect labels | true labels \n')
    for item in mat_out:
        temp_str = ','.join(str(e) for e in item)
        if temp_str:  # if string is not empty
            file.write(temp_str)
            file.write('\n')

    # file.write('incorrect samples \n')
    # for item in samples_out:
    #     temp_str = ''.join(str(e) for e in item)
    #     if temp_str:  # if string is not empty
    #         file.write(temp_str)
    #         file.write('\n')
    # file.write('classifier bad answers \n')
    # for item in classifier_ans_out:
    #     temp_str= ''.join(str(e) for e in item)
    #     if temp_str:
    #         file.write(temp_str)
    #         file.write('\n')
    # file.write('actual answers \n')
    # for item in correct_ans_out:
    #     temp_str = ''.join(str(e) for e in item)
    #     if temp_str:
    #         file.write(temp_str)
    #         file.write('\n')

param_str = 'hidden size: {} \t # epochs: {} \t batch size: {} \t learning rate: {:.3f} \n'.format(hidden_size,num_epochs, max_accuracy, epoch_out)

with open(csv_out_path, 'w', newline='') as csvfile:
    mywriter = csv.writer(csvfile, delimiter=',')
    # mywriter.writerow(param_str)
    for item in mat_out:
        if len(item) > 0:
            # print(item)
            mywriter.writerow(item)
        # temp_str = ''.join(str(e) for e in item)
        # if temp_str:  # if string is not empty

            # mywriter.writerow('\n')

torch.save(net.state_dict(), '../data/model.pkl')

net.load_state_dict(torch.load('../data/model.pkl'))
# print(samples_out)
# print(classifier_ans_out)
# print(correct_ans_out)


    # # for i, (data) in enumerate(train_loader):
    # # for i,(images, labels) in enumerate(train_loader):
    #
    # # print(images)
    # # print (data)
    # # images = Variable(images)
    # # # images = Variable(images.view(-1, 28*28))
    # # labels = Variable(labels)
    #
    # for batch_idx, sample_batched in enumerate(train_loader):
    #     # print(i, sample_batched['image'].size(),
    #     #       sample_batched['label'].size())
    #     # if i==3:
    #     #     break
    #
    #     # if (i) % 10 == 0:
    #     #     print(i)
    #     # print (epoch)
    #     images = Variable(sample_batched['image'])
    #     # print(images.shape)
    #     labels = Variable(sample_batched['label'])
    #     # print(labels.size(0))
    #     # print(labels.size(1))
    #     labels = labels.view(labels.size(0))
    #     # print(labels.shape)
    #     #
    #     # Forward + Backward + Optimize
    #     optimizer.zero_grad()
    #     outputs = net(images)
    #     # outputs = train_net(images)
    #     # if (i<1):
    #         # print(outputs.shape)
    #         # print(labels.shape)
    #         # break
    #     loss = criterion(outputs, labels)
    #     loss.backward()
    #     optimizer.step()
    #
    #     # if batch_idx % log_interval == 0:
    #     #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #     #         epoch, batch_idx * len(images), len(train_loader.dataset),
    #     #                100. * batch_idx / len(train_loader), loss.data[0]))
    #
    #     # if (i + 1) % 11 == 0:
    #     if ((batch_idx + 1) % 11 == 0 and (epoch+1) % 100 ==0 ) :
    #     # if (i+1) % 200 == 0:
    #         print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.6f'
    #                % (epoch+1, num_epochs, batch_idx+1, len(digit_train_dataset)//batch_size, loss.item()))
    
# # Test the Model
# # test_net = NeuralNet(input_size, num_classes, 0)
# correct = 0
# total = 0
# # test_loss = 0
# for test_sample in test_loader:
#     images_test = Variable(test_sample['image'])
#     labels_test = test_sample['label']
#     labels_test = labels_test.view(labels_test.size(0))
#
#     outputs_test = net(images_test)
#
#
# #     test_loss += criterion(outputs_test, labels_test).data[0]
# #     pred = outputs_test.data.max(1)[1]  # get the index of the max log-probability
# #     correct += pred.eq(labels_test.data).sum()
# #
# # test_loss /= len(test_loader.dataset)
# # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
# #         test_loss, correct, len(test_loader.dataset),
# #         100. * correct / len(test_loader.dataset)))
#
#     _, predicted = torch.max(outputs_test.data, 1)
#     # print(predicted.shape)
#     # print(labels_test.shape)
#
#     total += labels_test.size(0)
#     correct += (predicted == labels_test).sum()
#
# print('Accuracy of the model on the 1000 test images: %d %%' % (100 * correct / total))

# # Save the Model
# torch.save(model.state_dict(), 'model.pkl')


