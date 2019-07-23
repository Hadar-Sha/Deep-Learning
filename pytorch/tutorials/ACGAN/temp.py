import torch
import numpy as np

batchSize = 1
nz = 110
batch_size = batchSize
num_classes = 10
noise = torch.FloatTensor(batchSize, nz, 1, 1)
eval_noise = torch.FloatTensor(batchSize, nz, 1, 1).normal_(0, 1)

noise.data.resize_(batch_size, nz, 1, 1).normal_(0, 1)
label = np.random.randint(0, num_classes, batch_size)
noise_ = np.random.normal(0, 1, (batch_size, nz))
class_onehot = np.zeros((batch_size, num_classes))
class_onehot[np.arange(batch_size), label] = 1
noise_[np.arange(batch_size), :num_classes] = class_onehot[np.arange(batch_size)]
noise_ = (torch.from_numpy(noise_))
noise.data.copy_(noise_.view(batch_size, nz, 1, 1))
# aux_label.data.resize_(batch_size).copy_(torch.from_numpy(label))

a=3
b=9