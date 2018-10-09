# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 14:11:01 2018

@author: H
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils

digits_train = pd.read_csv('../data/data-deep-train.csv',header = None)
digits_train_tensor = digits_train.as_matrix()

dd= torch.from_numpy(digits_train_tensor)

training_data= digits_train_tensor[: , :24]
training_labels= digits_train_tensor[: , 24]
print(training_data.shape)
print(training_labels.shape)

#x= digits_train_tensor[:,]
print (digits_train_tensor[0])
print (dd)
print (digits_train_tensor.dtype)

#print (digits_train[0])
