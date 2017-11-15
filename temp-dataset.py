# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 19:46:11 2017

@author: H
"""


"""
Created on Sat Oct 14 15:08:14 2017

@author: H
"""

import numpy as np
import tensorflow as tf
import math

import time
import os

NUM_CLASSES = 10
VECTORE_SIZE = 24
DATA_SIZE= 1000
TRAINING_DATA_SIZE= int(0.8*DATA_SIZE)

# Data sets
SCHI_TRAINING = './data/data-deep-train.csv'
SCHI_TEST = './data/data-deep-test.csv'
SCHI_VALIDATION = './data/data-deep-validation.csv'

learning_rate= 0.01
max_steps= 2000
hidden1= 24
hidden2= 12
batch_size= 10
input_data_dir= SCHI_TRAINING
log_dir= './log-Hadar'

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_without_header(filename=SCHI_TRAINING,target_dtype=np.int,features_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv_without_header(filename=SCHI_TEST,target_dtype=np.int,features_dtype=np.int)
validation_set = tf.contrib.learn.datasets.base.load_csv_without_header(filename=SCHI_VALIDATION,target_dtype=np.int,features_dtype=np.int)

features= training_set[0]
labels= training_set[1]

#sess = tf.Session()
#sess.run(tf.global_variables_initializer())

#sess.run([training_set,test_set])