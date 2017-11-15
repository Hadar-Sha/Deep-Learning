# -*- coding: utf-8 -*-

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.
See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn import datasets
from tensorflow.python.platform import gfile
import csv

import tensorflow as tf
import numpy as np

FLAGS = None

SCHI_TRAINING = './data/data-deep-train.csv'
SCHI_TEST = './data/data-deep-test.csv'
SCHI_VALIDATION = './data/data-deep-validation.csv'

input_data_dir= SCHI_TRAINING

#training_set = datasets.base.load_csv_without_header(filename=SCHI_TRAINING,target_dtype=np.int,features_dtype=np.int)
#test_set = datasets.base.load_csv_without_header(filename=SCHI_TEST,target_dtype=np.int,features_dtype=np.int)
#validation_set = datasets.base.load_csv_without_header(filename=SCHI_VALIDATION,target_dtype=np.int,features_dtype=np.int)

batch_size= 100

def my_next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def my_load_csv_without_header(filename,target_dtype,features_dtype,target_column=-1, target_len=1):
  """Load dataset from CSV file without a header row."""
  with gfile.Open(filename) as csv_file:
    data_file = csv.reader(csv_file)
    data, target = [], []
    for row in data_file:
      target.append(row.pop(target_column))
      data.append(np.asarray(row, dtype=features_dtype))

  target = np.array(target, dtype=target_dtype)
  data = np.array(data)
  return Dataset(data=data, target=target)
  
def fill_feed_dict(data_set,images_pl, labels_pl):
  """Fills the feed_dict for training the given step.

  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }

  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().

  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  
  
  images_input= data_set.data

  labels_input= data_set.target

  images_feed, labels_feed = my_next_batch(batch_size,images_input,labels_input)
  
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  
  return feed_dict
    
def main(_):
  # Import data
#  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  training_set = datasets.base.load_csv_without_header(filename=SCHI_TRAINING,target_dtype=np.int,features_dtype=np.int)
  test_set = datasets.base.load_csv_without_header(filename=SCHI_TEST,target_dtype=np.int,features_dtype=np.int)
  validation_set = datasets.base.load_csv_without_header(filename=SCHI_VALIDATION,target_dtype=np.int,features_dtype=np.int)
  
#  training_concat= 

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  print(training_set)
  # Train
  for i in range(1000):
    batch_xs, batch_ys = my_next_batch(100,training_set.data,training_set.target)
#    batch_xs, batch_ys = mnist.train.next_batch(100)
    if i%1000==0:
        print("x shape is: " + str(batch_xs.shape))
        print("y shape is: " + str(batch_ys.shape))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
#                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)