# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 23:24:03 2017

@author: H
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import csv

#from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

#i=0
#datasize=1000
#training_data=[]
#validation_data=[]
#test_data=[]
DEEP_TRAINING='./data/data-deep-train.csv'
DEEP_TEST='./data/data-deep-test.csv'

'''with open('data-deep.csv', newline='') as csvfile:
    filereader = csv.reader(csvfile) # , delimiter=' ', quotechar='|'
    for row in filereader:
        if i<0.8*1000:
            training_data.append[row[1:25]]
        if i<0.9*1000:
            validation_data.append[row[1:25]]
        else:
            test_data.append[row[1:25]]
        #print(', '.join(row))'''

def create_file_reader_ops(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = reader.read(filename_queue)
    record_defaults = [[0]]+[[""]]*25
    
    index,topR,topG,topB,medR,medG,medB,lowR,lowG,lowB,lTopR,lTopG,lTopB,lBotR,lBotG,lBotB,rTopR,rTopG,rTopB,rBotR,rBotG,rBotB,BgR,BgG,BgB,digit = tf.decode_csv(csv_row, record_defaults=record_defaults)

    
#    topSeg = tf.stack([topR,topG,topB])
#    medSeg = tf.stack([medR,medG,medB])
#    botSeg = tf.stack([lowR,lowG,lowB])
#    leftTopSeg= tf.stack([lTopR,lTopG,lTopB])
#    leftBotSeg= tf.stack([lBotR,lBotG,lBotB])
#    RightTopSeg= tf.stack([rTopR,rTopG,rTopB])
#    RightBotSeg= tf.stack([rBotR,rBotG,rBotB])
#    Background= tf.stack([BgR,BgG,BgB])
    
    features = tf.stack([topR,topG,topB,medR,medG,medB,lowR,lowG,lowB,lTopR,lTopG,lTopB,lBotR,lBotG,lBotB,rTopR,rTopG,rTopB,rBotR,rBotG,rBotB,BgR,BgG,BgB])

    return features, digit
#    return topSeg,medSeg,botSeg,leftTopSeg,leftBotSeg,RightTopSeg,RightBotSeg,Background, digit
        
FLAGS = None

filename_queue = tf.train.string_input_producer([DEEP_TRAINING])
features, digit= create_file_reader_ops(filename_queue)

with tf.Session() as sess:
  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  for i in range(800):
    # Retrieve a single instance:
    example, label = sess.run([features, digit])

  coord.request_stop()
  coord.join(threads)
#with tf.Session() as sess:
#    tf.global_variables_initializer().run()
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    while True:
#        try:
#            example_data, country_name = sess.run([example, country])
#            print(example_data, country_name)
#        except tf.errors.OutOfRangeError:
#            break





    
  # Define the training inputs
#def get_train_inputs():
#    x = tf.constant(training_set.data)
#    y = tf.constant(training_set.target)
#
#    return x, y
#    
def get_test_inputs():
#    x = tf.constant(test_set.data)
#    y = tf.constant(test_set.target)
    filename_queue = tf.test.string_input_producer([DEEP_TEST])
    features, digit= create_file_reader_ops(filename_queue)
    
    with tf.Session() as sess:
      # Start populating the filename queue.
      coord = tf.test.Coordinator()
      threads = tf.test.start_queue_runners(coord=coord)
    
      for i in range(100):
        # Retrieve a single instance:
        example, label = sess.run([features, digit])
    
      coord.request_stop()
      coord.join(threads)

    return example, label
#  
#def main(_):
x = tf.placeholder(tf.float32, [None, 24])
W = tf.Variable(tf.zeros([24, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
# Train
for _ in range(800):
    batch_xs, batch_ys = features.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.int))

#x1,y1=get_test_inputs()

print(sess.run(accuracy, feed_dict={x: x1, y_: y1}))
#    
#    
#    
##if __name__ == '__main__':
##    a=1
##    main()