# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 15:08:14 2017

@author: H
"""

import numpy as np
import tensorflow as tf
import math

from tensorflow.contrib.data import Dataset

import argparse
import os
import sys
import time
import matplotlib.pyplot as plt

FLAGS = None

NUM_CLASSES = 10
VECTORE_SIZE = 24

#DATA_SIZE= 1000
#TRAINING_DATA_SIZE= int(0.8*DATA_SIZE)

# Data sets
# SCHI_TRAINING = '../data/w-o_images/data-deep-train.csv'
# SCHI_TEST = '../data/w-o_images/data-deep-test.csv'
# SCHI_VALIDATION = '../data/w-o_images/data-deep-validation.csv'

num_of_epoch= 100
max_steps= 10000
learning_rate= 0.01
batch_size= 100

# SCHI_TRAINING = '../data/data-deep-train.csv'
# SCHI_TEST = '../data/data-deep-test.csv'
# SCHI_VALIDATION = '../data/data-deep-validation.csv'

# SCHI_TRAINING = '../data/exp-only/exp-data-train.csv'

SCHI_TRAINING = '../data/exp-data-train.csv'
SCHI_TEST = '../data/exp-data-test.csv'
SCHI_VALIDATION = '../data/exp-data-validation.csv'

# learning_rate= 0.005


# max_steps= 2000
hidden1= 120
hidden2= 60
#hidden1= 180
#hidden2= 120
# batch_size= 100
input_data_dir= SCHI_TRAINING
# num_of_epoch= 200

my_keep_prob = 0.75
test_keep_prob= 1

precision_list=[]

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_without_header(filename=SCHI_TRAINING,target_dtype=np.int,features_dtype=np.int)
test_set = tf.contrib.learn.datasets.base.load_csv_without_header(filename=SCHI_TEST,target_dtype=np.int,features_dtype=np.int)
validation_set = tf.contrib.learn.datasets.base.load_csv_without_header(filename=SCHI_VALIDATION,target_dtype=np.int,features_dtype=np.int)

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.

  Args:
    batch_size: The batch size will be baked into both placeholders.

  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,VECTORE_SIZE))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  keep_prob = tf.placeholder(tf.float32)
  return images_placeholder, labels_placeholder, keep_prob
  
def inference(images, hidden1_units, hidden2_units,keep_prob):
  """Build the SCHI model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """ 
  # Hidden 1 - with dropout
  with tf.name_scope('hidden1'):
    weights = tf.Variable(tf.truncated_normal([VECTORE_SIZE, hidden1_units],
                            stddev=1.0 / math.sqrt(float(VECTORE_SIZE))), name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
#    hidden1 = tf.nn.relu(tf.matmul(images_sh, weights) + biases)

    # keep_prob = tf.placeholder(tf.float32)
    # hidden1_drop= tf.nn.dropout(hidden1, keep_prob)
    hidden1_drop = hidden1

  # Hidden 2- with dropout
  with tf.name_scope('hidden2'):
    weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1_drop , weights) + biases)
    # hidden2_drop = tf.nn.dropout(hidden2, keep_prob)
    hidden2_drop = hidden2
    # hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),name='biases')
    logits = tf.matmul(hidden2_drop, weights) + biases
    # logits = tf.matmul(hidden2, weights) + biases
  
  return logits
    
def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')  

def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.  
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))

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

  
def fill_feed_dict(data_set,images_pl, labels_pl, keep_prob, isTest):
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

  prob_val= my_keep_prob
  if isTest:
      prob_val= test_keep_prob

  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
      keep_prob: prob_val
  }
  
  return feed_dict

def do_eval(sess,eval_correct,images_placeholder,labels_placeholder,data_set, keep_prob, isTest):
  """Runs one evaluation against the full epoch of data.

  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  num_of_examples= data_set.target.shape[0]
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = num_of_examples // batch_size
  num_examples = steps_per_epoch * batch_size
  for step in range(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,images_placeholder,labels_placeholder,keep_prob, isTest)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  precision_list.append(precision)
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))
  return
  
def run_training():
  """Train SCHI digits for a number of steps."""

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
      with tf.name_scope('input'):
      # Input data, pin to CPU because rest of pipeline is CPU-only
          with tf.device('/cpu:0'):
              input_images = tf.constant(training_set[0])
              input_labels = tf.constant(training_set[1])
    
          image, label = tf.train.slice_input_producer([input_images, input_labels], num_epochs=num_of_epoch)
          label = tf.cast(label, tf.int32)
          images, labels = tf.train.batch([image, label], batch_size=batch_size)
          # Generate placeholders for the images and labels.
      images_placeholder, labels_placeholder, keep_prob = placeholder_inputs(batch_size)

        # Build a Graph that computes predictions from the inference model.
      logits = inference(images_placeholder,hidden1,hidden2,keep_prob)

    # Add to the Graph the Ops for loss calculation.
      loss_res = loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
      train_op = training(loss_res, learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
      eval_correct = evaluation(logits, labels_placeholder)

    # Build the summary Tensor based on the TF collection of Summaries.
      summary = tf.summary.merge_all()

    # Add the variable initializer Op.
      init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
      saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
      sess = tf.Session()

    # # Instantiate a SummaryWriter to output summaries and the Graph.
    #   summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
      sess.run(init)

    # Start the training loop.
      for step in range(max_steps):
          start_time = time.time()

          # print(step)
          # print(num_of_epoch)

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
          feed_dict = fill_feed_dict(training_set,images_placeholder,labels_placeholder, keep_prob, False)
#          print(feed_dict)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
          _, loss_value = sess.run([train_op, loss_res],feed_dict=feed_dict)
    
          duration = time.time() - start_time
    
          # Write the summaries and print an overview fairly often.
          if step % 100 == 0:
            # Print status to stdout.
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
            # Update the events file.
            summary_str = sess.run(summary, feed_dict=feed_dict)
            # summary_writer.add_summary(summary_str, step)
            # summary_writer.flush()
    
          # Save a checkpoint and evaluate the model periodically.
          if (step + 1) % 1000 == 0 or (step + 1) == max_steps:

            # checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
            # saver.save(sess, checkpoint_file, global_step=step)

            # Evaluate against the training set.
            print('Training Data Eval:')
            do_eval(sess,eval_correct,images_placeholder,labels_placeholder,training_set, keep_prob, False)
            # Evaluate against the validation set.
            print('Validation Data Eval:')
            do_eval(sess,eval_correct,images_placeholder,labels_placeholder,validation_set, keep_prob, True)
            # Evaluate against the test set.
            print('Test Data Eval:')
            do_eval(sess,eval_correct,images_placeholder,labels_placeholder,test_set, keep_prob, True)

#          return
      #plt.plot(precision_list)
      #plt.show()
      return
        
######## main #########
#if not tf.gfile.Exists(log_dir):
#    tf.gfile.MakeDirs(log_dir)
##else:
##    tf.gfile.DeleteRecursively(log_dir)
#run_training()

def main(_):
  # if tf.gfile.Exists(FLAGS.log_dir):
  #     #a=1
  #   tf.gfile.DeleteRecursively(FLAGS.log_dir)
  # tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()
  return

if __name__ == '__main__':
    tf.app.run(main=main)

#   parser = argparse.ArgumentParser()
#   parser.add_argument(
#       '--learning_rate',
#       type=float,
#       default=0.01,
#       help='Initial learning rate.'
#   )
#   parser.add_argument(
#       '--max_steps',
#       type=int,
#       default=2000,
#       help='Number of steps to run trainer.'
#   )
#   parser.add_argument(
#       '--hidden1',
#       type=int,
#       default=18,
# #      default=128,
#       help='Number of units in hidden layer 1.'
#   )
#   parser.add_argument(
#       '--hidden2',
#       type=int,
#       default=12,
# #      default=32,
#       help='Number of units in hidden layer 2.'
#   )
#   parser.add_argument(
#       '--batch_size',
#       type=int,
#       default=100,
#       help='Batch size.  Must divide evenly into the dataset sizes.'
#   )
#   parser.add_argument(
#       '--input_data_dir',
#       type=str,
# #      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),'tensorflow/mnist/input_data'),
#       default= SCHI_TRAINING,
#       help='Directory to put the input data.'
#   )
#   parser.add_argument(
#       '--log_dir',
#       type=str,
# #      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),'tensorflow/mnist/logs/fully_connected_feed'),
#       default= os.path.expanduser(os.getenv('USERPROFILE'))+'/Documents/Haifa Univ/Thesis/deep-learning/log-folder',
# #      default= os.path.expanduser(os.getenv('USERPROFILE'))+'/Documents/Haifa Univ/Thesis/deep learning/log-folder',
#       help='Directory to put the log data.'
#   )
#   parser.add_argument(
#       '--fake_data',
#       default=False,
#       help='If true, uses fake data for unit testing.',
#       action='store_true'
#   )
#
#   # parser.add_argument(
#   #     '--h_keep_prob',
#   #     default=0.5,
#   #     help='keep probability of each neuron'
#   # )
#
#   FLAGS, unparsed = parser.parse_known_args()

  # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
