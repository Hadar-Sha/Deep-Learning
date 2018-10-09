# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:20:20 2017

@author: H
"""
import tensorflow as tf
import math

import time

NUM_CLASSES = 10
VECTORE_SIZE = 24
DATA_SIZE= 1000
TRAINING_DATA_SIZE= int(0.8*DATA_SIZE)

DEEP_TRAINING='./data/data-deep-train.csv'
DEEP_TEST='./data/data-deep-test.csv'

learning_rate= 0.01
max_steps= 2000
hidden1= 24
hidden2= 12
batch_size= 10
input_data_dir= DEEP_TRAINING
log_dir= './log-Hadar'
#log_dir= 'C:/Users/H/Documents/Haifa Univ/Thesis/log-Hadar'


def create_file_reader_ops(filename_queue):
    reader = tf.TextLineReader()
    _, csv_row = reader.read(filename_queue)
    record_defaults = [[0.0]]+[[0.0]]*24+[[0]]
    
    index,topR,topG,topB,medR,medG,medB,lowR,lowG,lowB,lTopR,lTopG,lTopB,lBotR,lBotG,lBotB,rTopR,rTopG,rTopB,rBotR,rBotG,rBotB,BgR,BgG,BgB,digit = tf.decode_csv(csv_row, record_defaults=record_defaults)
    
    features = tf.stack([topR,topG,topB,medR,medG,medB,lowR,lowG,lowB,lTopR,lTopG,lTopB,lBotR,lBotG,lBotB,rTopR,rTopG,rTopB,rBotR,rBotG,rBotB,BgR,BgG,BgB])
    labels= tf.stack([digit])
#    features= tf.reshape(features,[1,VECTORE_SIZE])
#    digit= tf.reshape(digit,[1,1])
#    outdigit= tf.stack([digit])
    return features, labels

def inference(images, hidden1_units, hidden2_units):
  """Build the SCHI model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
#  print(images)
  images_sh= tf.reshape(images,[1,24])
#  print(images_sh)

#  sess= tf.Session()
#  with sess.as_default():
#      temp= images.eval()
#  print(temp)
  
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(tf.truncated_normal([VECTORE_SIZE, hidden1_units],
                            stddev=1.0 / math.sqrt(float(VECTORE_SIZE))), name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images_sh, weights) + biases)
#    hidden1 = tf.nn.relu(tf.matmul(images_sh, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),name='biases')
    logits = tf.matmul(hidden2, weights) + biases
  
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
  print("logits shape is "+ str(logits))
  print("labels shape is "+ str(labels))
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))
  
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
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         VECTORE_SIZE))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder

def fill_feed_dict(images_pl, labels_pl):
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
  
#  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,FLAGS.fake_data)
  filename_queue = tf.train.string_input_producer([DEEP_TRAINING])
#  features, label= create_file_reader_ops(filename_queue)
  images_feed, labels_feed = create_file_reader_ops(filename_queue)

#  print(images_feed[0])
#  print(labels_feed)

#  temp= tf.Session().run(images_feed[0])
#  print(type(temp))
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  print(feed_dict)
  return feed_dict

  


def run_training():
  """Train MNIST for a number of steps."""
#  # Get the sets of images and labels for training, validation, and
#  # test on MNIST.
#  data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)

#  filename_queue = tf.train.string_input_producer([DEEP_TRAINING])
#  features, label= create_file_reader_ops(filename_queue)
  
  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    
    # Generate placeholders for the images and labels.
#    images_placeholder, labels_placeholder = placeholder_inputs(batch_size)

    # Build a Graph that computes predictions from the inference model.
    filename_queue = tf.train.string_input_producer([DEEP_TRAINING])
    images_feed, labels_feed = create_file_reader_ops(filename_queue)
    logits = inference(images_feed, hidden1, hidden2)

    # Add to the Graph the Ops for loss calculation.
#    print("logits shape is "+ str(logits))
#    print(tf.shape(logits))
#    print("labels shape is "+ str(labels_feed))
#    labels_sh= tf.reshape(labels,[1,24])
    
    loss_res = loss(logits, labels_feed)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = training(loss_res, learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = evaluation(logits, labels_feed)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

    # And then after everything is built:

    # Run the Op to initialize the variables.
    sess.run(init)
#    filename_queue = tf.train.string_input_producer([DEEP_TRAINING])
#    images_feed, labels_feed = create_file_reader_ops(filename_queue)

    # Start the training loop.
    for step in range(TRAINING_DATA_SIZE):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
#      feed_dict = fill_feed_dict(images_placeholder,labels_placeholder)
      
      

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss_res])
#      _, loss_value = sess.run([train_op, loss],feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary)
#        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

###### main ######
#if tf.gfile.Exists(log_dir):
#    tf.gfile.DeleteRecursively(log_dir)

if not tf.gfile.Exists(log_dir):
    tf.gfile.MakeDirs(log_dir)
run_training()
#with tf.Graph().as_default():
#    filename_queue = tf.train.string_input_producer([DEEP_TRAINING])
    
#    features, label= create_file_reader_ops(filename_queue)
    
    #printerop = tf.Print(label, [features, label], name='printer')
    
    #print(features)
    #print(label)
    #training_set=tf.placeholder(tf.int32, shape=(TRAINING_DATA_SIZE,VECTORE_SIZE+1))
    
    #training_set= tf.concat(1,[features,label])
    ########### main ##########
    
    
#    with tf.Session() as sess:
#      # Start populating the filename queue.
#      coord = tf.train.Coordinator()
#      threads = tf.train.start_queue_runners(coord=coord)
#      
#    
#      for i in range(TRAINING_DATA_SIZE):
#        # Retrieve a single instance:
##        example, digit = sess.run(printerop, feed_dict={features:features, label:label})
#        
#        example, digit = sess.run([features, label])
#    #    print(example,digit)
#    
#      coord.request_stop()
#      coord.join(threads)
      
#    run_training()