# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 18:25:28 2017

@author: H
"""

import tensorflow as tf

#graph = tf.get_default_graph()
#
#input_value = tf.constant(1.0)
#
#operations= graph.get_operations()
##operations
#
##print (operations[0].node_def)
##print (input_value)
#weight = tf.Variable(0.8)
##for op in graph.get_operations(): print(op.name)
#
#output_value = weight * input_value
#
#op = graph.get_operations()[-1]
#print (op.name)
#
#for op_input in op.inputs: print(op_input)
#
sess = tf.Session()
#init = tf.global_variables_initializer()
#sess.run(init)
#sess.run(output_value)
#
#print(output_value)

x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.multiply(w, x, name='output')

y_ = tf.constant(0.0)
loss = (y - y_)**2
summary_y = tf.summary.scalar('output', y)
#optim = tf.train.GradientDescentOptimizer(learning_rate=0.025)
#grads_and_vars = optim.compute_gradients(loss)

summary_writer = tf.summary.FileWriter('log_simple_graph', sess.graph)
sess.run(tf.global_variables_initializer())
#sess.run(grads_and_vars[0][0])
#sess.run(optim.apply_gradients(grads_and_vars))
train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)
for i in range(100):
     summary_str = sess.run(summary_y)
     summary_writer.add_summary(summary_str, i)
     sess.run(train_step)

print(sess.run(y))
#print(sess.run(w))

#grads_and_vars.clear()

#sess.close()

#summary_writer = tf.summary.FileWriter('log_simple_graph', sess.graph)