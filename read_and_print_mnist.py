# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:11:25 2017

@author: H
"""

from tensorflow.examples.tutorials.mnist import input_data

input_data_dir= 'C:/tmp/tensorflow\mnist\input_data'

data_sets = input_data.read_data_sets(input_data_dir, False)
print (data_sets)

