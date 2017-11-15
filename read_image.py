# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 16:30:10 2017

@author: H
"""
import numpy as np
import tensorflow as tf

fn = '3.png'
image_contents = tf.read_file(fn)
image = tf.image.decode_image(image_contents, channels=3)


with tf.Session() as sess:
    image_tensor= sess.run(image)
    print (image_tensor)
#    img_resize = tf.image.resize_images(image, [28,28])
#    im= sess.run(img_resize)

#    image.shape
    
#    image.eval().shape
    
#    