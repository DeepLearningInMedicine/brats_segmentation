import glob
import PIL
import os
import fnmatch
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data
from prepare_data import BRATS
from prepare_data import Training_batch_iterator
import tensorflow as tf

FLAGS = None


br=BRATS()
itr=Training_batch_iterator()


if __name__ == '__main__':
    global br
    train_x,train_y,test_x,test_y=br.get_test_n_train_data('BRATS-Training/**/**/**/*.mha')

    # Create the model
    x = tf.placeholder(tf.float32, [None, 625])
    W = tf.Variable(tf.zeros([625, 5]))
    b = tf.Variable(tf.zeros([5]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 5])

    pred=tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    cross_entropy = tf.reduce_mean(pred)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for _ in range(1023):
        batch_xs, batch_ys = itr.get_next_batch(train_x,train_y,30)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # test_x only contains labels. Read its data again
    test_x=br.read_test_patches(test_x)

    print(sess.run(accuracy, feed_dict={x: test_x,
                                        y_: test_y}))












