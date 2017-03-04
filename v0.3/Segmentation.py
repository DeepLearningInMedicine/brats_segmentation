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
from prepare_graph import Build_Graph
import tensorflow as tf

br=BRATS()
itr=Training_batch_iterator()
bg=Build_Graph()

def my_app():
    global br
    train_x, train_y, test_x, test_y = br.get_test_n_train_data('BRATS-Training/**/**/**/*.mha')

    """ CREATE MODEL """

    # ---------INPUT LAYER ---------------------------------
    x = tf.placeholder(tf.float32, [None, 625])
    y_ = tf.placeholder(tf.float32, [None, 5])


    # -- Convolutional layer  -------------------------------------------
    x_image = tf.reshape(x, [-1, 25, 25, 1])        # x=625, x_image=25x25
    W_conv1 = bg.weight_variable([5, 5, 1, 32])     # [f-width,f-height,input channels,output channels]
    b_conv1 = bg.bias_variable([32])
    h_conv1 = bg.conv2d(x_image,W_conv1)+b_conv1    # dimension of x_image and W_conv1 must be same

    h_pool1 = bg.max_pool_2x2(h_conv1)

    # --fully connected layer---------------------------------------------
    W_fc1 = bg.weight_variable([13 * 13 * 32, 1024])
    b_fc1 = bg.bias_variable([1024])
    h_pool1_flat = tf.reshape(h_pool1, [-1, 13 * 13 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)


    # ---Drop out--------------------------------------------------------------------
    #keep_prob = tf.placeholder(tf.float32)
    #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob), keep_prob: 0.5

    # ---- OUTPUT LAYER -------------------------------------------------------------
    W_fc2 = bg.weight_variable([1024, 5])
    b_fc2 = bg.bias_variable([5])
    y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
    pred = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    cross_entropy = tf.reduce_mean(pred)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # run graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(1023):
        batch_xs, batch_ys = itr.get_next_batch(train_x, train_y, 30)
        if(i%100==0):
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys }) # keep_prob: 1.0
            print("step %d, training accuracy %g"% (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  # Y_ is actual labels, Y are predicted labels

    #==== Test trained model ============================================
    test_x = br.read_test_patches(test_x)
    feed_dict = feed_dict={x: test_x, y_: test_y}
    print(sess.run(accuracy, feed_dict))


# Two layerd neural network
if __name__ == '__main__':
    my_app()












