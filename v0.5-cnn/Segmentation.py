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

    x,y_=bg.create_input_layer(625,5)


    # -- 1st Convolutional layer  -------------------------------------------
    new_shape=[-1, 25, 25, 1]
    w_shape=[5, 5, 1, 32]                               # filter size - receptive field size is 5x5
    b_shape=[32]
    h_conv1=bg.create_1st_conv_layer( x, new_shape, w_shape, b_shape) # h_conv1.shape=13x13

    # pooling layer
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    h_pool1 = bg.max_pool_2x2(h_conv1,ksize,strides)    # it reduces pathces from 25x25 to 13x13

    # -- 2nd Convolutional layer  -------------------------------------------

    w_shape = [3, 3, 32, 64]                             # 3x3 filter and 64 neurons.
    b_shape = [64]
    h_conv1 = bg.create_next_conv_layer(h_pool1, w_shape, b_shape)

    # pooling layer
    ksize = [1, 2, 2, 1]
    strides = [1, 2, 2, 1]
    h_pooln = bg.max_pool_2x2(h_conv1, ksize, strides)

    # --fully connected layer---------------------------------------------
    w_shape=[7 * 7 * 64, 1024]
    b_shape=[1024]
    flat_shape=[-1, 7 * 7 * 64]
    h_fc1=bg.create_fully_connected_layer(h_pooln, w_shape, b_shape,flat_shape)


    # ---Drop out--------------------------------------------------------------------
    h_fc1_drop,keep_prob=bg.drop_out(h_fc1)


    # ---- OUTPUT LAYER -------------------------------------------------------------
    w_shape=[1024, 5]
    b_shape=[5]
    y_conv,cross_entropy=bg.create_output_layer(h_fc1_drop, w_shape, b_shape, y_)


    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # run graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(1023):
        batch_xs, batch_ys = itr.get_next_batch(train_x, train_y, 30)
        if(i%100==0):
            train_accuracy = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys ,keep_prob:1.0}) # keep_prob: 1.0
            print("step %d, training accuracy %g"% (i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys,keep_prob:1.0})  # Y_ is actual labels, Y are predicted labels

    #==== Test trained model ============================================
    test_x = br.read_test_patches(test_x)
    feed_dict = feed_dict={x: test_x, y_: test_y,keep_prob:1.0}
    print("final accuracy is :",sess.run(accuracy, feed_dict))


# Two layerd neural network
if __name__ == '__main__':
    my_app()












