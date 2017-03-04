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

FLAGS = None


br=BRATS()
itr=Training_batch_iterator()
bg=Build_Graph()

def my_app():
    global br
    train_x, train_y, test_x, test_y = br.get_test_n_train_data('BRATS-Training/**/**/**/*.mha')

    """
    CREATE MODEL
    """
    # ===  INPUT LAYER ==
    x = tf.placeholder(tf.float32, [None, 625])
    y_ = tf.placeholder(tf.float32, [None, 5])

    # === OUTPUT LAYER ==
    W = bg.weight_variable([625, 5])
    b = bg.bias_variable([5])
    y = tf.matmul(x, W) + b

    # Define loss and optimizer for forward pass
    pred = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)

    # ====== TRAINING STEP ==
    cross_entropy = tf.reduce_mean(pred)

    # define the training step of model
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # GRAPH  is completly specified so now we run it
    #========================================================================
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for _ in range(1023):
        batch_xs, batch_ys = itr.get_next_batch(train_x, train_y, 30)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})  # Y_ is actual labels, Y are predicted labels

    #==== Test trained model ============================================
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # test_x only contains labels. Read its data again
    test_x = br.read_test_patches(test_x)
    feed_dict = {x: test_x, y_: test_y}
    print(sess.run(accuracy, feed_dict))


# Two layerd neural network
if __name__ == '__main__':
    my_app()











