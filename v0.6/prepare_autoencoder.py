
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class AutoEncoder(object):
    def __init__(self):
        self.learning_rate=0.01
        self.X=None

    def set_parameters(self,learning_rate,training_epochs,batch_size):
        self.learning_rate=learning_rate
        self.training_epochs=training_epochs
        self.batch_size=batch_size

    def create_layer(self,input,W,b):
        prod=tf.matmul(input,W)+b
        return tf.nn.sigmoid(prod)

    def encoder(self,input,W1,W2,W3,b1,b2,b3):
        layer_1 = self.create_layer(input,W1,b1)
        layer_2=self.create_layer(layer_1,W2,b2)
        layer_3 = self.create_layer(layer_2, W3, b3)
        return layer_3

    def decoder(self,input,W1,W2,W3,b1,b2,b3):
        """
        This method upscales the downscaled image
        :param input:   [?,64]
        :param W1:      [64,128]
        :param W2:      [128,256]
        :param W3:      [256,784]
        :param b1:      [128]
        :param b2:      [256]
        :param b3:      [784]
        :return:        [?,784]
        """
        layer_1=self.create_layer(input,W1,b1)
        layer_2=self.create_layer(layer_1,W2,b2)
        layer_3=self.create_layer(layer_2,W3,b3)
        return layer_3

    def decoder_new(self,input,W1,W2,W3,b1,b2,b3):
        # downscaled images

        layer_1=self.create_layer(input,W1,b1)
        layer_2=self.create_layer(layer_1,W2,b2)
        layer_3=self.create_layer(layer_2,W3,b3)
        return layer_3

    def train(self,sess,total_batch,epochs,batch,optimizer,cost):
            # Training cycle
            for epoch in range(epochs):
                # Loop over all batches
                for i in range(total_batch):
                    batch_xs, batch_ys = batch[0],batch[1]
                    # Run optimization op (backprop) and cost op (to get loss value)
                    d=sess.run(optimizer, feed_dict={self.X: batch_xs})
                    c=sess.run(cost,feed_dict={self.X:batch_xs})
                # Display logs per epoch step
                if epoch % 1 == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

            print("Optimization Finished!")

    def test(self,sess,test_set,y_pred):
        # Applying encode and decode over test set
        encode_decode = sess.run(y_pred, feed_dict={self.X: test_set})
        return encode_decode

    def show_images(self,mnist,reconstructions):
        # Compare original images with their reconstructions
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(10):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(reconstructions[i], (28, 28)))
        f.show()
        plt.draw()
        plt.waitforbuttonpress()

    def master(self):

        # Import MNIST data
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

        batch_size = 256
        display_step = 1
        examples_to_show = 10
        training_epochs = 2

        # Network Parameters
        n_hidden_1 = 256  # 1st layer num features
        n_hidden_2 = 128  # 2nd layer num features
        n_hidden_3 = 64   # 3rd layer num features
        n_input = 784

        self.X = tf.placeholder("float", [None, n_input])

        W_h1_f=tf.Variable(tf.random_normal([n_input, n_hidden_1]))
        W_h2_f=tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
        W_h3_f=tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3]))

        W_h1_r=tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2]))
        W_h2_r=tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1]))
        W_h3_r = tf.Variable(tf.random_normal([n_hidden_1, n_input]))

        b_h1_f=tf.Variable(tf.random_normal([n_hidden_1]))
        b_h2_f=tf.Variable(tf.random_normal([n_hidden_2]))
        b_h3_f = tf.Variable(tf.random_normal([n_hidden_3]))

        b_h1_r = tf.Variable(tf.random_normal([n_hidden_2]))
        b_h2_r=tf.Variable(tf.random_normal([n_hidden_1]))
        b_h3_r=tf.Variable(tf.random_normal([n_input]))

        # Construct model - down scalaing
        encoder_op = self.encoder(self.X,
                                  W_h1_f,W_h2_f,W_h3_f,
                                  b_h1_f,b_h2_f,b_h3_f)


        # reconstruction - upscaling
        decoder_op = self.decoder(encoder_op,
                                  W_h1_r,W_h2_r,W_h3_r,
                                  b_h1_r,b_h2_r,b_h3_r)

        # reconstruction - upscaling
        decoder_op_new = self.decoder_new(encoder_op,
                                  W_h1_r, W_h2_r, W_h3_r,
                                  b_h1_r, b_h2_r, b_h3_r)

        y_pred = decoder_op
        y_true = self.X

        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(cost)

        sess=tf.Session()

        # Initializing the variables
        init = tf.global_variables_initializer()
        sess.run(init)
        total_batch = int(mnist.train.num_examples / batch_size)
        batch=mnist.train.next_batch(batch_size)
        self.train(sess,total_batch,training_epochs,batch,optimizer,cost)

        # just to show downscaling
        d = sess.run(encoder_op, feed_dict={self.X: batch[0]})
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(10):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(d[i], (8, 8)))
        f.show()
        plt.draw()
        #plt.waitforbuttonpress()

        # after training weights are automatically updated so these could be used for upscaling
        e = sess.run(decoder_op, feed_dict={self.X: d})
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(10):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(e[i], (8, 8)))
        f.show()
        plt.draw()
        plt.waitforbuttonpress()

        test_set=mnist.test.images[:examples_to_show]
        reconstructions=self.test(sess,test_set,y_pred)
        self.show_images(mnist,reconstructions)

a=AutoEncoder()
a.master()
print('run completed')










