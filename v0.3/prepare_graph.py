import tensorflow as tf

# conv and max pools are defined here but not used in this version
class Build_Graph(object):
    def __init__(self):
        self.w=[0]

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
        # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
        # strides: A list of ints. 1-D vector of length 4. strides tell that how much filter be moved in each direction 1 mean move a filter to one pixel in each direction


    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
        # value: tensor with shape [batch,height,width,channels]
        # ksize: list of ints with length >=4. size of windows from which max is to be selected. Here batch-size=1, window size=2x2 and channels are 1
        # stride: how much to move in each direction
        # padding: how to take windows just like in convolution


