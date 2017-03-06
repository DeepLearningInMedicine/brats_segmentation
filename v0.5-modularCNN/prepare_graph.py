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


    def max_pool_2x2(self,x,ksz,strds):
        """

        :param x:       input to be pooled
        :param ksz:     list of ints with length >=4. size of windows from which max is to be selected. Here batch-size=1, window size=2x2 and channels are 1
        :param strds:   how much to move in each direction
        :return:        pooled input.
        """
        return tf.nn.max_pool(x, ksize=ksz, strides=strds, padding='SAME')


    def create_input_layer(self,input_size,output_size):
        x = tf.placeholder(tf.float32, [None, input_size])
        y_ = tf.placeholder(tf.float32, [None, output_size])
        return x,y_

    def create_1st_conv_layer(self,input,new_shape,w_shape,b_shape):
        """
        :param input:       input to be convolved
        :param new_shape:   new_shape is required as input image is in flat format
        :param w_shape:     [w,x,y,z]--> w,x are filter size. y in no. of input channels. z is no. of out channels (neurons)
        :param b_shape:     [x] --> no. of neurons in layer
        :return: h_conv1    [w,x,y,z] --> w,z is receptive field size, y is no.of input channels, z is no. of output channels/feature maps
        """
        x_image = tf.reshape(input, new_shape)  # x=625, x_image=25x25
        W_conv1 = self.weight_variable(w_shape)  # [f-width,f-height,input channels,output channels]
        b_conv1 = self.bias_variable(b_shape)
        h_conv1 = self.conv2d(x_image, W_conv1) + b_conv1  # dimension of x_image and W_conv1 must be same
        return h_conv1

    def create_next_conv_layer(self,input,w_shape,b_shape):
        """
        :param input:       input to be convolved
        :param w_shape:     [w,x,y,z]--> w,x are filter size. y in no. of input channels. z is no. of out channels (neurons)
        :param b_shape:     [x] --> no. of neurons in layer
        :return: h_conv1    [w,x,y,z] --> w,z is receptive field size, y is no.of input channels, z is no. of output channels/feature maps
        """
        # x_image = tf.reshape(input, new_shape)  # no need to reshape as previous conv layer output is already in required shape
        W_conv = self.weight_variable(w_shape)  # [f-width,f-height,input channels,output channels]
        b_conv = self.bias_variable(b_shape)
        h_conv = self.conv2d(input, W_conv) + b_conv  # dimension of x_image and W_conv1 must be same
        return h_conv

    def create_fully_connected_layer(self,input,w_shape,b_shape,flat_shape):
        W_fc1 = self.weight_variable(w_shape)
        b_fc1 = self.bias_variable(b_shape)
        h_pool1_flat = tf.reshape(input, flat_shape)
        h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
        return h_fc1

    def drop_out(self,input):
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(input, keep_prob)
        return h_fc1_drop,keep_prob

    def create_output_layer(self,input,w_shape,b_shape,actual_output):
        W_fc2 = self.weight_variable(w_shape)
        b_fc2 = self.bias_variable(b_shape)
        y_conv = tf.matmul(input, W_fc2) + b_fc2
        cross_entropy=self.loss(actual_output,y_conv)
        return y_conv,cross_entropy

    def loss(self,labels,logits):
        pred = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        error=cross_entropy = tf.reduce_mean(pred,name='xentropy_mean')
        return error
