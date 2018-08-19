# non-plas-resnet

import tensorflow as tf
import numpy as np

def resnet_layer(inputs, scope_name, filters=16, kernel_size=3, strides=1, 
                activation=tf.nn.relu,batch_normalization=True):
    # Function from keras's example for Cifar10:
    # https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
    with tf.variable_scope(scope_name):
        x = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                             strides=strides, padding='same')
        if batch_normalization:
            x = tf.layers.batch_normalization(x)
        if activation is not None:
            x = activation(x)
    return x



def resnet_model(features, scope, num_res_blocks, classcount):
    # Function from keras's example for Cifar10:
    # https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
    with tf.variable_scope(scope):
        filters = 16
        
        features = tf.cast(features, dtype=tf.float32)
        x = resnet_layer(inputs=features, scope_name='first')
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:
                    strides = 2
                y = resnet_layer(inputs=x, filters=filters, scope_name="{}_{}_one".format(
                    stack, res_block), strides=strides)
                y = resnet_layer(inputs=y, filters=filters, scope_name="{}_{}_two".format(
                    stack, res_block), activation=None)
                if stack > 0 and res_block == 0:
                    x = resnet_layer(inputs=x, filters=filters, scope_name="{}_{}_three".format(
                                     stack, res_block), kernel_size=1,
                                     strides=strides, activation=None,
                                     batch_normalization=False)
                x = tf.nn.relu(x + y)
            filters *= 2
        
        x = tf.layers.average_pooling2d(x, pool_size=8, strides=8)
        x = tf.contrib.layers.flatten(x)

    with tf.variable_scope('classifier'):
    #x = tf.layers.dense(inputs=x, units=10)
        w = tf.get_variable('w',(int(x.shape[1]), classcount),initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('b',(classcount,),initializer=tf.contrib.layers.xavier_initializer())
        x = tf.matmul(x, w) + b
        x = tf.identity(x, name="logits")

    return x