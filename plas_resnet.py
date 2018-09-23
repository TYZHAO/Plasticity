# plas-resnet

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

def hebb_resnet_layer(inputs, scope_name, update_ops, filters=16, kernel_size=3, strides=1, 
                activation=tf.nn.relu, batch_normalization=True):
                 #activation=tf.nn.relu, batch_normalization=True):
    # Function from keras's example for Cifar10:
    # https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
    # x = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size,
    #                     strides=strides, padding='same')
    with tf.variable_scope(scope_name):
        
        w = tf.get_variable('conv_w', (kernel_size, kernel_size,
                                       int(inputs.shape[-1]), filters),
                                       initializer=tf.contrib.layers.xavier_initializer())
        alpha = tf.get_variable('conv_alpha', (kernel_size, kernel_size,
                                               int(inputs.shape[-1]), filters),
                                               initializer=tf.random_uniform_initializer(0,1))
        hebb = tf.get_variable('conv_hebb', (kernel_size, kernel_size,
                                             int(inputs.shape[-1]), filters), trainable=False,
                                             initializer=tf.zeros_initializer)
        b = tf.get_variable('conv_b', (filters,), initializer=tf.contrib.layers.xavier_initializer())
        eta = tf.get_variable('eta', (), initializer=tf.constant_initializer(.01))
        hebb_update = tf.get_variable('hebb_update', (kernel_size, kernel_size,
                                                     int(inputs.shape[-1]), filters), trainable=False,
                                                     initializer=tf.zeros_initializer)
        #inp = tf.get_variable('inputs', inputs.shape[1:],trainable = False)
        #update_inputs = tf.assign(inp, inputs[0])
        #update_ops.append(update_inputs)
        #inputs = tf.identity(inputs, 'inputs')

        new_hebb = eta * hebb_update  + (1-eta)*hebb

        x = tf.nn.conv2d(input=inputs, filter=w + tf.multiply(alpha, new_hebb),
                         strides=[1, strides, strides, 1], padding='SAME') + b
        
        with tf.control_dependencies([x]):
            _hebb = tf.assign(hebb, new_hebb)
            update_ops.append(_hebb)

        if batch_normalization:
            x = tf.layers.batch_normalization(x)
        if activation is not None:
            x = activation(x)

        # y is to be the output reshaped so as to be used as a kernel for convolution
        #     on the input to get the Hebbian update        
        upsample = tf.contrib.keras.layers.UpSampling2D(size=(strides, strides), data_format=None)
        y = upsample(x)
        
        y = tf.tanh(y)

        y = tf.transpose(y, [1, 2, 0, 3])

        # in_mod is the input padded a/c to prev. convolution
        in_mod = tf.pad(inputs, [[0, 0], [int(np.floor((kernel_size - 1) / 2)), int(np.ceil((kernel_size - 1) / 2))], 
                    [int(np.floor((kernel_size - 1) / 2)), int(np.ceil((kernel_size - 1) / 2))], [0, 0]])
        in_mod = tf.tanh(in_mod)

        # in_mod is now modded so as to preserve channels and sum over mini-batch
        #     samples for Hebbian update convolution
        in_mod = tf.transpose(in_mod, [3, 1, 2, 0])

        hebb_new = tf.nn.conv2d(input=in_mod, filter=y, strides=([1] * 4), padding='VALID')
        hebb_new = tf.transpose(hebb_new, [1, 2, 0, 3])
        
        shapex = tf.cast(tf.shape(x), tf.float32)
        shapey = tf.cast(tf.shape(y), tf.float32)
        hebb_new = hebb_new/(shapey[0] * shapey[1] * shapey[2]) #- tf.reduce_mean(tf.square(x), axis=[0,1,2,3]) *hebb

        hebb_ = tf.assign(hebb_update, hebb_new)

        update_ops.append(hebb_)
        
    return x


def resnet_model(features, onehot_labels, scope, num_res_blocks, classcount):
    # Function from keras's example for Cifar10:
    # https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
    update_ops = []
    features = tf.identity(features,'feat')
    with tf.variable_scope(scope):    
        filters = 16

        features = tf.cast(features, dtype=tf.float32)
        x = resnet_layer(inputs=features, scope_name='first')

        stack = 0
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            y = resnet_layer(inputs=x, scope_name="{}_{}_one".format(
                stack, res_block), filters=filters, strides=strides)
            y = resnet_layer(inputs=y, scope_name="{}_{}_two".format(
                stack, res_block), filters=filters, activation=None)
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x, scope_name="{}_{}_three".format(
                    stack, res_block), filters=filters, kernel_size=1,
                    strides=strides, activation=None,
                    batch_normalization=False)
            x = tf.nn.relu(x + y)
        filters *= 2

        stack = 1
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            y = resnet_layer(inputs=x, scope_name="{}_{}_one".format(
                stack, res_block), filters=filters, strides=strides)
            y = resnet_layer(inputs=y, scope_name="{}_{}_two".format(
                stack, res_block), filters=filters, activation=None)
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x, scope_name="{}_{}_three".format(
                    stack, res_block), filters=filters, kernel_size=1,
                    strides=strides, activation=None,
                    batch_normalization=False)
            x = tf.nn.relu(x + y)
        filters *= 2

        stack = 2
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2
            y = resnet_layer(inputs=x, scope_name="{}_{}_one".format(
                stack, res_block), filters=filters, strides=strides)#, update_ops=update_ops)
            y = resnet_layer(inputs=y, scope_name="{}_{}_two".format(
                stack, res_block), filters=filters, activation=None)#), update_ops=update_ops)
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x, scope_name="{}_{}_three".format(
                    stack, res_block), filters=filters, kernel_size=1,
                    strides=strides, activation=None,
                    batch_normalization=False)
            x = tf.nn.relu(x + y)
        filters *= 2

        x = tf.layers.average_pooling2d(x, pool_size=8, strides=8)
        x = tf.contrib.layers.flatten(x)
        
    with tf.variable_scope('classifier'):
        w = tf.get_variable('w',(int(x.shape[1]), classcount),initializer=tf.contrib.layers.xavier_initializer())
        alpha = tf.get_variable('alpha',(int(x.shape[1]), classcount),initializer=tf.random_uniform_initializer(0,1))
        hebb = tf.get_variable('hebb', (int(x.shape[1]), classcount), trainable=False,
                                             initializer=tf.zeros_initializer)        
        b = tf.get_variable('b',(classcount,),initializer=tf.contrib.layers.xavier_initializer())
        eta = tf.get_variable('eta', (), initializer=tf.constant_initializer(.01))
        hebb_update = tf.get_variable('hebb_update', (int(x.shape[1]), classcount), trainable=False,                                      
                                             initializer=tf.zeros_initializer) 
        
        new_hebb = eta * hebb_update + hebb

        y = tf.matmul(x, w + tf.multiply(alpha, new_hebb)) + b

        with tf.control_dependencies([y]):
            _hebb = tf.assign(hebb, new_hebb)
            update_ops.append(_hebb)

        shapex = tf.cast(tf.shape(x), tf.float32)
        limitedx = tf.sign(x)*tf.maximum(tf.abs(x)-1,0)
        clamped_output = onehot_labels*tf.nn.softmax(y)
        tf.identity(clamped_output, 'co')
        hebb_new = tf.reduce_mean(tf.matmul(tf.expand_dims(limitedx, axis=-1),tf.expand_dims(clamped_output, axis=1)), axis=0)
        hebb_reduce = tf.matmul(tf.matmul(hebb, tf.transpose(onehot_labels,[1,0])), onehot_labels)/shapex[0]
        
        hebb_ = tf.assign(hebb_update, hebb_new - hebb_reduce)
        
        update_ops.append(hebb_)
        x = y
        
    return x, update_ops

def hebb_standardization(w, hebb):    
    hebb_reduce_mean = hebb - tf.reduce_mean(hebb, axis = [0,1])
    stddev = tf.sqrt(tf.reduce_sum(tf.square(hebb_reduce_mean), axis=[0,1]))
    minstddev = 1/tf.multiply(hebb.shape[0], hebb.shape[1])
    minstddev = tf.cast(minstddev, tf.float32)
    conv_strd = hebb_reduce_mean/tf.maximum(stddev, minstddev)
    return conv_strd



def old_resnet_layer(inputs, scope_name, update_ops, filters=16, kernel_size=3, strides=1,
                 activation=tf.nn.relu, batch_normalization=True):
    # Function from keras's example for Cifar10:
    # https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
    # x = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size,
    #                     strides=strides, padding='same')
    with tf.variable_scope(scope_name):
        
        w = tf.get_variable('conv_w', (kernel_size, kernel_size,
                                       int(inputs.shape[-1]), filters),
                                       initializer=tf.contrib.layers.xavier_initializer())
        alpha = tf.get_variable('conv_alpha', (kernel_size, kernel_size,
                                               int(inputs.shape[-1]), filters),
                                               initializer=tf.contrib.layers.xavier_initializer())
        hebb = tf.get_variable('conv_hebb', (kernel_size, kernel_size,
                                             int(inputs.shape[-1]), filters), trainable=False,
                                             initializer=tf.zeros_initializer)
        b = tf.get_variable('conv_b', (filters,), initializer=tf.contrib.layers.xavier_initializer())
        eta = tf.get_variable('eta', (), initializer=tf.constant_initializer(.01))
        hebb_update = tf.get_variable('hebb_update', (kernel_size, kernel_size,
                                                     int(inputs.shape[-1]), filters), trainable=False,
                                                     initializer=tf.zeros_initializer)
        #inp = tf.get_variable('inputs', inputs.shape[1:],trainable = False)
        #update_inputs = tf.assign(inp, inputs[0])
        #update_ops.append(update_inputs)
        #inputs = tf.identity(inputs, 'inputs')

        new_hebb = eta * hebb_update  + (1 - eta) * hebb        

        x = tf.nn.conv2d(input=inputs, filter=w + tf.multiply(alpha, new_hebb),
                         strides=[1, strides, strides, 1], padding='SAME') + b
        
        with tf.control_dependencies([x]):
            _hebb = tf.assign(hebb, new_hebb)
            update_ops.append(_hebb)

        # y is to be the output reshaped so as to be used as a kernel for convolution
        #     on the input to get the Hebbian update        
        upsample = tf.contrib.keras.layers.UpSampling2D(size=(strides, strides), data_format=None)
        y = upsample(x)
            
        y = tf.transpose(y, [1, 2, 0, 3])

        # in_mod is the input padded a/c to prev. convolution
        in_mod = tf.pad(inputs, [[0, 0], [int(np.floor((kernel_size - 1) / 2)), int(np.ceil((kernel_size - 1) / 2))], 
                    [int(np.floor((kernel_size - 1) / 2)), int(np.ceil((kernel_size - 1) / 2))], [0, 0]])

        # in_mod is now modded so as to preserve channels and sum over mini-batch
        #     samples for Hebbian update convolution
        in_mod = tf.transpose(in_mod, [3, 1, 2, 0])

        hebb_new = tf.nn.conv2d(input=in_mod, filter=y, strides=([1] * 4),
                                   padding='VALID')
        hebb_new = tf.transpose(hebb_new, [1, 2, 0, 3])
        
        # standardization/average        
        if kernel_size > 1:
            hebb_new = hebb_new - tf.reduce_mean(hebb_new, axis = [0,1])
            #hebb_new = hebb_standardization(w, hebb_new)
        shape = tf.shape(y)
        hebb_ = tf.assign(hebb_update, hebb_new/tf.cast(shape[0] * shape[1] * shape[2], tf.float32))

        update_ops.append(hebb_)        

        if batch_normalization:
            x = tf.layers.batch_normalization(x)
        if activation is not None:
            x = activation(x)
        
    return x

def old_resnet_model(features, scope, num_res_blocks, classcount):
    # Function from keras's example for Cifar10:
    # https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
    update_ops = []
    features = tf.identity(features,'feat')
    with tf.variable_scope(scope):    
        filters = 16

        features = tf.cast(features, dtype=tf.float32)
        x = resnet_layer(inputs=features, scope_name='first', update_ops=update_ops)
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:
                    strides = 2
                y = resnet_layer(inputs=x, scope_name="{}_{}_one".format(
                    stack, res_block), filters=filters, strides=strides, update_ops=update_ops)
                y = resnet_layer(inputs=y, scope_name="{}_{}_two".format(
                    stack, res_block), filters=filters, activation=None, update_ops=update_ops)
                if stack > 0 and res_block == 0:
                    x = resnet_layer(inputs=x, scope_name="{}_{}_three".format(
                        stack, res_block), filters=filters, kernel_size=1,
                        strides=strides, activation=None,
                        batch_normalization=False, update_ops=update_ops)
                x = tf.nn.relu(x + y)
            filters *= 2
        
        x = tf.layers.average_pooling2d(x, pool_size=8, strides=8)
        x = tf.contrib.layers.flatten(x)
        
    with tf.variable_scope('classifier'):
        w = tf.get_variable('w',(int(x.shape[1]), classcount),initializer=tf.contrib.layers.xavier_initializer())
        alpha = tf.get_variable('alpha',(int(x.shape[1]), classcount),initializer=tf.contrib.layers.xavier_initializer())
        hebb = tf.get_variable('hebb', (int(x.shape[1]), classcount), trainable=False,
                                             initializer=tf.zeros_initializer)        
        b = tf.get_variable('b',(classcount,),initializer=tf.contrib.layers.xavier_initializer())
        eta = tf.get_variable('eta', (), initializer=tf.constant_initializer(.01))
        hebb_update = tf.get_variable('hebb_update', (int(x.shape[1]), classcount), trainable=False,                                      
                                             initializer=tf.zeros_initializer)
        
        new_hebb = eta * hebb_update  + (1 - eta) * hebb

        y = tf.matmul(x, w + tf.multiply(alpha, new_hebb)) + b

        with tf.control_dependencies([y]):        
            _hebb = tf.assign(hebb, new_hebb)
            update_ops.append(_hebb)
            
        hebb_ = tf.assign(hebb_update, tf.reduce_mean(
            tf.matmul(tf.expand_dims(x, axis=-1),tf.expand_dims(y, axis=1)), axis=0))
        
        update_ops.append(hebb_)
        x = y
        
    return x, update_ops
