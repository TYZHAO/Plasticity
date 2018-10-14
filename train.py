import tensorflow as tf
import numpy as np
import calendar
import time
import os
import cv2
import argparse 
import sys 

from tensorflow.contrib import rnn

tf.logging.set_verbosity(tf.logging.INFO)

file_name = '/beegfs/rw1691/inputs.tfrecord'

batchsize = 128 
maxepochs = 100
clips = 10
num_gpus = 2

def conv_block(inputs, scope_name, filters=16, kernel_size=3, strides=1, 
                activation=tf.nn.relu,batch_normalization=True):
    with tf.variable_scope(scope_name):
        x = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                             strides=strides, padding='same')
        if batch_normalization:
            x = tf.layers.batch_normalization(x)
        if activation is not None:
            x = activation(x)
    return x

def lr_schedule(epoch):
    lr = 1e-3
    cond1 = tf.cond(tf.greater(epoch, 40), lambda:lr*1e-1, lambda:lr)
    cond2 = tf.cond(tf.greater(epoch, 60), lambda:lr*1e-2, lambda:cond1)
    cond3 = tf.cond(tf.greater(epoch, 80), lambda:lr*1e-3, lambda:cond2)
    #cond4 = tf.cond(tf.greater(epoch, 90), lambda:lr*0.5e-3, lambda:cond3)
    #cond4 = tf.identity(cond4, name='cond4')
    return cond3

def model(x, num_hidden=256):
    #x = tf.zeros((10,5,32,32,3))
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        x = tf.reshape(x, [-1,64,64,3])
        x = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), x)
        x = conv_block(x, 'conv1', filters=16, kernel_size=3, strides=1, activation=tf.nn.relu,batch_normalization=True)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
        sc1 = x # 32
        x = conv_block(x, 'conv2', filters=32, kernel_size=3, strides=1, activation=tf.nn.relu,batch_normalization=True)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
        sc2 = x # 16
        x = conv_block(x, 'conv3', filters=64, kernel_size=3, strides=1, activation=tf.nn.relu,batch_normalization=True)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
        sc3 = x # 8
        x = conv_block(x, 'conv4', filters=128, kernel_size=3, strides=1, activation=tf.nn.relu,batch_normalization=True)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
        sc4 = x # 4
        x = conv_block(x, 'conv5', filters=256, kernel_size=3, strides=1, activation=tf.nn.relu,batch_normalization=True)
        x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
        sc5 = x # 2
        x = tf.layers.average_pooling2d(x, pool_size=2, strides=2)
        x = tf.contrib.layers.flatten(x)
        #print(x)
        x = tf.reshape(x, [-1, clips, 256])
        inputs = []
        for i in range(clips): 
            c=x[:,i,:]
            inputs.append(c)
        #print(inputs)
        lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
        o, s = rnn.static_rnn(lstm_cell, inputs, dtype=tf.float32)

        o = tf.stack(o,1)
        o = tf.expand_dims(o,-2)
        o = tf.expand_dims(o,-2)

        o = tf.reshape(o, (-1,1,1,num_hidden))
        #stacked = o.get_shape().as_list()
        #print(stacked)
        #print(tf.concat([o,sc4],-1))
        upsample = tf.keras.layers.UpSampling2D(size=(2,2))
        o = upsample(o)
        shape = tf.shape(o)    
        
        kernel = tf.get_variable('k', (3,3,128,256*2))
        o = tf.nn.conv2d_transpose(tf.concat([o,sc5],-1), kernel, tf.stack((shape[0],4,4,128)), strides=[1,2,2,1], padding='SAME')
        kerkernel = tf.get_variable('skr', (3,3,64,128*2))
        o = tf.nn.conv2d_transpose(tf.concat([o,sc4],-1), kerkernel, tf.stack((shape[0],8,8,64)), strides=[1,2,2,1], padding='SAME')
        kerrkernel = tf.get_variable('skrr', (3,3,32,64*2))
        o = tf.nn.conv2d_transpose(tf.concat([o,sc3],-1), kerrkernel, tf.stack((shape[0],16,16,32)), strides=[1,2,2,1], padding='SAME')
        kerrrkernel = tf.get_variable('skrrr', (3,3,16,32*2))
        o = tf.nn.conv2d_transpose(tf.concat([o,sc2],-1), kerrrkernel, tf.stack((shape[0],32,32,16)), strides=[1,2,2,1], padding='SAME')
        kerrrrkernel = tf.get_variable('skrrrr', (3,3,3,16*2))
        o = tf.nn.conv2d_transpose(tf.concat([o,sc1],-1), kerrrrkernel, tf.stack((shape[0],64,64,3)), strides=[1,2,2,1], padding='SAME')

        o = tf.nn.sigmoid(o)*255
    return o

def _parse_function(serialized_example):
    sequence_features = {
        'inputs': tf.FixedLenSequenceFeature(shape=[],
                                       dtype=tf.string)}

    _, sequence = tf.parse_single_sequence_example(
    serialized_example, sequence_features=sequence_features)
    return sequence['inputs']

def train():
    tf.reset_default_graph()
    
    dataset = tf.data.TFRecordDataset(file_name)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(maxepochs)
    dataset = dataset.batch(batchsize)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
  
    x = tf.placeholder(tf.float32, shape=(None,None,64,64,3))    

    split_x = tf.split(x, num_gpus)

    for i in range(num_gpus):
        with tf.device('/GPU:%d' % i):

            o = model(split_x[i], num_hidden=256)

            o = tf.reshape(o, tf.shape(split_x[i]))

            if i == 0:
                loss = tf.losses.mean_squared_error(split_x[i][:,1:], o[:,:-1])
            else:
                loss += tf.losses.mean_squared_error(split_x[i][:,1:], o[:,:-1])
    loss = loss/num_gpus

    global_step=tf.get_variable('global_step',(), trainable=False, initializer=tf.constant_initializer([1]))    
    epoch = tf.ceil(global_step*batchsize//20000+1)            
    lr = lr_schedule(epoch) 
    optimizer = tf.train.AdamOptimizer(lr)
          
    train_op = optimizer.minimize(loss=loss, global_step=global_step)
        
    init_op = tf.global_variables_initializer()



    config = tf.ConfigProto(allow_soft_placement = True)
    
    with tf.Session(config = config) as sess:
        sess.run(init_op)

        model_dir = os.path.join('/beegfs/rw1691/models', FLAGS.save_dir)
        model_name = model_dir+'/good'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        else:
            restorer = tf.train.Saver()
            pretrained_model = tf.train.latest_checkpoint(model_dir)
            restorer.restore(sess, pretrained_model) 

        saver = tf.train.Saver(max_to_keep=5)

        tf.logging.info('start training')
        ls_sum = 0
        while True:
            try:
                steps = tf.train.global_step(sess, global_step)
                input_data = sess.run(next_element)
                imgs = np.zeros(np.concatenate((input_data.shape, (64,64,3)), axis=0))
                for i in range(input_data.shape[0]):
                    for j in range(input_data.shape[1]):
                        imgs[i][j] = cv2.imdecode(np.fromstring(input_data[i][j], dtype=np.uint8), -1)
                batch_data = imgs                
                ls,_,e = sess.run([loss, train_op, epoch], feed_dict={x:batch_data})
                if steps%10 == 0:
                    tf.logging.info("epoch: {} steps: {} loss: {}".format(e, steps, ls_sum/10))
                    ls_sum = 0
                else:
                    ls_sum += ls
                if steps%500 == 0:
                    tf.logging.info("saving model")
                    saver.save(sess, model_name, global_step = steps)
            except tf.errors.OutOfRangeError:
                break

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--save_dir',
        type=str,
        default='1014')

    FLAGS, unparsed = parser.parse_known_args()

    train()

