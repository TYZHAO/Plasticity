# coding: utf-8

# plasticity
# pretrain part 
# train on 1
# transfer to 2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import pickle
import os
import calendar;
import time;
import argparse
import sys
import util

tf.logging.set_verbosity(tf.logging.INFO)

cifar10_path = '/scratch/tz1303/divided_cifar10_data'
model_root = '/scratch/tz1303/ckpts_5v5_plas_transfer_on_2'
pretrained_model = '/scratch/tz1303/ckpts_5v5_plas_train_on_1/1534256230/pre_plas_res20-9500'  

num_res_blocks = 3

BATCH_SIZE = 256
EPOCHS = 100

_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3

classcount = 5
data_amount = 25000
use_train = 10000
use_val = 5000

update_ops = []

def unpickle(file):

    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def fetch_data(fetch, datainput):
    data = datainput[0]
    label = datainput[1]
    
    index = np.arange(int(label.shape[0]))
    np.random.shuffle(index)
    data = data[index]
    label = label[index]

    sample_per_class = np.zeros(classcount)
    fetch_per_class = np.floor(fetch/classcount)

    d_buf = np.zeros((fetch, 32, 32, 3))
    l_buf = np.zeros((fetch,))

    p = 0
    for i in range(label.shape[0]):
        if sample_per_class[int(label[i])] < fetch_per_class:
            sample_per_class[int(label[i])] += 1
            d_buf[p] = data[i]
            l_buf[p] = label[i]
            p += 1
        else:
            continue
    print(d_buf.shape)
    print(l_buf.shape)

    return (d_buf, l_buf)

def train_data_loader():
    print("Loading training data...")
    datafile = os.path.join(cifar10_path, 'data2.npy')
    labelfile = os.path.join(cifar10_path, 'label2.npy')
    input_data=np.load(datafile)
    input_label=np.load(labelfile)
    input_label -= 5
    
    print(input_data.shape)
    print(input_label.shape)
    print("All data loaded")
    return (input_data, input_label)

def val_data_loader():
    print("Loading testing data...")
    datafile = os.path.join(cifar10_path, 'vdata2.npy')
    labelfile = os.path.join(cifar10_path, 'vlabel2.npy')
    val_data=np.load(datafile)
    val_label=np.load(labelfile)
    val_label -= 5

    print(val_data.shape)
    print(val_label.shape)
    print("All data loaded")
    return (val_data, val_label)

def train_input_fn():
    train_data = fetch_data(use_train, train_data_loader())
    train_dataset = tf.contrib.data.Dataset.from_tensor_slices(train_data)
    train_dataset = train_dataset.map(train_preprocessing).shuffle(data_amount)
    train_dataset = train_dataset.batch(BATCH_SIZE).repeat(EPOCHS)  
    return train_dataset

def val_input_fn():
    val_data = fetch_data(use_val, val_data_loader())
    val_dataset = tf.contrib.data.Dataset.from_tensor_slices(val_data)
    val_dataset = val_dataset.map(val_preprocessing)
    val_dataset = val_dataset.batch(BATCH_SIZE)

    return val_dataset

def train_preprocessing(img, lbl):
    img = tf.cast(img, tf.float32)
    lbl = tf.cast(lbl, tf.uint8)

    img = tf.image.resize_image_with_crop_or_pad(
    img, _HEIGHT + 8, _WIDTH + 8)
    img = tf.random_crop(img, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

    img = tf.image.random_flip_left_right(img)

    img = tf.image.per_image_standardization(img)    
    
    return img, lbl

def val_preprocessing(img, lbl):
    img = tf.cast(img, tf.float32)
    lbl = tf.cast(lbl, tf.uint8)

    img = tf.image.per_image_standardization(img)    
    
    return img, lbl

def make_model_dir():
    if not FLAGS.save_dir == '---':
        model_dir = os.path.join(model_root, FLAGS.save_dir)
    else:
        timestamp = calendar.timegm(time.gmtime())
        model_dir = os.path.join(model_root, str(timestamp))        
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    tf.logging.info(model_dir)
    return model_dir

def hebb_standardization(w, hebb):    
    hebb_reduce_mean = hebb - tf.reduce_mean(hebb, axis = [0,1])
    stddev = tf.sqrt(tf.reduce_sum(tf.square(hebb_reduce_mean), axis=[0,1]))
    minstddev = 1/tf.multiply(hebb.shape[0], hebb.shape[1])
    minstddev = tf.cast(minstddev, tf.float32)
    conv_strd = hebb_reduce_mean/tf.maximum(stddev, minstddev)
    return conv_strd

def trainable_var():
    var_names = []    
    for var in tf.trainable_variables():    
        var_names.append(var.name)
        print(var.name)

def lr_schedule(epoch):
    lr = 1e-3
    cond1 = tf.cond(tf.greater(epoch, 40), lambda:lr*1e-1, lambda:lr)
    cond2 = tf.cond(tf.greater(epoch, 60), lambda:lr*1e-2, lambda:cond1)
    cond3 = tf.cond(tf.greater(epoch, 80), lambda:lr*1e-3, lambda:cond2)
    cond4 = tf.cond(tf.greater(epoch, 90), lambda:lr*0.5e-3, lambda:cond3)
    cond4 = tf.identity(cond4, name='cond4')
    return cond4

def resnet_layer(inputs, scope_name, filters=16, kernel_size=3, strides=1,
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
        
        inputs = tf.identity(inputs, 'inputs')

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
        #if kernel_size > 1:
            #hebb_new = hebb_new - tf.reduce_mean(hebb_new, axis = [0,1])
            #hebb_new = hebb_standardization(w, hebb_new)
        shape = tf.shape(y)
        hebb_ = tf.assign(hebb_update, hebb_new/tf.cast(shape[0] * shape[1] * shape[2], tf.float32))

        update_ops.append(hebb_)        

        if batch_normalization:
            x = tf.layers.batch_normalization(x)
        if activation is not None:
            x = activation(x)
        
    return x

def resnet_model(features):
    # Function from keras's example for Cifar10:
    # https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
    features = tf.identity(features,'feat')
    with tf.variable_scope('feature_extractor'):    
        filters = 16

        features = tf.cast(features, dtype=tf.float32)
        x = resnet_layer(inputs=features, scope_name='first')
        for stack in range(3):
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
        
    return x

def inference():
    tf.reset_default_graph()

    train_dataset = train_input_fn()
    train_iterator = train_dataset.make_one_shot_iterator()

    val_dataset = val_input_fn()
    val_iterator = val_dataset.make_initializable_iterator()

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.contrib.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    features, labels = iterator.get_next()

    model_dir = make_model_dir()
    model_name = model_dir+'/pre_plas_res'+str(6*num_res_blocks+2)
    
    global_step=tf.get_variable('global_step',(), trainable=False, initializer=tf.zeros_initializer)
    
    epoch = tf.ceil(global_step*BATCH_SIZE/data_amount)
    epoch = tf.identity(epoch, name='epoch')
    
    logits = resnet_model(features)
    logits = tf.identity(logits, 'logits')
    onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), classcount)
    
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits)
    loss = tf.reduce_mean(loss)

    lr = lr_schedule(epoch)
    lr = tf.identity(lr, name='lr')

    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss=loss, global_step=global_step)
    
    classes = tf.argmax(input=logits, axis=1)
    correct_prediction = tf.equal(tf.cast(classes, tf.uint8), labels)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = tf.identity(acc, name='accuracy_tensor')

    init_op = tf.global_variables_initializer()
        
    with tf.Session() as sess:
        sess.run(init_op)
        util.init_from_checkpoint(pretrained_model, {'feature_extractor/':'transfer/'})        
        
        saver = tf.train.Saver(max_to_keep=5)
        train_handle = sess.run(train_iterator.string_handle())
        val_handle = sess.run(val_iterator.string_handle())
        
        while True:
            try:
                steps = tf.train.global_step(sess, global_step)
                
                ops = [train_op, loss, epoch, acc, lr, update_ops]
                _, lossvalue, epoch_num, accuracy, learning_rate, _ = sess.run(ops, feed_dict = {handle: train_handle})
                
                if steps % 10 == 0:
                    tf.logging.info('epoch:'+str(epoch_num)+' step:'+str(steps)+' learning rate:'+str(learning_rate))
                    tf.logging.info('loss:'+str(lossvalue)+' batch accuracy:'+str(accuracy))
                if steps % 500 == 0:
                    saver.save(sess, model_name, global_step = steps)                    
                    evalutation(sess, val_iterator, correct_prediction, handle, val_handle)
                    
            except tf.errors.OutOfRangeError:
                break
        evalutation(sess, val_iterator, correct_prediction, handle, val_handle)
        
def evalutation(sess, val_iterator, correct_prediction, handle, val_handle):
    correct_sum = 0
    count = 0

    sess.run(val_iterator.initializer)
    while True:
        try:                            
            correct_prediction_value = sess.run(correct_prediction, feed_dict = {handle: val_handle})
            count += correct_prediction_value.shape[0]
            correct_sum += np.sum(correct_prediction_value)
        except tf.errors.OutOfRangeError:
            break

    tf.logging.info('eval accuracy:'+str(correct_sum/count))
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_dir',
        type=str,
        default='---')
    FLAGS, unparsed = parser.parse_known_args()

    inference()