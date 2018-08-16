# coding: utf-8
# pretrain part 
# train on 1
# transfer to 2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
#import cPickle
import pickle
import os
import calendar;
import time;
import argparse
import sys

tf.logging.set_verbosity(tf.logging.INFO)

cifar10_path = '/scratch/tz1303/divided_cifar10_data'
BATCH_SIZE = 256
EPOCHS = 100
_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
classcount=5
data_amount=25000
'''
def unpickle(file):

    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict
'''
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def train_data_loader():
    print("Loading training data...")
    datafile = os.path.join(cifar10_path, 'data1.npy')
    labelfile = os.path.join(cifar10_path, 'label1.npy')
    input_data=np.load(datafile)
    input_label=np.load(labelfile)
    
    print(input_data.shape)
    print(input_label.shape)
    print("All data loaded")
    return (input_data, input_label)

def val_data_loader():
    print("Loading testing data...")
    datafile = os.path.join(cifar10_path, 'vdata1.npy')
    labelfile = os.path.join(cifar10_path, 'vlabel1.npy')
    val_data=np.load(datafile)
    val_label=np.load(labelfile)
    print(val_data.shape)
    print(val_label.shape)
    print("All data loaded")
    return (val_data, val_label)

def train_input_fn(train_data):
    train_dataset = tf.contrib.data.Dataset.from_tensor_slices(
        #{"img": train_data[0],
        # "lbl": train_data[1]})
        train_data)
    train_dataset = train_dataset.map(train_preprocessing).shuffle(data_amount)
    train_dataset = train_dataset.batch(BATCH_SIZE).repeat(EPOCHS)
    iterator = train_dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()    
    return features, labels
    #return train_dataset

def test_input_fn(val_data):
    val_dataset = tf.contrib.data.Dataset.from_tensor_slices(val_data)
    val_dataset = val_dataset.map(val_preprocessing)
    val_dataset = val_dataset.batch(BATCH_SIZE)
    iterator = val_dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()    
    return features, labels

def train_preprocessing(img, lbl):

    img = tf.cast(img, tf.float32)
    lbl = tf.cast(lbl, tf.uint8)
    img = tf.image.resize_image_with_crop_or_pad(
    img, _HEIGHT + 8, _WIDTH + 8)
    img = tf.random_crop(img, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

    img = tf.image.random_flip_left_right(img)
    '''
    img = tf.image.random_brightness(img, max_delta=63)
    img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
    '''
    img = tf.image.per_image_standardization(img)    
    
    return img, lbl

def val_preprocessing(img, lbl):
    img = tf.cast(img, tf.float32)
    lbl = tf.cast(lbl, tf.uint8)
    img = tf.image.per_image_standardization(img)    
    
    return img, lbl

def resnet_layer(inputs, filters=16, kernel_size=3, strides=1, activation=tf.nn.relu,
                 batch_normalization=True):
    # Function from keras's example for Cifar10:
    # https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
    x = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size,
                         strides=strides, padding='same')
    if batch_normalization:
        x = tf.layers.batch_normalization(x)
    if activation is not None:
        x = activation(x)
    return x



def resnet_model(features, labels, mode):
    # Function from keras's example for Cifar10:
    # https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
    with tf.variable_scope('feature_extractor'):
        filters = 16
        num_res_blocks = 3
        
        features = tf.cast(features, dtype=tf.float32)
        x = resnet_layer(inputs=features)
        for stack in range(3):
            for res_block in range(num_res_blocks):
                strides = 1
                if stack > 0 and res_block == 0:
                    strides = 2
                y = resnet_layer(inputs=x, filters=filters, strides=strides)
                y = resnet_layer(inputs=y, filters=filters, activation=None)
                if stack > 0 and res_block == 0:
                    x = resnet_layer(inputs=x, filters=filters, kernel_size=1,
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

    classes = tf.argmax(input=x, axis=1)
    correct_prediction = tf.equal(tf.cast(classes, tf.uint8), labels)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = tf.identity(acc, name='accuracy_tensor')
    global_step=tf.train.get_global_step()
    global_step = tf.identity(global_step, name='global_step')

    predictions = {'classes': classes, 'accuracy': acc}
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), classcount)
    onehot_labels = tf.identity(onehot_labels, name='1_labels')
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=x)
    loss = tf.reduce_mean(loss)
    
    #tf.summary.scalar('accuracy', acc)
    #tf.summary.scalar('loss', loss)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        epoch = tf.ceil(global_step*BATCH_SIZE/data_amount)
        epoch = tf.identity(epoch, name='epoch')
        lr = lr_schedule(epoch)
        lr = tf.identity(lr, name='lr')

        optimizer = tf.train.AdamOptimizer(lr)
        
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {'accuracy': tf.metrics.accuracy(labels=labels,
                                                       predictions=classes)}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss,
                                      eval_metric_ops=eval_metric_ops)
    
def lr_schedule(epoch):
    # 100 epoch
    lr = 1e-3
    cond1 = tf.cond(tf.greater(epoch, 40), lambda:lr*1e-1, lambda:lr)
    cond2 = tf.cond(tf.greater(epoch, 60), lambda:lr*1e-2, lambda:cond1)
    cond3 = tf.cond(tf.greater(epoch, 80), lambda:lr*1e-3, lambda:cond2)
    cond4 = tf.cond(tf.greater(epoch, 90), lambda:lr*0.5e-3, lambda:cond3)
    cond4 = tf.identity(cond4, name='cond4')
    return cond4


def inference():
    inp = train_data_loader()
    val = val_data_loader()

    timestamp = calendar.timegm(time.gmtime())

    model_root = '/scratch/tz1303/ckpts_5v5_train_on_1'
    if not FLAGS.save_dir == '---':
        model_dir = os.path.join(model_root, FLAGS.save_dir)
    else:
        model_dir = os.path.join(model_root, str(timestamp))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print(model_dir)

    tensors_to_log = {'batch accuracy': 'accuracy_tensor', 'learning rate': 'lr'}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=100)

    model = tf.estimator.Estimator(model_fn=resnet_model, 
        config=tf.estimator.RunConfig().replace(save_checkpoints_steps=2000, 
                                                log_step_count_steps=1000,
                                                save_summary_steps=100000,),
        model_dir=model_dir)

    validation_hook = tf.contrib.learn.monitors.replace_monitors_with_hooks([
        tf.contrib.learn.monitors.ValidationMonitor(input_fn=lambda:test_input_fn(val))],
        model)[0]

    model.train(input_fn=lambda:train_input_fn(inp),
                #hooks=[logging_hook])
                hooks=[logging_hook, validation_hook])

    print(model.evaluate(input_fn=lambda:test_input_fn(val)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_dir',
        type=str,
        default='08091927-fulldata-pre')
    FLAGS, unparsed = parser.parse_known_args()


    inference()