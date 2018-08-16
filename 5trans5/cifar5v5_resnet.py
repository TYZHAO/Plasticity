# coding: utf-8

# In[1]:


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
#import cPickle
import pickle
import os
import calendar;
import time;
import argparse
import sys

tf.logging.set_verbosity(tf.logging.INFO)

cifar10_path = '/scratch/tz1303/divided_cifar10_data'
num_res_blocks = 3
BATCH_SIZE = 256
EPOCHS = 100
_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
classcount = 5
data_amount = 25000
use_train = 25000
use_val = 5000
update_ops = []

'''
def unpickle(file):

    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict
'''
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
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
    datafile = os.path.join(cifar10_path, 'data1.npy')
    labelfile = os.path.join(cifar10_path, 'label1.npy')
    input_data=np.load(datafile)
    input_label=np.load(labelfile)
    #input_label -= 5
    
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
    #val_label -= 5

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


# In[2]:


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



def resnet_model(features):
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

    return x


# In[3]:


def hebb_standardization(w, hebb):
    
    hebb_reduce_mean = hebb - tf.reduce_mean(hebb, axis = [0,1])
    stddev = tf.sqrt(tf.reduce_sum(tf.square(hebb_reduce_mean), axis=[0,1]))
    minstddev = 1/tf.multiply(hebb.shape[0], hebb.shape[1])
    minstddev = tf.cast(minstddev, tf.float32)
    conv_strd = hebb_reduce_mean/tf.maximum(stddev, minstddev)
    return conv_strd


# In[4]:


def lr_schedule(epoch):
    lr = 1e-3
    cond1 = tf.cond(tf.greater(epoch, 40), lambda:lr*1e-1, lambda:lr)
    cond2 = tf.cond(tf.greater(epoch, 60), lambda:lr*1e-2, lambda:cond1)
    cond3 = tf.cond(tf.greater(epoch, 80), lambda:lr*1e-3, lambda:cond2)
    cond4 = tf.cond(tf.greater(epoch, 90), lambda:lr*0.5e-3, lambda:cond3)
    cond4 = tf.identity(cond4, name='cond4')
    return cond4


# In[5]:


def inference():

    train_dataset = train_input_fn()
    train_iterator = train_dataset.make_one_shot_iterator()

    val_dataset = val_input_fn()
    val_iterator = val_dataset.make_initializable_iterator()

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.contrib.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    features, labels = iterator.get_next()
    # next_element = iterator.get_next()
    timestamp = calendar.timegm(time.gmtime())

    model_root = '/scratch/tz1303/ckpts_5v5_train_on_1'
    if not FLAGS.save_dir == '---':
        model_dir = os.path.join(model_root, FLAGS.save_dir)
    else:
        model_dir = os.path.join(model_root, str(timestamp))        
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print(model_dir)
    model_name = model_dir+'/pre_res'+str(6*num_res_blocks+2)
    
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
    
    predictions = {'classes': classes, 'accuracy': acc}
    
    init_op = tf.global_variables_initializer()
        
    with tf.Session() as sess:
        sess.run(init_op)
        saver = tf.train.Saver(max_to_keep=5)
        train_handle = sess.run(train_iterator.string_handle())
        val_handle = sess.run(val_iterator.string_handle())

        while True:
            try:
                steps = tf.train.global_step(sess, global_step)
                ops = [train_op, loss, epoch, acc, lr, update_ops]
                # sess.run(next_element ,feed_dict = {handle: train_iterator})
                _, lossvalue, epoch_num, accuracy, learning_rate, _ = sess.run(ops, feed_dict = {handle: train_handle})
                
                if steps%10 == 0:
                    print('epoch:'+str(epoch_num)+' step:'+str(steps)+' learning rate:'+str(learning_rate))
                    print('loss:'+str(lossvalue)+' batch accuracy:'+str(accuracy))
                if steps % 500 == 0:
                    saver.save(sess, model_name, global_step = steps)                    
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

                    print('eval accuracy:'+str(correct_sum/count))
                        #print(tf.get_default_graph().get_tensor_by_name('feature_extractor/first/xbb:0').eval())
                        #print(tf.get_default_graph().get_tensor_by_name('feature_extractor/0_1_two/conv_w:0').eval())
                        #print(tf.get_default_graph().get_tensor_by_name('feature_extractor/0_1_two/conv_hebb:0').eval())
                        #print(tf.get_default_graph().get_tensor_by_name('classifier/hebb:0').eval())
            except tf.errors.OutOfRangeError:
                break
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
        print('final accuracy:'+str(correct_sum/count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_dir',
        type=str,
        default='---')
    FLAGS, unparsed = parser.parse_known_args()

    inference()