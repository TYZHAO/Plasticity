from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os

#cifar10_path = '/scratch/tz1303/divided_cifar10_data'

def unpickle(file):

    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


def fetch_data(fetch, datainput, classcount):
    data = datainput[0]
    label = datainput[1]
    
    index = np.arange(int(label.shape[0]))
    #np.random.shuffle(index)
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
#-------------------------------------------------------------------------------------#
def train_data_loader(dataset, datapath):
    print("Loading training data...")
    if dataset == 2:
        datafile = os.path.join(datapath, 'data2.npy')
        labelfile = os.path.join(datapath, 'label2.npy')
        input_data=np.load(datafile)
        input_label=np.load(labelfile)
        input_label -= 5
    elif dataset == 1:
        datafile = os.path.join(datapath, 'data1.npy')
        labelfile = os.path.join(datapath, 'label1.npy')      
        input_data=np.load(datafile)
        input_label=np.load(labelfile)
    else:
        raise ValueError('unknown dataset')
    
    print(input_data.shape)
    print(input_label.shape)
    print("All data loaded")
    return (input_data, input_label)

def val_data_loader(dataset, datapath):
    print("Loading testing data...")
    if dataset == 2:
        datafile = os.path.join(datapath, 'vdata2.npy')
        labelfile = os.path.join(datapath, 'vlabel2.npy')
        val_data=np.load(datafile)
        val_label=np.load(labelfile)
        val_label -= 5
    elif dataset == 1:
        datafile = os.path.join(datapath, 'vdata1.npy')
        labelfile = os.path.join(datapath, 'vlabel1.npy')      
        val_data=np.load(datafile)
        val_label=np.load(labelfile)
    else:
        raise ValueError('unknown dataset')

    print(val_data.shape)
    print(val_label.shape)
    print("All data loaded")
    return (val_data, val_label)

def train_input_fn(classcount, use_train, dataset, datapath, batchsize, maxepochs):
    train_data = fetch_data(use_train, train_data_loader(dataset, datapath), classcount)
    train_dataset = tf.contrib.data.Dataset.from_tensor_slices(train_data)
    train_dataset = train_dataset.map(train_preprocessing).shuffle(use_train)
    train_dataset = train_dataset.batch(batchsize).repeat(maxepochs)  
    return train_dataset

def val_input_fn(classcount, use_val, dataset, datapath, batchsize, maxepochs):
    val_data = fetch_data(use_val, val_data_loader(dataset, datapath), classcount)
    val_dataset = tf.contrib.data.Dataset.from_tensor_slices(val_data)
    val_dataset = val_dataset.map(val_preprocessing)
    val_dataset = val_dataset.batch(batchsize)

    return val_dataset
#-------------------------------------------------------------------------------------%
def train_preprocessing(img, lbl):
    img = tf.cast(img, tf.float32)
    lbl = tf.cast(lbl, tf.uint8)
    
    img = tf.image.resize_image_with_crop_or_pad(
    img, 32 + 8, 32 + 8)
    img = tf.random_crop(img, [32, 32, 3])
    
    img = tf.image.random_flip_left_right(img)

    img = tf.image.per_image_standardization(img)    
    
    return img, lbl

def val_preprocessing(img, lbl):
    img = tf.cast(img, tf.float32)
    lbl = tf.cast(lbl, tf.uint8)

    img = tf.image.per_image_standardization(img)    
    
    return img, lbl
