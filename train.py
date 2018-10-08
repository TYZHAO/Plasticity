import tensorflow as tf
import numpy as np
import os
import cv2
from tensorflow.contrib import rnn

batchsize = 16
maxepochs = 100
clips = 20

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

def model(x, num_hidden=512):
    #x = tf.zeros((10,5,32,32,3))
    #x = tf.reshape(x, [-1,32,32,3])
    x = conv_block(x, 'conv1', filters=16, kernel_size=3, strides=1, activation=tf.nn.relu,batch_normalization=True)
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
    sc1 = x
    x = conv_block(x, 'conv2', filters=32, kernel_size=3, strides=1, activation=tf.nn.relu,batch_normalization=True)
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
    sc2 = x
    x = conv_block(x, 'conv3', filters=64, kernel_size=3, strides=1, activation=tf.nn.relu,batch_normalization=True)
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
    sc3 = x
    x = conv_block(x, 'conv4', filters=128, kernel_size=3, strides=1, activation=tf.nn.relu,batch_normalization=True)
    x = tf.layers.max_pooling2d(x, pool_size=2, strides=2)
    sc4 = x
    x = tf.contrib.layers.flatten(x)
    print(x)
    x = tf.reshape(x, [-1, clips, 512])
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

    o = tf.reshape(o, (-1,2,2,128))
    #stacked = o.get_shape().as_list()
    #print(stacked)
    #print(tf.concat([o,sc4],-1))
    kernel = tf.get_variable('k', (3,3,64,128*2))
    shape = tf.shape(o)
    
    o = tf.nn.conv2d_transpose(tf.concat([o,sc4],-1), kernel, tf.stack((shape[0],4,4,64)), strides=[1,2,2,1], padding='SAME')
    kerkernel = tf.get_variable('skr', (3,3,32,64*2))
    o = tf.nn.conv2d_transpose(tf.concat([o,sc3],-1), kerkernel, tf.stack((shape[0],8,8,32)), strides=[1,2,2,1], padding='SAME')
    kerrkernel = tf.get_variable('skrr', (3,3,16,32*2))
    o = tf.nn.conv2d_transpose(tf.concat([o,sc2],-1), kerrkernel, tf.stack((shape[0],16,16,16)), strides=[1,2,2,1], padding='SAME')
    kerrrkernel = tf.get_variable('skrrr', (3,3,3,16*2))
    o = tf.nn.conv2d_transpose(tf.concat([o,sc1],-1), kerrrkernel, tf.stack((shape[0],32,32,3)), strides=[1,2,2,1], padding='SAME')
    return o
def train():
    tf.reset_default_graph()
    file_name = '../trainlist.txt'
    #file_name = 'F:\\data\\ucf101_jpegs_256.zip~\\train_list.txt'
    train_img = []
    for line in open(file_name):
        line = line.split()
        train_img.append(line)

    #train_img = [item[0] for item in train_img]

    train_img = [train_img[x:x+clips] for x in range(0, len(train_img), clips)]

    dataset = tf.data.Dataset.from_tensor_slices(train_img)
    dataset = dataset.batch(batchsize).repeat(maxepochs)
    iterator = dataset.make_one_shot_iterator()
    get_data = iterator.get_next()

    global_step=tf.get_variable('global_step',(), trainable=False, initializer=tf.constant_initializer([1]))
    global_step_update = tf.assign(global_step, global_step+1)
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))

    o = model(x, num_hidden=512)

    loss = tf.losses.mean_squared_error(x[1:], o[:-1])

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss=loss, global_step=global_step)

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        ls_sum = 0
        while True:
            try:
                steps = tf.train.global_step(sess, global_step)
                train_data = sess.run(get_data)
                batch_data = []
                for i in range(len(train_data)):
                    d = [cv2.resize(cv2.imread(str(d[0], encoding = "utf-8")),(32,32)) for d in train_data[i]]
                    d = np.array(d)
                    batch_data.append(d)
                batch_data = np.array(batch_data)
                batch_data = np.reshape(batch_data, [-1,32,32,3])
                #print(batch_data.shape)

                ls,_ = sess.run([loss, train_op], feed_dict={x:batch_data})

                if steps%10 == 0:
                    print("steps: {} loss: {}".format(steps, ls_sum/10))
                    ls_sum = 0
                else:
                    ls_sum += ls
                    
            except tf.errors.OutOfRangeError:
                break

if __name__ == '__main__':
    train()