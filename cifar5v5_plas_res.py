# coding: utf-8

# plasticity

from __future__ import absolute_import 
from __future__ import division 
from __future__ import print_function

import numpy as np 
import tensorflow as tf 
import pickle 
import os 
import calendar
import time
import argparse 
import sys 
import plas_resnet
import data

tf.logging.set_verbosity(tf.logging.INFO)

cifar10_path = '/scratch/tz1303/divided_cifar10_data'
model_root = '/scratch/tz1303/ckpts_5v5_plas'

#pretrained_model = '/scratch/tz1303/ckpts_5v5_plas/1534905992/pre_plas_res20-7500' 
#pretrained_model = '/scratch/tz1303/ckpts_5v5_plas/1534950884/trans_plas_res20-1800'
#pretrained_model = '/scratch/tz1303/ckpts_5v5_plas/1534971640/pre_plas_res20-7500'
#pretrained_model = '/scratch/tz1303/ckpts_5v5_plas/1534982030/pre_plas_res20-8000'

# limited(conv+fc) plas resnet
pretrained_model = ''

num_res_blocks = 3

batchsize = 256 
maxepochs = 100

_HEIGHT = 32 
_WIDTH = 32 
_NUM_CHANNELS = 3

classcount = 5 

use_val = 5000

def make_model_dir():
    if not FLAGS.save_dir == 'default':
        model_dir = os.path.join(model_root, FLAGS.save_dir)
    else:
        timestamp = calendar.timegm(time.gmtime())
        model_dir = os.path.join(model_root, str(timestamp))
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    tf.logging.info(model_dir)
    return model_dir

def train_layers(min_stack=0, first_min_block=0):
    train_scope = []
    
    min_block = first_min_block
    for stack in range(min_stack, 3):
        for res_block in range(min_block, num_res_blocks):
            train_scope.append("feature_extractor/{}_{}_one/".format(stack, res_block))

            train_scope.append("feature_extractor/{}_{}_two/".format(stack, res_block))
            if stack>0 and res_block == 0:
                train_scope.append("feature_extractor/{}_{}_three/".format(stack, res_block))
        min_block = 0      

    var = []
    if min_stack == 0:
        var.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'feature_extractor/first'))
    for scope in train_scope:
        var.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope))
    var.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'classifier'))
    train_var = [item for sublist in var for item in sublist]
    for item in train_var:
        tf.logging.info(item)

    return train_var

def load_layers(min_stack=0, first_min_block=0):
    train_scope = []
    
    min_block = first_min_block
    for stack in range(0, min_stack):
        for res_block in range(min_block, num_res_blocks):
            train_scope.append("feature_extractor/{}_{}_one/".format(stack, res_block))

            train_scope.append("feature_extractor/{}_{}_two/".format(stack, res_block))
            if stack>0 and res_block == 0:
                train_scope.append("feature_extractor/{}_{}_three/".format(stack, res_block))
        min_block = 0      

    var = []
    if min_stack > 0:
        var.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'feature_extractor/first'))
    for scope in train_scope:
        var.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope))
    train_var = [item for sublist in var for item in sublist]
    for item in train_var:
        tf.logging.info(item)

    return train_var

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


def inference(transfer):
    tf.reset_default_graph()

    #-------------------------------------------------------------------------------------#
    train_dataset = data.train_input_fn(classcount=classcount, use_train=FLAGS.use_train, datapath=cifar10_path,
                                        dataset=FLAGS.dataset, batchsize=batchsize, maxepochs=maxepochs)
    train_iterator = train_dataset.make_one_shot_iterator()

    val_dataset = data.val_input_fn(classcount=classcount, use_val=use_val, datapath=cifar10_path,
                                    dataset=FLAGS.dataset, batchsize=batchsize, maxepochs=maxepochs)
    val_iterator = val_dataset.make_initializable_iterator()
    #-------------------------------------------------------------------------------------%

    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.contrib.data.Iterator.from_string_handle(
        handle, train_dataset.output_types, train_dataset.output_shapes)
    features, labels = iterator.get_next()
    
    global_step=tf.get_variable('global_step',(), trainable=False, initializer=tf.zeros_initializer)
    
    epoch = tf.ceil(global_step*batchsize/FLAGS.use_train)
    epoch = tf.identity(epoch, name='epoch')
    #-------------------------------------------------------------------------------------#
    logits, update_ops = plas_resnet.resnet_model(features, 'feature_extractor', num_res_blocks, classcount)
    logits = tf.identity(logits, 'logits')
    onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), classcount)
    #-------------------------------------------------------------------------------------%
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=logits)
    loss = tf.reduce_mean(loss)

    lr = lr_schedule(epoch)
    lr = tf.identity(lr, name='lr')
    optimizer = tf.train.AdamOptimizer(lr)
    #--------------------------------------------------------------------------------------#
    model_dir = make_model_dir()
    
    # transfer
    if transfer:
        train_var = train_layers(FLAGS.from_stack, 0)
        train_op = optimizer.minimize(loss=loss, global_step=global_step, var_list=train_var)
        model_name = model_dir+'/trans_plas_res'+str(6*num_res_blocks+2)
    # pretrain
    else:
        train_op = optimizer.minimize(loss=loss, global_step=global_step)
        model_name = model_dir+'/pre_plas_res'+str(6*num_res_blocks+2)    
    #-------------------------------------------------------------------------------------%
    classes = tf.argmax(input=logits, axis=1)
    correct_prediction = tf.equal(tf.cast(classes, tf.uint8), labels)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc = tf.identity(acc, name='accuracy_tensor')

    init_op = tf.global_variables_initializer()
        
    with tf.Session() as sess:
        sess.run(init_op)
    #-------------------------------------------------------------------------------------#
        saver = tf.train.Saver(max_to_keep=5)
        if transfer:
            restore_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='feature_extractor/')
            #var_list = load_layers(FLAGS.from_stack, 0)
            restorer = tf.train.Saver(var_list=restore_var_list)
            #util.init_from_checkpoint(pretrained_model, {'feature_extractor/':'transfer/'})
            restorer.restore(sess, pretrained_model)
    #-------------------------------------------------------------------------------------%
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
                if steps % FLAGS.eval_step == 0:
                    if not transfer:
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
        default='default')

    parser.add_argument(
        '--use_train',
        type=int,
        default='25')

    parser.add_argument(
        '--mode',
        type=str,
        default='transfer')

    parser.add_argument(
        '--from_stack',
        type=int,
        default='0')

    parser.add_argument(
        '--dataset',
        type=int,
        default='2')    

    parser.add_argument(
        '--eval_step',
        type=int,
        default='500')

    FLAGS, unparsed = parser.parse_known_args()
    tf.logging.info('plasticity: True')
    tf.logging.info('mode: '+str(FLAGS.mode))
    tf.logging.info('traing on dataset: '+str(FLAGS.dataset))
    tf.logging.info('use train: '+str(FLAGS.use_train))
    tf.logging.info('from stack: '+str(FLAGS.from_stack))
    tf.logging.info('model save dir: '+str(FLAGS.save_dir))
    tf.logging.info('eval step: '+str(FLAGS.eval_step))

    if FLAGS.mode == 'transfer':
        transfer = True
    elif FLAGS.mode == 'pretrain':
        transfer = False
    else:
        raise ValueError('unknown mode')

    inference(transfer)