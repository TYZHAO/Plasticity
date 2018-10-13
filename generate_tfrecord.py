import tensorflow as tf
import numpy as np
import random
import cv2

tf.logging.set_verbosity(tf.logging.INFO)

writer = tf.python_io.TFRecordWriter('/beegfs/rw1691/inputs.tfrecord')
clip_num = 10
tf.reset_default_graph()

file_name = '/home/rw1691/2018summer/trainlist.txt'

train_img = []
for line in open(file_name):
    line = line.split()
    train_img.append(line)

train_img = [train_img[x:x+clip_num] for x in range(0, len(train_img), clip_num)]

train_img = [ train_img[i] for i in sorted(random.sample(range(len(train_img)), 20000)) ]
#train_img = random.sample(train_img, 20000)

tf.logging.info(len(train_img))
for index,clip in enumerate(train_img):
    clips = []
    for img in clip:
        std_img = cv2.resize(cv2.imread(str(img[0])),(64,64))
        img_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, img_encode = cv2.imencode('.jpg', std_img, img_param)
        std_img = tf.compat.as_bytes(img_encode.tostring())
        clips.append(std_img)

    input_features = [tf.train.Feature(bytes_list = tf.train.BytesList(value=[item])) for item in clips]
    feature_list = {
      'inputs': tf.train.FeatureList(feature=input_features)
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    example = tf.train.SequenceExample(feature_lists=feature_lists)

    serialized = example.SerializeToString()
    writer.write(serialized)
    if index%100 == 0:
        tf.logging.info('finish write to {}'.format(index))

writer.close()