import tensorflow as tf
import numpy as np
import cv2

writer = tf.python_io.TFRecordWriter('test.tfrecord')
clips = 4
tf.reset_default_graph()
#file_name = '/home/rw1691/2018summer/trainlist.txt'
file_name = 'F:\\data\\ucf101_jpegs_256.zip~\\train_list.txt'
train_img = []
for line in open(file_name):
    line = line.split()
    train_img.append(line)

#train_img = [item[0] for item in train_img]

train_img = [train_img[x:x+clips] for x in range(0, len(train_img), clips)]
batch_data = []
print(len(train_img))
for index,clip in enumerate(train_img):
    cc = []
    for img in clip:
        std_img = cv2.resize(cv2.imread(str(img[0])),(224,224))
        img_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, img_encode = cv2.imencode('.jpg', std_img, img_param)
        std_img = tf.compat.as_bytes(img_encode.tostring())
        batch_data.append(std_img)


    input_features = [tf.train.Feature(bytes_list = tf.train.BytesList(value=[item])) for item in batch_data]
    feature_list = {
      'inputs': tf.train.FeatureList(feature=input_features)
    }
    feature_lists = tf.train.FeatureLists(feature_list=feature_list)
    example = tf.train.SequenceExample(feature_lists=feature_lists)

    serialized = example.SerializeToString()
    writer.write(serialized)
    print('finish write to {}'.format(index))
writer.close()