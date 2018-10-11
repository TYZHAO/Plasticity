import tensorflow as tf
import numpy as np

writer = tf.python_io.TFRecordWriter('test.tfrecord')

raw = open('1.jpg','rb').read()
anzin = open('2.jpg','rb').read()
rw = open('3.jpg','rb').read()
c = tf.compat.as_bytes(raw)
d = tf.compat.as_bytes(anzin)
e = tf.compat.as_bytes(rw)
input_features = [tf.train.Feature(bytes_list = tf.train.BytesList(value=[c])),tf.train.Feature(bytes_list = tf.train.BytesList(value=[d])),
                  tf.train.Feature(bytes_list = tf.train.BytesList(value=[e]))]
print(type(input_features[0]))
feature_list = {
  'inputs': tf.train.FeatureList(feature=input_features)
}
feature_lists = tf.train.FeatureLists(feature_list=feature_list)
example = tf.train.SequenceExample(feature_lists=feature_lists)

serialized = example.SerializeToString()
writer.write(serialized)

writer.close()