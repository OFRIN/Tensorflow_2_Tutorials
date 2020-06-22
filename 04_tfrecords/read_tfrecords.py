
import tensorflow as tf

dataset = tf.data.TFRecordDataset('./dataset/tfrecords/train.tfrecord')

feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64),
    'width': tf.io.FixedLenFeature([], tf.int64),
    'depth': tf.io.FixedLenFeature([], tf.int64),
    'label': tf.io.FixedLenFeature([], tf.int64),
    'image_raw': tf.io.FixedLenFeature([], tf.string),
}

def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)

for features in dataset.map(_parse_function):
    features['image_raw'].numpy()