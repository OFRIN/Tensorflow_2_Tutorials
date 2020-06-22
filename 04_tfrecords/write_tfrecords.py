import os
import cv2
import json
import pickle

import tensorflow as tf

from core.dataset import *
from core.augmentors import *

from utility.utils import *
from utility.tensorflow_utils import *

tfrecord_dir = './dataset/tfrecords/'
if not os.path.isdir(tfrecord_dir):
    os.makedirs(tfrecord_dir)

tfrecord_paths = [tfrecord_dir + 'train.tfrecord', tfrecord_dir + 'test.tfrecord']
json_paths = ['./dataset/train.json', './dataset/test.json']

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def example_for_classification(image_path, label):
    image = cv2.imread(image_path)

    h, w, c = image.shape
    encoded_image = encode_image(image)

    feature = {
        'height' : _int64_feature(h),
        'width' : _int64_feature(w),
        'depth' : _int64_feature(c),
        'label' : _int64_feature(label),
        'image_raw' : _bytes_feature(encoded_image.tobytes())
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

for tfrecord_path, json_path in zip(tfrecord_paths, json_paths):
    data_dic = read_json(json_path)
    image_paths = list(data_dic.keys())
    
    with tf.io.TFRecordWriter(tfrecord_path) as writer:
        for image_path in image_paths:
            label = data_dic[image_path]

            tf_example = example_for_classification(image_path, label)
            writer.write(tf_example.SerializeToString())

