import os
import cv2
import json
import pickle

from core.dataset import *
from core.augmentors import *

from utility.utils import *

pickle_dir = './dataset/pickles/'
if not os.path.isdir(pickle_dir):
    os.makedirs(pickle_dir)

tags = ['Train', 'Test']
json_paths = ['./dataset/train.json', './dataset/test.json']

the_number_of_sample_per_pickle = 100

for tag, json_path in zip(tags, json_paths):
    data_dic = read_json(json_path)
    image_paths = list(data_dic.keys())

    dataset = []
    dataset_index = 1

    dataset_format = pickle_dir + '{}_{}.pkl'

    for image_path in image_paths:
        image = cv2.imread(image_path)
        label = data_dic[image_path]

        encoded_image = encode_image(image)

        dataset.append([image_path, encoded_image, label])
        
        if len(dataset) == the_number_of_sample_per_pickle:
            print(dataset_format.format(tag, dataset_index))
            dump_pickle(dataset_format.format(tag, dataset_index), dataset)

            dataset = []
            dataset_index += 1

    if len(dataset) > 0:
        print(dataset_format.format(tag, dataset_index))
        dump_pickle(dataset_format.format(tag, dataset_index), dataset)


