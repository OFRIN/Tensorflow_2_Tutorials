import copy
import glob

import numpy as np

from utility.utils import *

pickle_paths = glob.glob('./dataset/pickles/Train_*')
pickle_length = len(pickle_paths)

np.random.shuffle(pickle_paths)

selected_size = 5

loop_pickle_paths = copy.deepcopy(pickle_paths)
while len(loop_pickle_paths) > 0:

    sub_pickle_paths = loop_pickle_paths[:selected_size]
    loop_pickle_paths = loop_pickle_paths[selected_size:]

    dataset = []
    for pickle_path in sub_pickle_paths:
        for image_name, image, label in load_pickle(pickle_path):
            image = decode_image(image)
            
            dataset.append([image, label])
    
    print(len(dataset))

