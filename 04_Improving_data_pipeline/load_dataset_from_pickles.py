import os
import glob

from core.dataset import *
from core.augmentors import *

from utility.timer import *

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

if __name__ == '__main__':
    
    # 1. Dataset
    image_size = 299
    batch_size = 16
    classes = 5

    max_image_size = int(image_size * 1.25)
    min_image_size = max_image_size // 2

    train_transforms = DataAugmentation(
            [
                Random_Resize(min_image_size, max_image_size),
                RandAugment(),
                Random_Crop_with_Black((image_size, image_size))
            ]
    )

    test_transforms = DataAugmentation(
        [
            Fixed_Resize(image_size),
            Top_Left_Crop((image_size, image_size))
        ]
    )

    dataset_option = {
        'train_pickle_paths' : glob.glob('./dataset/pickles/Train*'),
        'test_pickle_paths' : glob.glob('./dataset/pickles/Test*'),

        'train_transforms' : train_transforms,
        'test_transforms' : test_transforms,

        'image_size' : [image_size, image_size],
        'batch_size' : batch_size,

        'selected_pickle_size' : 5,
        'use_cores' : 4,
        'max_size' : 5,
    }
    train_ds, test_ds = create_datasets_from_pickles(**dataset_option)

    timer = Timer()
    
    train_ds.init()
    test_ds.init()
    
    timer.tik()

    for images, labels in train_ds:
        pass

    for images, labels in test_ds:
        pass

    # train_ds.join()
    # test_ds.join()

    # load dataset : 11sec -> 4sec
    print('# load dataset : {}sec'.format(timer.tok()))
