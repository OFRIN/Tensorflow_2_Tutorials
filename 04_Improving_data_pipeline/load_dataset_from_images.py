import os

from core.dataset import *
from core.augmentors import *

from utility.timer import *

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
    'train_json_path' : './dataset/train.json',
    'test_json_path' : './dataset/test.json',

    'train_transforms' : train_transforms,
    'test_transforms' : test_transforms,

    'image_size' : [image_size, image_size],
    'batch_size' : batch_size,
    'classes' : classes
}
train_ds, test_ds = create_datasets_from_images(**dataset_option)

timer = Timer()
timer.tik()

for images, labels in train_ds:
    pass

# load dataset : 11sec
print('# load dataset : {}sec'.format(timer.tok()))