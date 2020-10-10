import os
from smoke.utils.mio import save_string_list
import numpy as np

kitti_path = "/home/zhangxiao/data/kitti"
train_test_dirs = os.listdir(kitti_path)

def split(image_list, shuffle=True, ratio=0.8):
    image_nums = len(image_list)
    offset = int(image_nums * ratio)
    if image_nums == 0 or offset < 1:
        return [], image_list

    if shuffle:
        np.random.shuffle(image_list)
    train_list = image_list[:offset]
    val_list = image_list[offset:]
    return train_list, val_list


for sub_dir in train_test_dirs:
    if not os.path.isdir(sub_dir):
        continue
    imageSets_path = os.path.join(kitti_path, sub_dir, "ImageSets")
    if not os.path.exists(imageSets_path):
        os.makedirs(imageSets_path)
    
    image_path = os.path.join(kitti_path, sub_dir, "image_2")
    image_list = sorted(os.listdir(image_path), key=lambda x: int(x.split('.')[0]))
    image_list = [img.split('.')[0] for img in image_list]

    image_nums = len(image_list)
    shuffle = True
    if "test" in sub_dir:
        shuffle = False
        ratio = 0
    else:
        ratio = 0.9
    train_split, test_split = split(image_list, shuffle=shuffle, ratio=ratio)

    save_string_list(os.path.join(imageSets_path, "train.txt"), train_split)
    save_string_list(os.path.join(imageSets_path, "trainval.txt"), test_split)
    