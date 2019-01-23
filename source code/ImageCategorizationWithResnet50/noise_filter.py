import os
import shutil

import numpy as np
from skimage import io

if __name__ == '__main__':
    data_loc = os.path.expanduser('~/Disk/ic-data/train_data')
    clean_data_loc = os.path.expanduser('~/Disk/ic-data/clean_train_data')
    label_loc = os.path.expanduser('~/Disk/ic-data/train.label')
    clean_label_loc = os.path.expanduser('~/Disk/ic-data/clean_train.label')
    class_dict = {}

    with open(label_loc) as f:
        for line in f:
            index, class_val = line.split(' ')
            class_dict[index] = class_val
    if not os.path.exists(clean_data_loc):
        os.makedirs(clean_data_loc)
    noise_list = []
    for path, _, file_names in os.walk(data_loc):
        for file_name in file_names:
            image_loc = os.path.join(path, file_name)
            image = io.imread(image_loc)
            mean = np.mean(image)
            if not 5.0 < mean < 250:
                name = file_name.split('.')[0]
                noise_list.append(name)
                class_dict.pop(name)
            else:
                shutil.move(image_loc, clean_data_loc)
    with open(clean_label_loc, 'w') as f:
        for index, class_val in class_dict.items():
            f.write('{} {}'.format(index, class_val))
    print(len(noise_list))
    print(noise_list)
