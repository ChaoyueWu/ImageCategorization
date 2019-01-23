import os
import re

class_dict = {}
clean_label_loc = os.path.expanduser('~/Disk/ic-data/clean_train.label')

with open(clean_label_loc) as f:
    for line in f:
        index, class_val = line.split(' ')
        class_dict[index] = class_val

data_dir = os.path.expanduser('~/Disk/ic-data/color_data')
lable_loc = os.path.expanduser('~/Disk/ic-data/color_train.label')

with open(lable_loc, 'w') as f:
    for file in os.listdir(data_dir):
        matchObj1 = re.match(r'(.*)_(.*).jpg', file, re.M | re.I)
        number = matchObj1.group(1)

        matchObj2 = re.match(r'(.*).jpg', file, re.M | re.I)
        name = matchObj2.group(1)

        class_val = class_dict[number]

        f.write('{} {}'.format(name, class_val))
