import os
import shutil
import re
import random

_NUM_TEST = 0

original_data_dir = os.path.expanduser("~/Disk/ic-data/clean_train_data")
original_lable = os.path.expanduser("~/Disk/ic-data/clean_train.label")

aug_data_dir = os.path.expanduser("~/Disk/ic-data/aug_train_data")
aug_lable = os.path.expanduser("~/Disk/ic-data/aug_train.label")

color_data_dir = os.path.expanduser("~/Disk/ic-data/color_train_data")
color_lable = os.path.expanduser("~/Disk/ic-data/color_train.label")

voc_data_dir = os.path.expanduser("~/Disk/ic-data/voc_train_data")
voc_lable = os.path.expanduser("~/Disk/ic-data/voc_train.label")

dst_train_dir = os.path.expanduser("~/Disk/ic-data/total_data_2/train")
dst_val_dir = os.path.expanduser("~/Disk/ic-data/total_data_2/val")

class_dict = {}

data_dir = voc_data_dir
lable = voc_lable

train_label_loc = os.path.expanduser('~/Disk/ic-data/split_train.label')
test_label_loc = os.path.expanduser('~/Disk/ic-data/split_val.label')

#生成文件与class对应的dict
with open(lable) as f:
    for line in f:
        index, class_val = line.split(' ')
        class_dict[index] = class_val
# with open(voc_data_dir) as f:
#     for line in f:
#         index, class_val = line.split(' ')
#         class_dict[index] = class_val

file_list = []
for file in os.listdir(data_dir):
    file_list.append(file)
# for file in os.listdir(voc_data_dir):
#     file_list.append(os.path.join(voc_data_dir, file))

random.seed(6666)
random.shuffle(file_list)
training_filenames = file_list[_NUM_TEST:]
testing_filenames = file_list[:_NUM_TEST]

for file in training_filenames:
    result = re.findall(r"(.*).jpg", file)
    number = result[0]
    image_loc = os.path.join(data_dir, file)
    class_name = class_dict[number].replace('\n', '')
    doc_name = os.path.join(dst_train_dir, class_name)
    print (image_loc, "to", doc_name)
    if not os.path.exists(doc_name):
        os.makedirs(doc_name)
    dst_loc = os.path.join(doc_name, file)
    shutil.copyfile(image_loc, dst_loc)

for file in testing_filenames:
    result = re.findall(r"(.*).jpg", file)
    number = result[0]
    image_loc = os.path.join(data_dir, file)
    class_name = class_dict[number].replace('\n','')
    doc_name = os.path.join(dst_val_dir, class_name)
    print (image_loc, "to", doc_name)
    if not os.path.exists(doc_name):
        os.makedirs(doc_name)
    dst_loc = os.path.join(doc_name,file)
    shutil.copyfile(image_loc, dst_loc)

# with open(train_label_loc, 'w') as f:
#     for file in training_filenames:
#         result = re.findall(r"(.*).jpg", file)
#         number = result[0]
#         class_name = class_dict[number]
#         f.write('{} {}'.format(number, class_name))
#
# with open(test_label_loc, 'w') as f:
#     for file in testing_filenames:
#         result = re.findall(r"(.*).jpg", file)
#         number = result[0]
#         class_name = class_dict[number]
#         f.write('{} {}'.format(number, class_name))



