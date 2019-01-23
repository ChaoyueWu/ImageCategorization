import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator,load_img
from keras.optimizers import SGD

from keras.preprocessing import image
import numpy as np
from keras.models import load_model
import re
from PIL import Image
import time

target_size = (224, 224)

change_list = {
    1:1,
    2:10,
    3:11,
    4:12,
    5:13,
    6:14,
    7:15,
    8:16,
    9:17,
    10:18,
    11:19,
    12:2,
    13:20,
    14:3,
    15:4,
    16:5,
    17:6,
    18:7,
    19:8,
    20:9
}


# 预测函数
# 输入：model，图片，目标尺寸
# 输出：预测predict
def predict(model, img, target_size):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return preds[0]

def get_label_from_preds(preds):
    index = -1
    max = -1
    for pred in preds:
        if pred > max:
            max = pred
            index = preds.tolist().index(pred)
    return change_list[index + 1]

# 7
# ~/Disk/models/resnet50/7/checkpoint-134-0.3948.hdf5   0.9383333333333334  0.9483631454606818
# ~/Disk/models/resnet50/7/checkpoint-127-0.3993.hdf5   0.9396666666666667  0.9497131285858926
# ~/Disk/models/resnet50/7/checkpoint-114-0.4032.hdf5   0.936   0.9460006749915626
# ~/Disk/models/resnet50/7/checkpoint-106-0.4069.hdf5   0.9363333333333334  0.9463381707728653
# 6
# ~/Disk/models/resnet50/6/checkpoint-195-0.3750.hdf5   0.9356666666666666  0.9460006749915626
# ~/Disk/models/resnet50/6/checkpoint-181-0.3847.hdf5   0.9343333333333333  0.9449881876476544
# ~/Disk/models/resnet50/6/checkpoint-178-0.3925.hdf5   0.9316666666666666  0.9419507256159299

model_path = os.path.expanduser("~/Disk/models/resnet50/7/checkpoint-127-0.3993.hdf5")
image_path = os.path.expanduser("~/Disk/ic-data/test_data/")
file_path = os.path.expanduser("~/Disk/ic-data/result.list")
real_lable = os.path.expanduser("~/Disk/ic-data/split_val.label")

class_dict = {}
with open(real_lable) as f:
    for line in f:
        index, class_val = line.split(' ')
        class_dict[index] = int(class_val)

our_class_dict = {}
# 载入模型

i = 1
t0 = time.time()
model = load_model(model_path)
with open(file_path, 'w') as f:
    for file in os.listdir(image_path):
        result = re.findall(r"(.*).jpg", file)
        number = result[0]
        img = Image.open(os.path.join(image_path,file))
        preds = predict(model, img, target_size)
        # print (preds)
        pred = get_label_from_preds(preds)
        # print (pred)
        our_class_dict[number] = pred
        f.write('{}\t{}\n'.format(number, pred))
        print(i)
        i += 1

# print(our_class_dict)

# count = 0
# totalCount = 0
# for number in our_class_dict:
#     totalCount += 1
#     tmp = class_dict[number]
#     pred = our_class_dict[number]
#     if tmp == pred:
#         count += 1
#
# rmCount = 0
# rmTotalCount = 0
# for number in our_class_dict:
#     if number in ["6961","6998","23510","26633","1751","2609","15146","2439","471","8483","23309",
#                   "24146","2854","10116","10211","14792","2051","8829","11904","26082","921","18886",
#                   "11312","32231","4408","23240","17439","17874","27996","2462","8980","4238","2097",
#                   "25004","2225","3021","22035"]:
#         continue
#     rmTotalCount += 1
#     tmp = class_dict[number]
#     pred = our_class_dict[number]
#     if tmp == pred:
#         rmCount += 1
#
# print (count)
# print (totalCount)
# print(count / totalCount)
#
# print (rmCount)
# print(rmTotalCount)
# print(rmCount / rmTotalCount)

t1 = time.time()
print(t1 - t0)




