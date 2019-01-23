import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.callbacks import ModelCheckpoint, TensorBoard

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

LOG_DIR = os.path.expanduser('~/Disk/logs/resnet50/inception')
MODEL_FILE_PATH = os.path.expanduser('~/Disk/models/resnet50/inception/checkpoint-{epoch:02d}-{val_loss:.4f}.hdf5')   # 模型Log文件以及.h5模型文件存放地址

FC_SIZE = 1024

# 添加新层
def add_new_last_layer(base_model, nb_classes):
  """
  添加最后的层
  输入
  base_model和分类数量
  输出
  新的keras的model
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(input=base_model.input, output=predictions)
  return model

# 冻上base_model所有层，这样就可以正确获得bottleneck特征
def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
  return cnt

def main():
    global Width, Height, pic_dir_out, pic_dir_data
    output_model_file = os.path.expanduser("~/Disk/models/resnet50/inception/model.h5")
    Width = 299
    Height = 299
    train_dir = os.path.expanduser('~/Disk/ic-data/total_data_2/train')  # 训练集数据
    val_dir = os.path.expanduser('~/Disk/ic-data/total_data_2/val')  # 验证集数据

    nb_classes = 20
    nb_epoch = 5
    batch_size = 90

    nb_train_samples = get_nb_files(train_dir)  # 训练样本个数
    nb_classes = len(glob.glob(train_dir + "/*"))  # 分类数
    nb_val_samples = get_nb_files(val_dir)  # 验证集样本个数
    nb_epoch = int(nb_epoch)  # epoch数量
    batch_size = int(batch_size)

    # 　图片生成器
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # 训练数据与测试数据
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(Width, Height),
        batch_size=batch_size, class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(Width, Height),
        batch_size=batch_size, class_mode='categorical')

    base_model = InceptionV3(weights="imagenet" ,include_top=False)
    model = add_new_last_layer(base_model, nb_classes)  # 从基本no_top模型上添加新层
    setup_to_transfer_learn(model, base_model)  # 冻结base_model所有层

    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    tensorboard = TensorBoard(log_dir=LOG_DIR, write_images=True)
    checkpoint = ModelCheckpoint(filepath=MODEL_FILE_PATH, monitor='val_loss', verbose=1, save_best_only=True)
    model.fit_generator(
        train_generator,
        steps_per_epoch=780,
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=30,
        class_weight='auto',
        callbacks=[tensorboard, checkpoint]
        , verbose=1
    )

    # 模型保存
    model.save(output_model_file)

if __name__ == '__main__':
    main()