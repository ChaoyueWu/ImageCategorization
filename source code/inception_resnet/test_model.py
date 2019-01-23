import argparse
import os
import math

import tensorflow as tf
import cv2
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt

from Alexnet.alexnet import AlexNet
from inception_resnet.inception_resnet_v2 import inception_resnet_v2
from inception_resnet.inception_resnet_v2 import inception_resnet_v2_arg_scope

slim = tf.contrib.slim

SUPPORT_MODEL_TYPE = ['inception_resnet_v2', 'alexnet']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_loc', dest='model_loc', default=None, type=str,
                        help='model location')
    parser.add_argument('--model_name', dest='model_name', default=None, type=str,
                        help='model name')
    parser.add_argument('--num_class', dest='num_class', default=None, type=int,
                        help='number of classes')
    parser.add_argument('--input_directory_loc', dest='input_loc', default=None, type=str,
                        help='input directory location')
    parser.add_argument('--prediction_batch_size', dest='batch_size', default=8, type=int,
                        help='size of per prediction batch')
    parser.add_argument('--input_size', dest='input_size', default=299, type=int,
                        help='input size')
    parser.add_argument('--model_type', dest='model_type', default=None, type=str,
                        help='type of model')
    parser.add_argument('--output_label_loc', dest='output_label_loc', default=None, type=str,
                        help='output label file location')
    return parser.parse_args()


class ImageSet:
    def __init__(self, image_set_dir_location, size=(299, 299), batch_size=8):
        if not os.path.exists(image_set_dir_location):
            print('No image set found!')
            exit(-1)
        if type(size) is not tuple:
            print('size: (height, width), found: {}'.format(size))
            exit(-1)
        self.size = size
        self.batch_size = batch_size
        self.image_loc = [os.path.join(image_set_dir_location, name) for name in os.listdir(image_set_dir_location)]

    def has_next_batch(self):
        return not len(self.image_loc) == 0

    def get_steps_num_left(self):
        return int(math.ceil(len(self.image_loc)) / self.batch_size)

    def get_next_batch(self):
        if not self.has_next_batch():
            print('No next batch!')
            return None
        this_batch_num = min(len(self.image_loc), self.batch_size)
        this_batch_loc = self.image_loc[:this_batch_num]
        this_batch_name = [os.path.split(loc)[-1].split('.')[0] for loc in this_batch_loc]
        self.image_loc = self.image_loc[this_batch_num:]
        this_batch = [cv2.imread(img_loc) for img_loc in this_batch_loc]
        this_batch = [transform.resize(img, (self.size[0], self.size[1])) for img in this_batch]
        return np.array(this_batch), this_batch_name


if __name__ == '__main__':
    args = get_args()
    if args.model_type not in SUPPORT_MODEL_TYPE:
        print('Supported models: {}'.format(SUPPORT_MODEL_TYPE))
        exit(-1)
    input_loc = os.path.expanduser(args.input_loc)
    if not os.path.exists(input_loc):
        print('Path {} not found!'.format(input_loc))
        exit(-1)
    model_loc = os.path.expanduser(args.model_loc)
    with tf.Graph().as_default() as graph:
        image_file_list = [os.path.join(input_loc, p) for p in os.listdir(input_loc)]
        image_name_list = [p.split('.')[0] for p in os.listdir(input_loc)]
        data_set = ImageSet(input_loc, (args.input_size, args.input_size), args.batch_size)
        with tf.Session() as sess:
            x = tf.placeholder(tf.float32, [args.batch_size, args.input_size, args.input_size, 3])
            if args.model_type == 'inception_resnet_v2':
                with slim.arg_scope(inception_resnet_v2_arg_scope()):
                    model, end_points = inception_resnet_v2(x, num_classes=args.num_class, is_training=False)
                predictions = tf.argmax(end_points['Predictions'], 1)
            else:
                model = AlexNet(x, keep_prob=1, num_classes=args.num_class, skip_layer=[])
                predictions = tf.argmax(model.fc8, 1)
            variables_to_restore = slim.get_variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)

            output_label_loc = os.path.expanduser(args.output_label_loc)
            output_label_parent_loc = os.path.dirname(output_label_loc)
            if not os.path.exists(output_label_parent_loc):
                os.makedirs(output_label_parent_loc)
            total_num = len(image_name_list)
            steps_num = int(math.ceil(total_num / float(args.batch_size)))

            saver.restore(sess, tf.train.latest_checkpoint(model_loc))
            i = 1
            steps = data_set.get_steps_num_left()
            with open(output_label_loc, 'w') as f:
                try:
                    while data_set.has_next_batch():
                        batch = data_set.get_next_batch()
                        for img in batch[0]:
                            plt.imshow(img)
                        prediction_result = sess.run(predictions, feed_dict={x: batch[0]})
                        prediction_images = batch[1]
                        zipped = zip(prediction_images, prediction_result)
                        for item in zipped:
                            print('({}/{}) {} {}'.format(i, steps_num + 1, item[0], item[1]))
                            f.write('{} {}\n'.format(item[0], item[1]))
                        i += 1
                except tf.errors.OutOfRangeError:
                    print("That's the end of the prediction...")
