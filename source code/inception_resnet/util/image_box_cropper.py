import argparse
import os

from lxml import etree
from skimage.io import imread
from skimage.io import imsave

class_dict = {
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20
}


def secure_cast(maybe_float):
    try:
        return int(maybe_float)
    except:
        return int(float(maybe_float))


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc_loc', dest='voc_loc', default=None, type=str,
                        help='voc folder location')
    parser.add_argument('--output_loc', dest='output_loc', default=None, type=str,
                        help='output folder location')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arg_parser()
    annotations_loc = os.path.join(os.path.expanduser(args.voc_loc), 'Annotations')
    jpegimages_loc = os.path.join(os.path.expanduser(args.voc_loc), 'JPEGImages')

    output_train_folder_loc = os.path.join(os.path.expanduser(args.output_loc), 'train_data')
    if not os.path.exists(output_train_folder_loc):
        os.makedirs(output_train_folder_loc)
    output_label_loc = os.path.join(os.path.expanduser(args.output_loc), 'train.label')

    label_writer = open(output_label_loc, 'w')

    for _, _, file_names in os.walk(annotations_loc):
        file_index = 1
        for file_name in file_names:
            image_name = file_name.split('.')[0]
            root = etree.parse(os.path.join(annotations_loc, file_name))
            objects = root.xpath('/annotation/object')
            obj_index = 1
            img = imread(os.path.join(jpegimages_loc, '{}.jpg'.format(image_name)))
            for obj in objects:
                obj_class = obj.xpath('./name')[0].text
                bnd_box = obj.xpath('./bndbox')[0]
                box_x_min = secure_cast(bnd_box.xpath('./xmin')[0].text)
                box_y_min = secure_cast(bnd_box.xpath('./ymin')[0].text)
                box_x_max = secure_cast(bnd_box.xpath('./xmax')[0].text)
                box_y_max = secure_cast(bnd_box.xpath('./ymax')[0].text)

                cropped_name = '{}_{}'.format(image_name, obj_index)
                cropped_img = img[box_y_min:box_y_max, box_x_min:box_x_max]
                imsave(os.path.join(output_train_folder_loc, '{}.jpg'.format(cropped_name)), cropped_img)

                label_writer.write('{} {}{}'.format(cropped_name, class_dict[obj_class], os.linesep))
                obj_index += 1
                print('{} {}'.format(file_index, cropped_name))
            file_index += 1

    label_writer.close()
