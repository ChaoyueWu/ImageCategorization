import argparse
import os

from skimage import transform
from skimage.io import imread
from skimage.io import imsave


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_loc', dest='data_loc', default=None, type=str,
                        help='data folder location')
    parser.add_argument('--output_loc', dest='output_loc', default=None, type=str,
                        help='output folder location')
    parser.add_argument('--image_size', dest='image_size', default=None, type=int,
                        help='image size')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    data_loc = os.path.expanduser(args.data_loc)
    output_loc = os.path.expanduser(args.output_loc)
    if not os.path.exists(output_loc):
        os.makedirs(output_loc)
    num = 1
    for path, _, file_names in os.walk(data_loc):
        for file_name in file_names:
            img = imread(os.path.join(path, file_name))
            resized_img = transform.resize(img, (args.image_size, args.image_size))
            imsave(os.path.join(output_loc, file_name), resized_img)
            print('{} {}'.format(num, file_name))
            num += 1
