import argparse
import os
import shutil


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_loc', dest='data_loc', default=None, type=str,
                        help='data folder location')
    parser.add_argument('--label_loc', dest='label_loc', default=None, type=str,
                        help='label location')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    data_loc = os.path.expanduser(args.data_loc)
    label_loc = os.path.expanduser(args.label_loc)
    assert os.path.exists(data_loc)
    assert os.path.exists(label_loc)
    class_dict = {}
    with open(label_loc) as f:
        for line in f:
            line = line.strip()
            index, class_val = line.split(' ')
            class_dict[index] = class_val
    for path, _, file_names in os.walk(data_loc):
        for file_name in file_names:
            name = file_name.split('.')[0]
            class_name = class_dict[name]
            class_loc = os.path.join(data_loc, class_name)
            if not os.path.exists(class_loc):
                os.makedirs(class_loc)
            shutil.move(os.path.join(path, file_name), class_loc)
