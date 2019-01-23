import argparse
import os

items_to_descriptions = {
    'image': 'A 3-channel RGB coloured flower image that is either tulips, sunflowers, roses, dandelion, or daisy.',
    'label': 'A label that is as such -- 0: aeroplane, 1: cow, 2:dining table, 3: dog, 4: horse, 5: motorbike, '
             '6: person, 7: potted plant, 8: sheep, 9: sofa, 10: train, 11: bicycle, 12: tv monitor, 13: bird, '
             '14: boat, 15: bottle, 16: bus, 17: car, 18: cat, 19: chair'
}

MIDDLE_TRANSFORM_GATE = {
    '0': '1',
    '1': '10',
    '2': '11',
    '3': '12',
    '4': '13',
    '5': '14',
    '6': '15',
    '7': '16',
    '8': '17',
    '9': '18',
    '10': '19',
    '11': '2',
    '12': '20',
    '13': '3',
    '14': '4',
    '15': '5',
    '16': '6',
    '17': '7',
    '18': '8',
    '19': '9',
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_label_loc', dest='eval_label_loc', default=None, type=str,
                        help='location of label file that is to be evaluated')
    parser.add_argument('--true_label_loc', dest='true_label_loc', default=None, type=str,
                        help='location of ground truth label file')
    parser.add_argument('--middle_transform', dest='middle_transform', default=True, type=bool,
                        help='whether there is a middle transform gate')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    eval_label_loc = os.path.expanduser(args.eval_label_loc)
    true_label_loc = os.path.expanduser(args.true_label_loc)
    eval_val = {}
    print('reading eval label...')
    with open(eval_label_loc) as f:
        for line in f:
            name, class_val = line.strip().split(' ')
            if args.middle_transform:
                eval_val[name] = MIDDLE_TRANSFORM_GATE[class_val]
            else:
                eval_val[name] = class_val
    true_val = {}
    print('reading true label...')
    with open(true_label_loc) as f:
        for line in f:
            name, class_val = line.strip().split(' ')
            true_val[name] = class_val
    print('evaluating...')
    match = 0
    for key, value in eval_val.items():
        if value == true_val[key]:
            match += 1
    print('Acc: {}'.format(match / len(eval_val)))
