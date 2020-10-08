import os
import random
from shutil import copyfile

random.seed(123)

ROOT_PATH = './mini_imagenet/'

k = 5
target_path = './mini_imagenet_5_splits/'

'''
Splits the training set to 5 folds.
In each split, the held out set is used for test.
val is copied to be similar to the original val.
'''

csv_path = os.path.join(ROOT_PATH, 'train' + '.csv')
lines = [x.strip() for x in open(csv_path, 'r').readlines()]
title = lines[0]
lines = lines[1:]

# There are 600 images per class.
assert len(lines) % 600 == 0
num_classes = int(len(lines) / 600)

classes = list(range(num_classes))
random.shuffle(classes)

for i in range(k):
    if i == k - 1:
        upper = num_classes
    else:
        upper = round(num_classes/k)*(i+1)
    test_classes = classes[round(num_classes/k)*i: upper]
    train_classes = [c for c in classes if c not in test_classes]

    # The classes in the original train.csv are sorted in batches.
    test_lines = []
    for j in test_classes:
        test_lines.extend(lines[600*j:600*(j+1)])
    train_lines = []
    for j in train_classes:
        train_lines.extend(lines[600*j:600*(j+1)])

    print(f"split {i}: train: {len(train_classes)}, test: {len(test_classes)}")

    os.makedirs(os.path.join(target_path, str(i)), exist_ok=True)
    with open(os.path.join(target_path, str(i), 'train.csv'), 'w') as f:
        f.write(title + '\n')
        f.writelines([x + '\n' for x in train_lines])
    with open(os.path.join(target_path, str(i), 'test.csv'), 'w') as f:
        f.write(title + '\n')
        f.writelines([x + '\n' for x in test_lines])

    copyfile(os.path.join(ROOT_PATH, 'val.csv'), os.path.join(target_path, str(i), 'val.csv'))
