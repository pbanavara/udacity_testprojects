'''
Generate training data and labels file from the UCF101 dataset.
Use the resulting file as an input to TensorFlow Queue 
'''
from __future__ import print_function
import os
import numpy as np
text_file = open('ucf101_data_labels.txt', 'w')
labels = []
for f in os.listdir("/opt/datasets/ucf101/"):
    if f.endswith(".avi"):
        label_name = f.split('_')[1]
        labels.append(label_name)
unique = reduce(lambda l, x: l.append(x) or l if x not in l else l, labels, [])
print(unique)
label_to_number = {}
for x in unique:
    label_to_number[x] = int(1000* np.random.random())
print(label_to_number)

for f in os.listdir("/opt/datasets/ucf101/"):
    if f.endswith(".avi"):
        label_name = f.split('_')[1]
        print(f + " " + str(label_to_number[label_name]), file = text_file)
