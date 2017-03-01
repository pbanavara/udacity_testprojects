from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

#constants
CROPPED_IMAGE_SIZE = 24

def read_image(input_queue, label_queue):
    image = input_queue
    label = label_queue
    image_contents = tf.read_file(image)
    image_tensor = tf.image.decode_png(image_contents, channels = 3)
    return image_tensor, label

"""
Create distorted images from the input filename. Input file name is of the
format 'filename, label' where filename refers to the image and label refers to
the corresponding label. The output of this function is an image tensor and a
label tensor denoting the class of the image
"""
def distorted_inputs(file_name, batch_size):
    f = open(file_name, 'r')
    filenames = []
    labels = []
    for line in f:
        filename, label = line[:-1].split(' ')
        labels.append(int(label))
        filenames.append(filename)
    images = ops.convert_to_tensor(filenames, dtype = dtypes.string)
    new_labels = ops.convert_to_tensor(labels, dtype = dtypes.int32)
    input_queue, label_queue = tf.train.slice_input_producer([images, new_labels],
                                                shuffle=False)
    temp_list = []
    for i in range(batch_size):
        image, label = read_image(input_queue, label_queue)
        cropped_height = CROPPED_IMAGE_SIZE
        cropped_width = CROPPED_IMAGE_SIZE

        reshaped_image = tf.cast(image, tf.float32)
        distorted_image = tf.random_crop(reshaped_image, [cropped_height, cropped_width, 3])
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        float_image = tf.image.per_image_standardization(distorted_image)
        float_image.set_shape([cropped_width, cropped_height, 3])
        temp_list.append(float_image)
    print("Temp List :::" , temp_list)
    train_image_batch, train_label_batch = tf.train.batch(
                                    [float_image, label],
                                    batch_size=5,
                                    num_threads=1)
    print ("Train image batch size ", train_image_batch.get_shape())
    print ("Train image label size ", train_label_batch.get_shape())
    return train_image_batch, train_label_batch
