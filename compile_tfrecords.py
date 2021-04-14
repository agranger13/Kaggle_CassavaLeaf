import numpy as np
import pandas as pd
import random
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf1
from PIL import Image
import io
import os
from functools import partial
    
def split_tfrecord(tfrecord_path, split_size):
    with tf.Graph().as_default(), tf1.Session() as sess:
        ds = tf1.data.TFRecordDataset(tfrecord_path).batch(split_size)
        batch = ds.make_one_shot_iterator().get_next()
        part_num = 0
        while True:
            try:
                records = sess.run(batch)
                part_path = tfrecord_path + '.{:03d}'.format(part_num)
                with tf1.python_io.TFRecordWriter(part_path) as writer:
                    for record in records:
                        writer.write(record)
                part_num += 1
            except tf1.errors.OutOfRangeError: break

# In[24]:
if __name__ == "__main__":

    VALIDATION_SPLIT = 0.33
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    IMAGE_SIZE = [512, 512]

    epochs = 400
    batch_size = 1024

    # filenames=tf.io.gfile.glob('./input/cassava-leaf-disease-classification/train_tfrecords/*.tfrec')
    # print(validation_filenames)
    train_csv= pd.read_csv("./input/train.csv")
    train_directory="./input/resize_images/"

    augmented_csv= pd.read_csv("./input/train_augmented.csv")
    augmented_directory="./input/augmented_images/"
    augmented_output="./input/augmented_tfrecords/"

    train_csv["label"] = train_csv["label"].astype(int)
    augmented_csv["label"] = augmented_csv["label"].astype(int)

    images=[]
    labels=[]
    for row in train_csv.index:
        img = Image.open(train_directory + train_csv['image_id'][row])

        im_resize = img.resize((256, 256))
        buf = io.BytesIO()
        im_resize.save(buf, format='JPEG')
        byte_im = buf.getvalue()

        images.append(byte_im)
        labels.append(train_csv['label'][row])

    for row in augmented_csv.index:
        img = Image.open(augmented_directory + augmented_csv['image_id'][row])

        im_resize = img.resize((256, 256))
        buf = io.BytesIO()
        im_resize.save(buf, format='JPEG')
        byte_im = buf.getvalue()

        images.append(byte_im)
        labels.append(augmented_csv['label'][row])
    
    tfrecord_writer = tf.io.TFRecordWriter(augmented_output + "data.tfrec")
    # iterate over images in directory
    for img, label in zip(images, labels):
        # create an example with the image and label
        print(type(img))
        example = tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img])),
        'target': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        # write example
        tfrecord_writer.write(example.SerializeToString())
    # close writer
    tfrecord_writer.close()

    raw_dataset = tf.data.TFRecordDataset(augmented_output + "data.tfrec")

    shards = 15

    for i in range(shards):
        writer = tf.data.experimental.TFRecordWriter(f""+augmented_output + "data-part-"+str(i)+".tfrec")
        writer.write(raw_dataset.shard(shards, i))

