#!/usr/bin/env python
# coding: utf-8

# # NOTEBOOK Kaggle Cassava Competion

# In[20]:


import numpy as np
import pandas as pd
import random
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import cv2
import os
from functools import partial
from PIL import Image


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## I. Pré-traitement
# ### a. Import des librairies

# ### b. Fonction scheduler

# In[2]:


def scheduler(epoch, lr):
    if epoch < 40:
        return lr
    else:
        return lr * tf.math.exp(-0.05)


# ### c. Fonction plots

# In[3]:


def plot_logs(all_logs):
    for logs in all_logs:
        losses = logs.history['loss']
        val_losses = logs.history['val_loss']
        plt.plot(list(range(len(losses))), losses, label="Train Loss")
        plt.plot(list(range(len(losses))), val_losses, label="Val Loss")
        plt.title("Evolution du loss") 
    plt.show()

    # for logs in all_logs:
    #     losses = logs.history['val_loss']
    #     plt.plot(list(range(len(losses))), losses)
    #     plt.title("Evolution du val_loss") 
    # plt.show()

    for logs in all_logs:
        metric = logs.history['categorical_accuracy']
        val_metric = logs.history['val_categorical_accuracy']
        plt.plot(list(range(len(metric))), metric, label="Train Accuracy")
        plt.plot(list(range(len(metric))), val_metric, label="Train Accuracy")
        plt.title("Evolution de l'accuracy") 
    plt.show()

    # for logs in all_logs:
    #     metric = logs.history['val_categorical_accuracy']
    #     plt.plot(list(range(len(metric))), metric)
    #     plt.title("Evolution de val_accuracy") 
    # plt.show()


#  ## II. Définition des modèles
#  ### a. Modèle linéaire

# In[12]:


def test_linear_model(train_dataset, val_dataset, opt, loss_func, epochs, batch_size):
    model = keras.models.Sequential()

    model.add(keras.layers.Input(shape=(32, 32, 3)))
    
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(5, activation=keras.activations.softmax))

    model.compile(optimizer=opt, loss=loss_func, metrics=keras.metrics.categorical_accuracy)

    logs = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, batch_size=batch_size,
                    steps_per_epoch=5, validation_steps=2,
                    callbacks=[keras.callbacks.LearningRateScheduler(scheduler)])

    model.summary()

    return logs


# In[6]:


def test_fully_connected(x, y, val_x, val_y, opt, loss_func, epochs, batch_size):
    model = keras.models.Sequential()
    #model.add(keras.Input(shape=(*IMAGE_SIZE,3)))
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(128, activation=keras.activations.tanh,
                                 kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(64, activation=keras.activations.tanh,
                                 kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(32, activation=keras.activations.tanh,
                                 kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(16, activation=keras.activations.tanh,
                                 kernel_regularizer=keras.regularizers.l2(0.01)))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(5, activation=keras.activations.softmax,
                                 kernel_regularizer=keras.regularizers.l2(0.001)))

    model.compile(optimizer=opt, loss=loss_func, metrics=keras.metrics.categorical_accuracy)

    logs = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size,
                     callbacks=[keras.callbacks.LearningRateScheduler(scheduler)])

    model.summary()

    return logs


# ### b. Modèle Perceptron Multicouche

# ### c. Modèle ConvNet

# In[7]:


def test_conv_net(x, y, val_x, val_y, opt, loss_func, epochs, batch_size):

    model = keras.models.Sequential()

    #model.add(keras.layers.Reshape((128, 128, 1)))
    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation=keras.activations.tanh,
                                  kernel_regularizer=keras.regularizers.l2(0.2)))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Dropout(0.2))

    # model.add(keras.layers.Conv2D(64, (3, 3), padding='same', activation=keras.activations.tanh,
    #                               kernel_regularizer=keras.regularizers.l2(0.0001)))
    # model.add(keras.layers.MaxPool2D())
    # model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Conv2D(32, (3, 3), padding='same', activation=keras.activations.tanh,
                                  kernel_regularizer=keras.regularizers.l2(0.2)))
    model.add(keras.layers.MaxPool2D())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(5, activation=keras.activations.softmax,
                                 kernel_regularizer=keras.regularizers.l2(0.0001)))

    model.compile(optimizer=opt, loss=loss_func, metrics=keras.metrics.categorical_accuracy)

    logs = model.fit(x, y, validation_data=(val_x, val_y), epochs=epochs, batch_size=batch_size,
                     callbacks=[keras.callbacks.LearningRateScheduler(scheduler)])

    model.summary()



    return logs

def conv_aerial(train_dataset, val_dataset, opt, loss_func, epochs, batch_size):
    model = keras.models.Sequential()
    #model.add(keras.Input(shape=(*IMAGE_SIZE,3)))
    model.add(keras.layers.Conv2D(128, (3,3), padding="same", activation = 'relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    # model.add(keras.layers.Conv2D(64, (3,3), activation = 'relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    # model.add(keras.layers.BatchNormalization(axis=-1))
    model.add(keras.layers.MaxPool2D(2,2))
    model.add(keras.layers.Dropout(0.2))

    # model.add(keras.layers.Conv2D(64, (3,3), activation = 'relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    # model.add(keras.layers.BatchNormalization(axis=-1))
    model.add(keras.layers.Conv2D(64, (3,3), activation = 'relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    # model.add(keras.layers.BatchNormalization(axis=-1))
    model.add(keras.layers.MaxPool2D(2,2))
    # model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Conv2D(32, (3,3), activation = 'relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Conv2D(32, (3,3), activation = 'relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    # model.add(keras.layers.BatchNormalization(axis=-1))
    model.add(keras.layers.MaxPool2D(2,2))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation = 'relu', kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(5, activation = 'softmax', kernel_regularizer=keras.regularizers.l2(0.001)))

    model.compile(optimizer=opt, loss=loss_func, metrics=keras.metrics.categorical_accuracy)

    logs = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, batch_size=batch_size,
                    steps_per_epoch=1, validation_steps=1,
                     callbacks=[keras.callbacks.LearningRateScheduler(scheduler)])
    model.save("./output/convnet_aerial")
    model.summary()
    # make_predictions(model)
    return logs

# ## III. Etudes
# ### a. Initialisation des variables

# In[8]:


def decode_image(image):
    print(image)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def read_tfrecord(example, labeled):
    tfrecord_format = (
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "target": tf.io.FixedLenFeature([], tf.int64),
        }
        if labeled
        else {"image": tf.io.FixedLenFeature([], tf.string),}
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])
    if labeled:
        label = tf.cast(example["target"], tf.int32)
        return image, label
    return image

def load_dataset(filenames, labeled=True):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )
    # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE
    )
    return dataset

def get_dataset(filenames, labeled=True):
    dataset = load_dataset(filenames, labeled=labeled)
    dataset = dataset.shuffle(1024)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset

def make_predictions(model):
    preds = []
    origin = "../input/cassava-leaf-disease-classification"
    # sample_sub = pd.read_csv(origin + '/train.csv')
    sample_sub = pd.read_csv(origin + '/sample_submission.csv')

    for image in sample_sub.image_id:
        img = tf.keras.preprocessing.image.load_img(origin + '/test_images/' + image)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.preprocessing.image.smart_resize(img, (256, 256))
        img = np.expand_dims(img, 0)
        prediction = model.predict(img)
        preds.append(np.argmax(prediction))

    my_submission = pd.DataFrame({'image_id': sample_sub.image_id, 'label': preds})
    my_submission.to_csv('submission.csv', index=False)

# In[24]:
if __name__ == "__main__":
    VALIDATION_SPLIT = 0.33
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    IMAGE_SIZE = [128, 128]

    epochs = 2000
    batch_size = 100

    print("# GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    filenames=tf.io.gfile.glob('./input/augmented_tfrecords/data-part*.tfrec')
    random.shuffle(filenames)
    split = int(len(filenames) * VALIDATION_SPLIT)

    training_filenames = filenames[split:]
    validation_filenames = filenames[:split]

    #test_filenames=tf.io.gfile.glob('./input/test_tfrecords/*.tfrec')

    # print(validation_filenames)

    train_csv= pd.read_csv("./input/train.csv")
    train_directory="./input/resize_images"
    test_directory="./input/test_images"

    # img = cv2.imread('./input/cassava-leaf-disease-classification/train_images/1000015157.jpg')
    # plt.figure(figsize = (10,8))
    # plt.imshow(img)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    # print(train_csv.head())

    

    train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale= 1./255, validation_split=VALIDATION_SPLIT, shear_range= 0.2, 
        zoom_range= 0.2, horizontal_flip= True, rotation_range=20, vertical_flip= True,)
    train_csv["label"] = train_csv["label"].astype(str)
    train_dataset = train_datagen.flow_from_dataframe(train_csv, 
                                                    directory= train_directory, 
                                                    subset= 'training',
                                                    x_col= 'image_id',
                                                    y_col= 'label',
                                                    image_size=(128, 128),
                                                    batch_size=batch_size
                                                   )


    valid_dataset = train_datagen.flow_from_dataframe(train_csv,
                                                  directory= train_directory,
                                                  subset= 'validation',
                                                  x_col= 'image_id',
                                                  y_col= 'label',
                                                  image_size=(128, 128),
                                                  batch_size=batch_size
                                                  )

    #print(train_dataset.take(1))
    #print(train_dataset.shape())
    all_logs = []


    #raw_dataset = tf.data.TFRecordDataset(training_filenames[0])

    #for raw_record in raw_dataset.take(1):
    #    example = tf.train.Example()
    #    example.ParseFromString(raw_record.numpy())
    #    print(example)


    # In[10]:
    # filename_queue = tf.train.string_input_producer(['./input/train_images/0_6103.jpg']) #  list of files to read

    # reader = tf.WholeFileReader()
    # key, value = reader.read(filename_queue)

    # my_img = tf.image.decode_png(value)
    # img = Image.open("./input/train_images/6103.jpg")
    # file_content=tf.io.read_file("./input/augmented_images/0_6103.jpg")
    # image = tf.image.decode_jpeg(file_content, channels=3)
    # print(image)
    # image
    

    # Initialisation des datasets train et test à partir de tfrecords

    # train_dataset = get_dataset(training_filenames)
    # valid_dataset = get_dataset(validation_filenames)
    # test_dataset = get_dataset(TEST_FILENAMES, labeled=False)

    # train_csv= pd.read_csv("./input/train.csv")
    # train_directory="./input/resize_images/"

    # augmented_csv= pd.read_csv("./input/train_augmented.csv")
    # augmented_directory="./input/augmented_images/"
    # augmented_output="./input/augmented_tfrecords/"

    # train_csv["label"] = train_csv["label"].astype(int)
    # augmented_csv["label"] = augmented_csv["label"].astype(int)

    # images=[]
    # labels=[]
    # for row in train_csv.index:
    #     img = Image.open(train_directory + train_csv['image_id'][row])
    #     images.append(np.array(img))
    #     labels.append(train_csv['label'][row])

    # for row in augmented_csv.index:
    #     img = Image.open(augmented_directory + augmented_csv['image_id'][row])
    #     images.append(np.array(img))
    #     labels.append(augmented_csv['label'][row])
    
    # split = int(len(images) * VALIDATION_SPLIT)

    # x_train=np.array(images[split:])
    # y_train=np.array(images[:split])

    # x_test=np.array(labels[split:])
    # y_test=np.array(labels[:split])



    # x_train, y_train = next(iter(train_dataset))
    # x_test, y_test = next(iter(valid_dataset))

    print("DATASET LOAD")

    # print(np.shape(x_train.numpy()))

    # img,label=next(iter(train_dataset))
    # new_img=[]
    # new_label=[]
    # for row in train_dataset:
    #     img,lab=augment_dataset(row[0],row[1])
    #     new_img.append(img)
    #     new_label.append(lab)
    # print(len(new_img))

    # def show_batch(image_batch, label_batch):
    #     plt.figure(figsize=(20, 20))
    #     for n in range(25):
    #         ax = plt.subplot(5, 5, n + 1)
    #         plt.imshow(image_batch[n] / 255.0)
    #         plt.title("label : " + str(label_batch[n]))
    #         plt.axis("off")
    # show_batch(x_train.numpy(), y_train.numpy())

    # x_train = x_train / 255.0
    # x_test = x_test / 255.0

    # print(np.shape(x_train))

    # y_train = keras.utils.to_categorical(y_train, 5)
    # y_test = keras.utils.to_categorical(y_test, 5)


    # Train d'un modèle linéaire
    log = test_linear_model(train_dataset, valid_dataset, keras.optimizers.SGD(lr=0.01, momentum=0.95), keras.losses.categorical_crossentropy, epochs, batch_size)
    all_logs.append(log)

    # Train d'un modèle percpetron multicouche
    log = test_fully_connected(x_train, y_train, x_test, y_test, keras.optimizers.SGD(lr=1, momentum=0), keras.losses.categorical_crossentropy, epochs, batch_size)
    all_logs.append(log)


    # Train d'un modèle de convolution
    log = test_conv_net(x_train, y_train, x_test, y_test, keras.optimizers.SGD(lr=0.15, momentum=0.80), keras.losses.categorical_crossentropy, epochs, batch_size)
    all_logs.append(log)

    log = conv_aerial(train_dataset, valid_dataset, keras.optimizers.Adam(lr=0.000001), keras.losses.categorical_crossentropy, epochs, batch_size)
    all_logs.append(log)

    
    plot_logs(all_logs)




# ## X. Liens utiles
# ### a. Datasets additionnels


# - Dataset d'images augmentés par images superposés fondus (clic tu comprendras c'est bon) : 
# https://www.kaggle.com/frankmollard/2500-mixup-augmented-images
    
# - Datasets de la compétition 2019 et 2020 fusionnés (sans duplication) : 
# https://www.kaggle.com/tahsin/cassava-leaf-disease-merged


# ### b. Notebooks intéressant


# - Etude du dataset suivi, conseils pour mener une compétition kaggle : 
# https://www.kaggle.com/tanulsingh077/how-to-become-leaf-doctor-with-deep-learning
    
# - Data analysis du dataset cool : 
# https://www.kaggle.com/ihelon/cassava-leaf-disease-exploratory-data-analysis
