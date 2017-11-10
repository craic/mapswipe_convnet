#!/usr/bin/env python3

# coding: utf-8

# mapswipe_train_convnet.py

# Copyright 2017  Robert Jones  jones@craic.com

# Project repo: https://github.com/craic/mapswipe_convnet

# Released under the terms of the MIT License

# This code was based on example 5.2 - Using convnets with small datasets from the book Deep Learning with Python
# by Francois Chollet - https://www.manning.com/books/deep-learning-with-python

# It hase been tuned to work with Bing Maps image tiles that are used in the MapSwipe mapping project
# http://mapswipe.org

# The output goes into a run directory named with the current date and a serial e.g. 20171108_1
# In that directory there will be :
#  README
#  model_summary.txt  - a text summary of the model structure
#  history.csv        - CSV file of loss and accuracy by epoch
#  plot_<run_id>.png  - PNG image of a plot showing loss and accuracy
#  mapswipe_model_checkpoint.h5  - HDF file containing the best model
#  mapswipe_model_final.h5       - HDF file containing the final model - not necessarily the same

# The script is run like this:
#
# $ ./mapswipe_convnet_20171106.py --project ../my_project_data --n_epochs 50 --output ~/output --message "changed droput to 0.6"


import os, shutil
import datetime
import re
import argparse
import numpy as np

import keras
from keras.preprocessing import image
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt


# Plot the results and save to file
def plot_accuracy_and_loss(run_id, run_dir, history):

    train_acc       = history.history['acc']
    validation_acc  = history.history['val_acc']
    train_loss      = history.history['loss']
    validation_loss = history.history['val_loss']

    epochs = range(len(train_acc))

    plt.ylim(0.0, 1.0)

    major_y_ticks = [ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    plt.grid(True, 'major', 'y')
    plt.tick_params(axis='both', which='both',
                      bottom='on', top='off',
                      left='on', right='off',
                      labelbottom='on', labelleft='on')

    plt.gca().set_yticks(major_y_ticks)
    plt.plot(epochs, train_acc, 'r', label='Training accuracy')
    plt.plot(epochs, validation_acc, 'b', label='Validation accuracy')
    plt.plot(epochs, train_loss, 'm', label='Training loss')
    plt.plot(epochs, validation_loss, 'c', label='Validation loss')
    plt.title(run_id)
    plt.legend()

    filename = os.path.join(run_dir, "plot_{}.png".format(run_id))
    plt.savefig(filename)

# write this to a csv file...
def save_accuracy_and_loss_to_csv_file(run_dir, history):
    n_epochs = len(history.history['acc'])
    csv_file = os.path.join(run_dir, 'history.csv')
    with open(csv_file, 'wt') as f:
        f.write("Epoch,Train Acc,Val Acc,Train Loss, Val Loss\n")
        for i in range(n_epochs):
            f.write("{:d}, {:4.3f}, {:4.3f}, {:4.3f}, {:4.3f}\n".format( i+1,
                 history.history['acc'][i],
                 history.history['val_acc'][i],
                 history.history['loss'][i],
                 history.history['val_loss'][i]))


def setup_model(image_size):

    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    # SeparableConv2d did not seem to make a difference
    #model.add(layers.SeparableConv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # RJ - add one more if using 256x256 input images
#    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
#    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.Adam(lr=0.0001),
                  metrics=['acc'])

    return(model)



# List the subdirectories and get the run serial from a subdir with the current date
# Generate run id - 20171108_1, 20171108_2, etc
def generate_run_id(output_dir):

    date = datetime.datetime.now().strftime("%Y%m%d")
    pattern = re.compile(date)
    run_serial = 0
    sub_dirs = os.listdir(output_dir)
    for subdir in os.listdir(output_dir):
        subdir_strings = subdir.split("_")
        if subdir_strings[0] == date:
            serial = int(subdir_strings[1])
            if serial > run_serial:
                run_serial = serial

    run_serial += 1
    run_id = "{}_{}".format(date, str(run_serial))

    run_dir = os.path.join(output_dir, run_id)
    os.makedirs(run_dir)

    return run_id, run_dir


def output_model_summary_to_file(run_dir, model):
    model_summary_file = os.path.join(run_dir, 'model_summary.txt')
    with open(model_summary_file, 'wt') as f:
        # this hack necessary to get the summary written to file
        model.summary(print_fn=lambda x: f.write(x + '\n'))


# Write the run_id and user message into a README file
def write_readme_file(run_dir, run_id, message, parameters):
    readme_file = os.path.join(run_dir, 'README')
    with open(readme_file, 'wt') as f:
        f.write("Run {}\n\n{}\n\n".format(run_id, message))
        for key in sorted(parameters):
            f.write("{}: {}\n".format(key, parameters[key]))

def main():
    parser = argparse.ArgumentParser(description="Train a Convnet to distinguish between positive and negative MapSwipe image tiles")
    parser.add_argument('--project', '-p', metavar='<project_dir>', required=True,
                    help='Directory containing image tiles in train, validation and test subdirectories')
    parser.add_argument('--output', '-o', metavar='<output_dir>', required=True,
                    help='Output Directory')
    parser.add_argument('--n_epochs', '-n', metavar='<n_epochs>', type=int, required=True,
                    help='Number of epochs to run the training - e.g. 50')
    parser.add_argument('--message', '-m', metavar='<message>',
                        help='Brief Message to write to a README file', default="")

    args = parser.parse_args()

    project_dir = args.project
    output_dir  = args.output
    n_epochs    = int(args.n_epochs)
    message     = args.message


    # Bing Maps image tiles are 256x256 but reducing to 128x128 does not make a big difference

    image_size = 128

    parameters = {}
    parameters['keras_version'] = keras.__version__
    parameters['n_epochs'] = n_epochs
    parameters['image_size'] = image_size


    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_id, run_dir = generate_run_id(output_dir)

    print("This is Run {}\n".format(run_id))

    write_readme_file(run_dir, run_id, message, parameters)

    train_dir      = os.path.join(project_dir, 'train')
    validation_dir = os.path.join(project_dir, 'validation')
    test_dir       = os.path.join(project_dir, 'test')

    model_checkpoint_hdf_file = os.path.join(run_dir, "mapswipe_model_checkpoint.h5")
    model_final_hdf_file      = os.path.join(run_dir, "mapswipe_model_final.h5")


    model = setup_model(image_size)

    output_model_summary_to_file(run_dir, model)


    # Use data augmentation to increase the effective size of the training dataset
    # Not using shear as we should not find that in real images

    train_datagen = ImageDataGenerator(
        rescale=1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    #    rotation_range=40,


    validation_datagen = ImageDataGenerator(rescale=1./255)

    # batch size 64 gives slight improvement over 32 but takes longer

    batch_size = 32

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='binary')



    callbacks_list = [
#        keras.callbacks.ReduceLROnPlateau(
#            # This callback will monitor the validation loss of the model
#            monitor='val_loss',
#            # It will divide the learning by 10 when it gets triggered
#            factor=0.5,
#            #        factor=0.1,
#            # It will get triggered after the validation loss has stopped improving for at least 10 epochs
#            patience=10,
#        ),
        # This callback will save the current weights after every epoch if the validation loss has improved
        keras.callbacks.ModelCheckpoint(
            filepath=model_checkpoint_hdf_file,  # Path to the destination model file
            monitor='val_loss',
            save_best_only=True,
        )
    ]


    history = model.fit_generator(
        train_generator,
        callbacks=callbacks_list,
        steps_per_epoch=100,
        epochs=n_epochs,
        validation_data=validation_generator,
        validation_steps=50)

    # Save our model after training and validation

    model.save(model_final_hdf_file)

    save_accuracy_and_loss_to_csv_file(run_dir, history)

    plot_accuracy_and_loss(run_id, run_dir, history)


main()
