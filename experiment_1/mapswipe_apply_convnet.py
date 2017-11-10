#!/usr/bin/env python3

# coding: utf-8

# mapswipe_apply_convnet.py

# Given a trained convnet that has been saved to a file and a directory of input images
# evaluate each of these and report a score

# Copyright 2017  Robert Jones  jones@craic.com

# Project repo: https://github.com/craic/mapswipe_convnet

# Released under the terms of the MIT License

# This code was based on example 5.2 - Using convnets with small datasets from the book Deep Learning with Python
# by Francois Chollet - https://www.manning.com/books/deep-learning-with-python

# It hase been tuned to work with Bing Maps image tiles that are used in the MapSwipe mapping project
# http://mapswipe.org


import os, shutil
import argparse
import numpy as np

import keras
from keras.preprocessing import image
from keras import layers
from keras import models
from keras.models import load_model
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt



def main():
    parser = argparse.ArgumentParser(description="Apply a trained Convnet to a set of MapSwipe image tiles")
    parser.add_argument('--dataset', '-d', metavar='<path to input image tiles>', required=True,
                    help='Directory containing image tiles for evaluation by the CNN')
    parser.add_argument('--model', '-m', metavar='<model>', required=True,
                    help='HD5 format file containing the trained CNN model')
    args = parser.parse_args()

    image_size = 128
    #image_size = 256

    model = load_model(args.model)

    image_dir = args.dataset

    i = 0
    for img_file in os.listdir(image_dir):
        tile_id = img_file.replace(".jpg", "")
        img_path = os.path.join(image_dir, img_file)
        img = image.load_img(img_path, target_size=(image_size, image_size))
        img_tensor = image.img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        
        result = model.predict(img_tensor)[0][0]
        
        print("{}, {:5.2f}".format(tile_id, result))


main()
