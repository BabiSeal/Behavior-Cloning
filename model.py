#!/usr/bin/env python
"""
Steering angle prediction model
"""
import os
import argparse
import json
import cv2
import random
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

INPUT_IMG_HEIGHT = 160
INPUT_IMG_WIDTH = 320
RESIZED_IMG_HEIGHT = INPUT_IMG_HEIGHT // 2
RESIZED_IMG_WIDTH = INPUT_IMG_WIDTH // 2
INPUT_CHANNELS = 3
LEFT_STEERING_CORRECTION = 0.25
RIGHT_STEERING_CORRECTION = -0.25
CAMERA_IMG_PATH_INDEXES = [0, 1, 2]
STEERING = 3
STEER_CORRECTION = [0, LEFT_STEERING_CORRECTION, RIGHT_STEERING_CORRECTION]

'''
Load an image from path specified and
and do initial image resizing
'''
def pre_process_image(path, data_dir):
    img_path_name = path.strip()
    if (img_path_name.startswith('IMG')):
        fname = os.path.join(data_dir, img_path_name)
    else:
        (filepath, filename) = os.path.split(img_path_name) 
        fname = os.path.join(data_dir, 'IMG', filename)
    img = cv2.imread(fname)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (RESIZED_IMG_WIDTH, RESIZED_IMG_HEIGHT))
    return img

'''
Image Brightness Augmentation
Vivek Yadav Blog post on Using Augmentation to mimic Human Driving
https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0
'''
def randomize_brightness(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    new_brightness = 0.25 + np.random.uniform()
    img[:, :, 2] = img[:, :, 2] * new_brightness
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img


'''
Image Translation Augmentation
Vivek Yadav Blog post on Using Augmentation to mimic Human Driving
https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.yh93soib0
Shift the camera horizontally as well as vertically.
For horizontal shift make sure to adjust steering
'''
def randomize_translation(image, steer):
    rows, cols, _ = image.shape
    trans_range = 100
    num_pixels = 10
    val_pixels = 0.4
    trans_x = trans_range * np.random.uniform() - trans_range / 2
    steer_angle = steer + trans_x / trans_range * 2 * val_pixels
    trans_y = num_pixels * np.random.uniform() - num_pixels / 2
    trans_M = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    image = cv2.warpAffine(image, trans_M, (cols, rows))
    return image, steer_angle

'''
Augment image by adjusting brightness and added translation
Adjust steering appropriately
'''
def transform_image(image, steer):
    image, angle = randomize_translation(image, steer)
    image = randomize_brightness(image)
    steer_angle = angle

    # Horizontal Flip 50% of time randomly
    # Take are of turns, skip flipping for steering angles 0
    # Dont forget to fix the steering angle
    if random.random() >= .5 and abs(angle) > 0.1:
        image = cv2.flip(image, 1)
        steer_angle = -steer_angle

    return image, steer_angle

'''
Load a row from the training data array, select a
a camera image-path name and corresponding steer angle randomly
Load the corresponding image, steering angle. Apply pre-processing
and image transformation.
'''
def pre_process_data_train_row(df_row, data_dir):
    camera = np.random.randint(3)
    path = df_row[CAMERA_IMG_PATH_INDEXES[camera]]
    steer = df_row[STEERING] + STEER_CORRECTION[camera]
    img = pre_process_image(path, data_dir)
    img = randomize_brightness(img)
    img_transform, steer_angle = transform_image(img, steer)
    return img_transform, steer_angle

'''
Load a row from the validation data array, select a
a camera image-path name and corresponding steer angle randomly
Load the corresponding image, steering angle. Apply pre-processing
and image transformation.
'''
def pre_process_data_valid_row(df_row, data_dir):
    camera = np.random.randint(3)
    path = df_row[CAMERA_IMG_PATH_INDEXES[camera]]
    steer = df_row[STEERING] + STEER_CORRECTION[camera]
    img = pre_process_image(path, data_dir)
    return img, steer

'''
Generator function for Training Data
'''
def gen_batch_train_from_data_array(df_aray, batch_size, data_dir):

    batch_images = np.zeros((batch_size, RESIZED_IMG_HEIGHT, RESIZED_IMG_WIDTH, INPUT_CHANNELS))
    batch_angles = np.zeros(batch_size)
    while True:
        for i in range(batch_size):
            # Randomly get a sample from the input data
            idx = np.random.randint(len(df_aray))
            df_row = df_aray[idx, :]
            img_transform, steer_angle = pre_process_data_train_row(df_row, data_dir)
            batch_images[i] = img_transform
            batch_angles[i] = steer_angle

        yield batch_images, batch_angles

'''
Generator function for Validation Data
'''
def gen_batch_valid_from_data_array(df_aray, batch_size, data_dir):
    batch_images = np.zeros((batch_size, RESIZED_IMG_HEIGHT, RESIZED_IMG_WIDTH, INPUT_CHANNELS))
    batch_angles = np.zeros(batch_size)
    while True:
        for i in range(batch_size):
            # Randomly get a sample from the input data
            idx = np.random.randint(len(df_aray))
            df_row = df_aray[idx, :]
            img, steer = pre_process_data_valid_row(df_row, data_dir)
            batch_images[i] = img
            batch_angles[i] = steer
        yield batch_images, batch_angles

'''
Retuns the Nvidia based Model
Based on paper End-to-end Learning for Self-Driving Cars
http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
'''
def get_model():

    model = Sequential()

    # Normalization Layer
    image_shape = (RESIZED_IMG_HEIGHT, RESIZED_IMG_WIDTH, INPUT_CHANNELS)
    model.add(Lambda(lambda x: x / 127.5 - 0.5,
                     input_shape=(image_shape),
                     output_shape=(image_shape)))

    # First Convolution Layer with 1x1 filter for optimal color selection
    model.add(Convolution2D(3, 1, 1, border_mode="same"))

    # Convolution Layers with Dropout and Activation Layers
    # ELU Activation layers introduce non-linearity
    num_filters = [24, 36, 48, 64, 64]
    drop_out = 0.6
    filter_sizes = [(5, 5), (5, 5), (5, 5), (3, 3), (3, 3)]
    subsample_sizes = [(2, 2), (2, 2), (2, 2), (1, 1), (1, 1)]
    for conv_layer in range(len(num_filters)):
        model.add(Convolution2D(num_filters[conv_layer],
                                filter_sizes[conv_layer][0], filter_sizes[conv_layer][1],
                                border_mode='valid',
                                subsample=subsample_sizes[conv_layer],
                                activation='elu'))
        model.add(Dropout(drop_out))

    # Flatten the model
    model.add(Flatten())

    # Fully Connected Layers
    fc_neurons = [100, 50, 10]
    for fc_layer in range(len(fc_neurons)):
        model.add(Dense(fc_neurons[fc_layer], activation='elu'))
        model.add(Dropout(drop_out))

    # Final Output Layer
    model.add(Dense(1, activation='elu', name='out'))

    model.compile(optimizer=Adam(lr=0.001), loss="mse")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Steering angle model trainer')
    parser.add_argument('--batch', type=int, default=128, help='Batch size.')
    parser.add_argument('--datadir', default="./data", help='Input Data directory.')
    parser.add_argument('--epoch', type=int, default=7, help='Number of epochs.')
    parser.add_argument('--epochsize', type=int, default=50000, help='How many frames per epoch.')
    args = parser.parse_args()

    batch_size = args.batch

    data_dir = args.datadir
    csv_file_name = os.path.join(data_dir, "driving_log.csv")
    df = pd.read_csv(csv_file_name)

    # Convert the Pandas DataFrame to NumPy Arrays
    df_aray = df.values

    # Split into training and validation sets
    df_train, df_valid = train_test_split(df_aray, test_size = 0.2)  


    # Fit the model - Use Keras Generator
    model = get_model()
    model.fit_generator(
        gen_batch_train_from_data_array(df_train, batch_size, data_dir),
        samples_per_epoch=60000,
        nb_epoch=args.epoch,
        validation_data=gen_batch_valid_from_data_array(df_valid, batch_size, data_dir),
        nb_val_samples=5000)

    print("Saving model weights and configuration file.")
    model.save_weights("./model.h5", True)
    with open('./model.json', 'w') as outfile:
        json.dump(model.to_json(), outfile)

    #print(model.summary())





