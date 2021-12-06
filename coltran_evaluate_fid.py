# %%
import numpy as np
import sys
import os
from glob import glob
import matplotlib.pyplot as plt

from numpy.random import randint
from scipy.linalg import sqrtm
from skimage.transform import resize
from skimage.io import imread

import tensorflow.compat.v2 as tf
import jx_lib

from icecream import ic

# %%
LUT = {
    "ColTran": {
        "ref": 'coltran/result/imagenet/color',
        "out": 'coltran/result/stage_3',
        "batch": 100,
    },
    "batch": {
        "ref": 'coltran/result-batch/imagenet/color',
        "out": 'coltran/result-batch/stage_3',
        "batch": 1,
    },
}

# [USER] selection:
TAG = "batch"

# def:
PATH_ORIGINAL = LUT[TAG]["ref"]
PATH_OUTPUT = LUT[TAG]["out"]
BATCH_SIZE = LUT[TAG]["batch"]

# %%
def load_and_preprocess_image(directory_1, directory_2, batch_size=20):
    def load_and_preprocess_image_map(path_1, path_2):
        image_str_1 = tf.io.read_file(path_1)
        image_str_2 = tf.io.read_file(path_2)

        image_1 = tf.image.decode_image(image_str_1, channels=3)
        image_2 = tf.image.decode_image(image_str_2, channels=3)
        
        return image_1, image_2

    child_files = tf.io.gfile.listdir(directory_1)
    pair_files = ([os.path.join(directory_1, file) for file in child_files], [os.path.join(directory_2, file) for file in child_files])
    dataset = tf.data.Dataset.from_tensor_slices(pair_files)
    dataset = dataset.map(load_and_preprocess_image_map)
    
    return dataset.batch(batch_size=batch_size), len(child_files)

# %%
model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3), weights='imagenet')
dataset, num_files = load_and_preprocess_image(directory_1=PATH_ORIGINAL, directory_2=PATH_OUTPUT, batch_size=BATCH_SIZE)
dataset_itr = iter(dataset)
resize_and_rescale = tf.keras.Sequential([
    tf.keras.layers.Resizing(299, 299),
    tf.keras.layers.Rescaling(1./255)
])

model.summary()


# %%
num_epochs = int(np.ceil(num_files / BATCH_SIZE))

activation_1 = []
activation_2 = []

for i in range(num_epochs):
    print("> [epoch] {}/{}".format(i, num_epochs))
    img1_batch, img2_batch = next(dataset_itr)
    ic(np.max(img1_batch))
    image_1_p = tf.cast(img1_batch, tf.float32)
    image_2_p = tf.cast(img2_batch, tf.float32)
    
    ic(np.max(img1_batch))
    image_1_p = tf.keras.applications.mobilenet.preprocess_input(image_1_p)
    image_2_p = tf.keras.applications.mobilenet.preprocess_input(image_2_p)
    
    ic(np.max(img1_batch))
    img1_batch = resize_and_rescale(image_1_p)
    img2_batch = resize_and_rescale(image_2_p)

    ic(np.max(img1_batch))
    act_1 = model.predict(img1_batch)
    act_2 = model.predict(img2_batch)

    activation_1.extend(act_1)
    activation_2.extend(act_2)


# %%
ic(np.shape(activation_1))
ic(np.shape(activation_2))

# %%

# calculate mean and covariance statistics
mu1, sigma1 = np.mean(activation_1, axis=0), np.cov(activation_1, rowvar=False)
mu2, sigma2 = np.mean(activation_2, axis=0), np.cov(activation_2, rowvar=False)

ic(mu1), ic(sigma1)
ic(mu2), ic(sigma2)

# calculate sum squared difference between means
ssdiff = np.sum((mu1 - mu2)**2.0)
# calculate sqrt of product between cov
covmean = sqrtm(sigma1.dot(sigma2))

if np.iscomplexobj(covmean):
    covmean = covmean.real

# calculate score
fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)


# %%
ic(fid)


