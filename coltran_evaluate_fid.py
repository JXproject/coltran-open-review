# %%
import numpy as np
import sys
import os
from glob import glob
import matplotlib.pyplot as plt

from numpy.random import randint
import scipy
from skimage.transform import resize
from skimage.io import imread
from torch.utils import data

import tensorflow.compat.v2 as tf
import jx_lib

from icecream import ic

# %%
LUT = {}
TITLE = []
def add_to_LUT(batch_name, batch_size, batch_folder, title):
    TITLE.append(title)
    LUT.update(
        {
            "{}-stage1".format(batch_name): {
                "ref": 'coltran/{}/imagenet/color_64'.format(batch_folder),
                "out": 'coltran/{}/stage_1'.format(batch_folder),
                "batch": batch_size,
            },
            "{}-stage2".format(batch_name): {
                "ref": 'coltran/{}/imagenet/color_64'.format(batch_folder),
                "out": 'coltran/{}/stage_2'.format(batch_folder),
                "batch": batch_size,
            },
            "{}-stage3".format(batch_name): {
                "ref": 'coltran/{}/imagenet/color'.format(batch_folder),
                "out": 'coltran/{}/stage_3'.format(batch_folder),
                "batch": batch_size,
            },
        }
    )

# [USER-INPUT] Please adjust the definition here:
add_to_LUT(batch_name="batch1", batch_size=10, batch_folder="result-batch1", title="batch1")
add_to_LUT(batch_name="batch2", batch_size=10, batch_folder="result-batch2", title="batch2")
add_to_LUT(batch_name="batch3", batch_size=10, batch_folder="result-batch3", title="batch3")
add_to_LUT(batch_name="ColTran", batch_size=100, batch_folder="result", title="ImageNet-2021")


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

def run_fid_score(TAG, INTERPOLATION, verbose=False):
    # def:
    PATH_ORIGINAL = LUT[TAG]["ref"]
    PATH_OUTPUT = LUT[TAG]["out"]
    BATCH_SIZE = LUT[TAG]["batch"]

    # %%
    model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3), weights='imagenet')
    dataset, num_files = load_and_preprocess_image(directory_1=PATH_ORIGINAL, directory_2=PATH_OUTPUT, batch_size=BATCH_SIZE)
    dataset_itr = iter(dataset)
    resize_and_rescale = tf.keras.Sequential([
        tf.keras.layers.Resizing(299, 299, interpolation=INTERPOLATION),
    ])

    if verbose:
        model.summary()


    # %%
    num_epochs = int(np.ceil(num_files / BATCH_SIZE))

    activation_1 = []
    activation_2 = []

    for i in range(num_epochs):
        if verbose:
            print("> [epoch] {}/{}".format(i, num_epochs))

        img1_batch, img2_batch = next(dataset_itr)
        image_1_p = tf.cast(img1_batch, tf.float32)
        image_2_p = tf.cast(img2_batch, tf.float32)
        
        image_1_p = resize_and_rescale(image_1_p)
        image_2_p = resize_and_rescale(image_2_p)
        
        image_1_p = tf.keras.applications.inception_v3.preprocess_input(image_1_p) # [0,255] -> [0,1]
        image_2_p = tf.keras.applications.inception_v3.preprocess_input(image_2_p)
        
        act_1 = model.predict(image_1_p)
        act_2 = model.predict(image_2_p)

        activation_1.extend(act_1)
        activation_2.extend(act_2)


    # %%
    # ic(np.shape(activation_1))
    # ic(np.shape(activation_2))

    # %%

    # calculate mean and covariance statistics
    mu1, sigma1 = np.mean(activation_1, axis=0), np.cov(activation_1, rowvar=False)
    mu2, sigma2 = np.mean(activation_2, axis=0), np.cov(activation_2, rowvar=False)

    # ic(mu1), ic(sigma1)
    # ic(mu2), ic(sigma2)
    # ic(np.max(mu1)), ic(np.max(sigma1))
    # ic(np.max(mu2)), ic(np.max(sigma2))
    # ic(np.min(mu1)), ic(np.min(sigma1))
    # ic(np.min(mu2)), ic(np.min(sigma2))

    # calculate sum squared difference between means
    epsilon=1e-6 # for numerical stability
    diff = (mu1 - mu2)
    # Product might be almost singular.
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print('[WARN] FID calculated produced singular product -> adding epsilon'
                        '(%g) to the diagonal of the covariances.', epsilon)
        offset = np.eye(sigma1.shape[0]) * epsilon
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component.
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('FID calculation lead to non-negligible imaginary component (%g)' % m)
        covmean = covmean.real

    # calculate score
    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean)


    # %%
    print("[{}:{}] FID Score: {}".format(TAG, INTERPOLATION, fid))
    return fid


# %% MAIN: 
def main():
    print(" \n ==== BEGIN ==== ")
    print("Computing FID for: ", TITLE)
    
    fid_s = {}
    for tag in LUT:
        for interpolation in ["bilinear", "nearest", "gaussian"]:
            if interpolation not in fid_s:
                fid_s[interpolation] = []
            
            fid_s[interpolation].append(run_fid_score(TAG=tag, INTERPOLATION=interpolation))
    
    # Plot FID Curve:
    DATA_DICT = {}
    def data_dict_update(name):
        n=len(DATA_DICT)
        DATA_DICT.update(
            {
                "{} - bilinear".format(name): {
                    "x": [1, 2, 3],
                    "y": fid_s["bilinear"][n:n+3],
                },
                "{} - nearest".format(name): {
                    "x": [1, 2, 3],
                    "y": fid_s["nearest"][n:n+3],
                },
                "{} - gaussian".format(name): {
                    "x": [1, 2, 3],
                    "y": fid_s["gaussian"][n:n+3],
                },
            }
        )
    for title in TITLE:
        data_dict_update(title)
    
    jx_lib.create_folder("output")
    jx_lib.output_plot(
        data_dict = DATA_DICT,
        Ylabel = "FID Score",
        Xlabel = "ColTran Stage",
        OUT_DIR = "output",
        tag = "FID Score At Different Stage",
    )
    print(" \n ==== END ==== ")


if __name__ == "__main__":
    main()


