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
LUT = {
    "ColTran-stage1": {
        "ref": 'coltran/result/imagenet/color_64',
        "out": 'coltran/result/stage_1',
        "batch": 100,
    },
    "ColTran-stage2": {
        "ref": 'coltran/result/imagenet/color_64',
        "out": 'coltran/result/stage_2',
        "batch": 100,
    },
    "ColTran-stage3": {
        "ref": 'coltran/result/imagenet/color',
        "out": 'coltran/result/stage_3',
        "batch": 100,
    },
    "batch-stage1": {
        "ref": 'coltran/result-batch/imagenet/color_64',
        "out": 'coltran/result-batch/stage_1',
        "batch": 10,
    },
    "batch-stage2": {
        "ref": 'coltran/result-batch/imagenet/color_64',
        "out": 'coltran/result-batch/stage_2',
        "batch": 10,
    },
    "batch-stage3": {
        "ref": 'coltran/result-batch/imagenet/color',
        "out": 'coltran/result-batch/stage_3',
        "batch": 10,
    },
}


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
    fid_s = {}
    for tag in LUT:
        for interpolation in ["bilinear", "nearest", "gaussian"]:
            if interpolation not in fid_s:
                fid_s[interpolation] = []
            
            fid_s[interpolation].append(run_fid_score(TAG=tag, INTERPOLATION=interpolation))
    
    # Plot FID Curve:
    data_dict = {
        "ImageNet2012 - bilinear": {
            "x": [1, 2, 3],
            "y": fid_s["bilinear"][0:3],
        },
        "ImageNet2012 - nearest": {
            "x": [1, 2, 3],
            "y": fid_s["nearest"][0:3],
        },
        "ImageNet2012 - gaussian": {
            "x": [1, 2, 3],
            "y": fid_s["gaussian"][0:3],
        },
        "Custom - bilinear": {
            "x": [1, 2, 3],
            "y": fid_s["bilinear"][3:6],
        },
        "Custom - nearest": {
            "x": [1, 2, 3],
            "y": fid_s["nearest"][3:6],
        },
        "Custom - gaussian": {
            "x": [1, 2, 3],
            "y": fid_s["gaussian"][3:6],
        },
    }
    jx_lib.create_folder("output")
    jx_lib.output_path(
        data_dict = data_dict,
        Ylabel = "FID Score",
        Xlabel = "ColTran Stage",
        OUTPUT = "output",
        tag = "FID Score At Different Stage",
    )


if __name__ == "__main__":
    main()


