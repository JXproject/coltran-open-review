# %% INIT
import os

import matplotlib.pyplot as plt
import numpy as np

import tensorflow.compat.v2 as tf

from absl import logging

from coltran import datasets
from coltran.models import colorizer
from coltran.models import upsampler
from coltran.utils import base_utils
from coltran.utils import datasets_utils
from coltran.utils import train_utils

from ml_collections import config_flags

from coltran_configs import CONFIG_COLTRAN_CORE, CONFIG_COLOR_UPSAMPLER, CONFIG_SPATIAL_UPSAMPLER

from jx_lib import create_all_folders
from icecream import ic

# %% USER PARAMS:
# TAG = "-potato"
TAG = ""
# TAG = "-batch"

# %% Definitions:
from enum import Enum

class COLORTRAN_STEPS(Enum):
    INIT = 0
    COLORIZER = 1
    COPLOR_UPSAMPLER = 2
    SPATIAL_UPSAMPLER = 3

CONFIG = {
    COLORTRAN_STEPS.INIT: {
        "image_directory": [], # append here
        "mode": "recolorize",
        "batch_size": 20,
        "output_path": 'coltran/result{}/imagenet'.format(TAG),
    },
    COLORTRAN_STEPS.COLORIZER: {
        "image_directory": 'coltran/result{}/imagenet/gray'.format(TAG),
        "model_config": CONFIG_COLTRAN_CORE,
        "batch_size": 20,
        "mode": "colorize",
        "output_path": 'coltran/result{}/stage_1'.format(TAG),
        "pre-built_log_dir": 'coltran/coltran/colorizer',
    },
    COLORTRAN_STEPS.COPLOR_UPSAMPLER: {
        "image_directory": 'coltran/result{}/imagenet/gray'.format(TAG),
        "model_config": CONFIG_COLOR_UPSAMPLER,
        "batch_size": 5,
        "mode": "colorize",
        "output_path": 'coltran/result{}/stage_2'.format(TAG),
        "pre-built_log_dir": 'coltran/coltran/color_upsampler',
    },
    COLORTRAN_STEPS.SPATIAL_UPSAMPLER: {
        "image_directory": 'coltran/result{}/imagenet/gray'.format(TAG),
        "model_config": CONFIG_SPATIAL_UPSAMPLER,
        "mode": "colorize",
        "batch_size": 5,
        "output_path": 'coltran/result{}/stage_3'.format(TAG),
        "pre-built_log_dir": 'coltran/coltran/spatial_upsampler',
    },
}

### INIT [USER INPUT]:
RUN_STEPS = [
    # COLORTRAN_STEPS.INIT,
    # COLORTRAN_STEPS.COLORIZER,
    # COLORTRAN_STEPS.COPLOR_UPSAMPLER,
    COLORTRAN_STEPS.SPATIAL_UPSAMPLER,
]

# grab 5k samples: 100 x 50 = 5k:
if "potato" in TAG:
    MASTER_DIRECTORY = 'coltran/potatoes_images'
elif "batch" in TAG:
    MASTER_DIRECTORY = 'coltran/custom-batch'
else:
    MASTER_DIRECTORY = '/home/jx/tensorflow_datasets/imagenet2012/val/'
validation_dir = [os.path.join(MASTER_DIRECTORY,dir) for dir in tf.io.gfile.listdir(MASTER_DIRECTORY)]
CONFIG[COLORTRAN_STEPS.INIT]["image_directory"] = validation_dir[0:100]

print("====== Image Directories: \n", CONFIG[COLORTRAN_STEPS.INIT]["image_directory"] , "========== ========== ==========")

# %%

# %% Functions:
#### DEFINE HELPER FUNCTIONS ####
def gen_grayscale_dataset_from_images(img_dir, batch_size, mode='colorize', convert_to_gray=True):
    def load_and_preprocess_image(path, child_path):
        image_str = tf.io.read_file(path)
        num_channels = 1 if mode == 'colorize' else 3
        image = tf.image.decode_image(image_str, channels=num_channels)

        # Central crop to square and resize to 256x256.
        image = datasets.resize_to_square(image, resolution=256, train=False)

        # Resize to a low resolution image.
        image_64 = datasets_utils.change_resolution(image, res=64)
        if mode == 'recolorize' and convert_to_gray:
            image = tf.image.rgb_to_grayscale(image)
            image_64 = tf.image.rgb_to_grayscale(image_64)
        return image, image_64, child_path

    all_child_files = []
    all_files = []
    for dir in img_dir:
        child_files = tf.io.gfile.listdir(dir)
        files = [os.path.join(dir, file) for file in child_files]
        all_child_files.extend(child_files)
        all_files.extend(files)

    dataset = tf.data.Dataset.from_tensor_slices((all_files, all_child_files))
    dataset = dataset.map(load_and_preprocess_image)
    return dataset.batch(batch_size=batch_size), len(all_child_files)

def save_image(dataset_itr, model_step, num_files, color=False):
    # - create output folders:
    if color:
        path_64 = os.path.join(CONFIG[model_step]["output_path"], "color_64")
        path = os.path.join(CONFIG[model_step]["output_path"], "color")
    else:
        path_64 = os.path.join(CONFIG[model_step]["output_path"], "gray_64")
        path = os.path.join(CONFIG[model_step]["output_path"], "gray")
    
    create_all_folders(DIR=path_64)
    create_all_folders(DIR=path)

    # - init:
    batch_size = CONFIG[model_step]["batch_size"]
    num_epochs = int(np.ceil(num_files / batch_size))

    # - iterate through datasets
    for i in range(num_epochs):
        print("[{}] > Running @ ({}/{})".format(model_step, i, num_epochs))

        img, img_64, child_paths = next(dataset_itr)
        child_paths = child_paths.numpy()
        child_paths = [child_path.decode('utf-8') for child_path in child_paths]

        for g, g_64, child_path in zip(img, img_64, child_paths):
            write_path = os.path.join(path, child_path)
            g_img = g.numpy().astype(np.uint8)
            with tf.io.gfile.GFile(write_path, 'wb') as f:
                if color:
                    plt.imsave(f, g_img)
                else:
                    w,h,_ = np.shape(g_img)
                    img = np.zeros((w,h,3))
                    img[:,:,0] = g_img[:,:,0]
                    img[:,:,1] = g_img[:,:,0]
                    img[:,:,2] = g_img[:,:,0]
                    img = img.astype(np.uint8)
                    plt.imsave(f, img)
            
            write_path = os.path.join(path_64, child_path)
            g_64_img = g_64.numpy().astype(np.uint8)
            with tf.io.gfile.GFile(write_path, 'wb') as f:
                if color:
                    plt.imsave(f, g_64_img)
                else:
                    w,h,_ = np.shape(g_64_img)
                    img = np.zeros((w,h,3))
                    img[:,:,0] = g_64_img[:,:,0]
                    img[:,:,1] = g_64_img[:,:,0]
                    img[:,:,2] = g_64_img[:,:,0]
                    img = img.astype(np.uint8)
                    plt.imsave(f, img)
    
    print("[{}] > saved total images: {} at: {}".format(model_step, num_files, path))
    print("[{}] > saved total images: {} at: {}".format(model_step, num_files, path_64))

def build_model(
        model_step: COLORTRAN_STEPS,
    ):
    """Builds model."""
    config = CONFIG[model_step]["model_config"].get_config()
    print("[{}] > model config: {}".format(model_step, config.model.name))
    optimizer = train_utils.build_optimizer(config)

    zero_64 = tf.zeros((1, 64, 64, 3), dtype=tf.int32)
    zero_64_slice = tf.zeros((1, 64, 64, 1), dtype=tf.int32)
    zero = tf.zeros((1, 256, 256, 3), dtype=tf.int32)
    zero_slice = tf.zeros((1, 256, 256, 1), dtype=tf.int32)

    if model_step is COLORTRAN_STEPS.COLORIZER:
        model = colorizer.ColTranCore(config.model)
        model(zero_64, training=False)
    
    elif model_step is COLORTRAN_STEPS.COPLOR_UPSAMPLER:
        model = upsampler.ColorUpsampler(config.model)
        model(inputs=zero_64, inputs_slice=zero_64_slice, training=False)
    
    elif model_step is COLORTRAN_STEPS.SPATIAL_UPSAMPLER:
        model = upsampler.SpatialUpsampler(config.model)
        model(inputs=zero, inputs_slice=zero_slice, training=False)

    ema_vars = model.trainable_variables
    ema = train_utils.build_ema(config, ema_vars)
    return model, optimizer, ema

# %% MAIN: optimize
# MAIN:
dataset = None
dataset_itr = None
step = COLORTRAN_STEPS.INIT

store_dir, img_dir = None, None


def run_model(
        model_step: COLORTRAN_STEPS,
        prev_gen_data_dir = None
    ):
    print("================================ RUNNING MODEL [{}] ================================".format(model_step))
    # - init:
    batch_size = CONFIG[model_step]["batch_size"]
    
    # - fetch dataset:
    dataset, num_files = gen_grayscale_dataset_from_images(
        img_dir     =[CONFIG[model_step]["image_directory"]], 
        batch_size  =batch_size, 
        mode        =CONFIG[model_step]["mode"]
    )
    dataset_itr = iter(dataset)

    # - create output folders:
    create_all_folders(DIR=CONFIG[model_step]["output_path"])


    # - fetch additional dataset:
    if prev_gen_data_dir:
        gen_dataset = datasets.create_gen_dataset_from_images(prev_gen_data_dir)
        gen_dataset = gen_dataset.batch(batch_size)
        gen_dataset_iter = iter(gen_dataset)

    # - fetch model:
    model, optimizer, ema = build_model(model_step)
    checkpoints = train_utils.create_checkpoint(model, optimizer=optimizer,ema=ema)
    train_utils.restore(model, checkpoints, CONFIG[model_step]["pre-built_log_dir"], ema)
    num_steps_v = optimizer.iterations.numpy()

    num_epochs = int(np.ceil(num_files / batch_size))
    print('> Total Epochs: {} [{}/{}]'.format(num_epochs, num_files, batch_size))
    print('> Producing sample after %d training steps.', num_steps_v)

    # - iterate through datasets
    for i in range(num_epochs):
        print("[{}] > Running @ ({}/{})".format(model_step, i+1, num_epochs))

        gray, gray_64, child_paths = next(dataset_itr)

        if prev_gen_data_dir is not None:
            prev_gen = next(gen_dataset_iter)

        if model_step is COLORTRAN_STEPS.COLORIZER:
            out = model.sample(gray_64, mode='sample')
            samples = out['auto_sample']

        elif model_step is COLORTRAN_STEPS.COPLOR_UPSAMPLER:
            prev_gen = base_utils.convert_bits(prev_gen, n_bits_in=8, n_bits_out=3)
            out = model.sample(bit_cond=prev_gen, gray_cond=gray_64)
            samples = out['bit_up_argmax']

        elif model_step is COLORTRAN_STEPS.SPATIAL_UPSAMPLER:
            prev_gen = datasets_utils.change_resolution(prev_gen, 256)
            out = model.sample(gray_cond=gray, inputs=prev_gen, mode='argmax')
            samples = out['high_res_argmax']

        child_paths = child_paths.numpy()
        child_paths = [child_path.decode('utf-8') for child_path in child_paths]
        logging.info(child_paths)

        for sample, child_path in zip(samples, child_paths):
            write_path = os.path.join(CONFIG[model_step]["output_path"], child_path)
            logging.info(write_path)
            sample = sample.numpy().astype(np.uint8)
            logging.info(sample.shape)
            with tf.io.gfile.GFile(write_path, 'wb') as f:
                plt.imsave(f, sample)
    
    print("================================ MODEL END @ [{}] ================================".format(model_step))

# color dataset preview save
if COLORTRAN_STEPS.INIT in RUN_STEPS:
    dataset_color, num_files = gen_grayscale_dataset_from_images(
        img_dir     =CONFIG[COLORTRAN_STEPS.INIT]["image_directory"], 
        batch_size  =CONFIG[COLORTRAN_STEPS.INIT]["batch_size"], 
        mode        =CONFIG[COLORTRAN_STEPS.INIT]["mode"],
        convert_to_gray=False,
    )
    dataset_color_itr = iter(dataset_color)
    save_image(dataset_itr=dataset_color_itr, num_files=num_files, model_step=COLORTRAN_STEPS.INIT, color=True)

    # dataset generation
    dataset, num_files = gen_grayscale_dataset_from_images(
        img_dir     =CONFIG[COLORTRAN_STEPS.INIT]["image_directory"], 
        batch_size  =CONFIG[COLORTRAN_STEPS.INIT]["batch_size"], 
        mode        =CONFIG[COLORTRAN_STEPS.INIT]["mode"]
    )
    dataset_itr = iter(dataset)
    save_image(dataset_itr=dataset_itr, num_files=num_files, model_step=COLORTRAN_STEPS.INIT, color=False)

tf.keras.backend.clear_session()

### step 1:
if COLORTRAN_STEPS.COLORIZER in RUN_STEPS:
    run_model(
        model_step = COLORTRAN_STEPS.COLORIZER,
        prev_gen_data_dir = None
    )
    tf.keras.backend.clear_session()

### step 2:
if COLORTRAN_STEPS.COPLOR_UPSAMPLER in RUN_STEPS:
    run_model(
        model_step = COLORTRAN_STEPS.COPLOR_UPSAMPLER,
        prev_gen_data_dir = CONFIG[COLORTRAN_STEPS.COLORIZER]["output_path"]
    )
    tf.keras.backend.clear_session()

### step 3:
if COLORTRAN_STEPS.SPATIAL_UPSAMPLER in RUN_STEPS:
    run_model(
        model_step = COLORTRAN_STEPS.SPATIAL_UPSAMPLER,
        prev_gen_data_dir = CONFIG[COLORTRAN_STEPS.COPLOR_UPSAMPLER]["output_path"]
    )
    tf.keras.backend.clear_session()




# %%
