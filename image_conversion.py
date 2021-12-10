#%% INIT for image conversion
import cv2
import tensorflow.compat.v2 as tf
import jx_lib
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
import os

MASTER_DIRECTORY = "coltran/potatoes_images"
validation_dir = jx_lib.get_files(DIR=MASTER_DIRECTORY, file_end=".jpg")

for itr, path in enumerate(validation_dir):
    ic(path)
    img = plt.imread(path)
    # img_tf = tf.io.decode_jpeg(path)
    ic(np.shape(img))

    scale_percent = 1024 / img.shape[1]  # percent of original size
    width = int(img.shape[1] * scale_percent )
    height = int(img.shape[0] * scale_percent )
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim)
    ic(np.shape(resized))

    resized= np.asarray(resized, dtype=np.uint8)


    print('converting image.....')
    img = np.zeros((height,width,3))
    img[:,:,0] = resized[:,:,0]
    img[:,:,1] = resized[:,:,1]
    img[:,:,2] = resized[:,:,2]
    img = img.astype(np.uint8)
    plt.imsave(os.path.join(MASTER_DIRECTORY,"A"+str(itr)+"-converted.jpg"), img)
    

# %%
