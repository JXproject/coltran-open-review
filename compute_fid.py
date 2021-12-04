import numpy as np
import sys
import os
from glob import glob

from numpy.random import randint
from scipy.linalg import sqrtm
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from skimage.transform import resize
from skimage.io import imread

 
# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return np.asarray(images_list)
 

# calculate frechet inception distance
def calculate_fid(model, images1, images2):
	# calculate activations
	act1 = model.predict(images1)
	act2 = model.predict(images2)
	# calculate mean and covariance statistics
	mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
	mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if np.iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid

def predict(images1, images2):
	# images1 = np.array(images1).astype('float32')
	# images2 = np.array(images2).astype('float32')
	images1 = images1.astype('float32')
	images2 = images2.astype('float32')
	# resize images
	images1 = scale_images(images1, (299,299,3))
	images2 = scale_images(images2, (299,299,3))
	print('Scaled', images1.shape, images2.shape)
	# pre-process images
	images1 = preprocess_input(images1)
	images2 = preprocess_input(images2)
	# fid between images1 and images1
	fid = calculate_fid(model, images1, images1)
	print('FID (same): %.3f' % fid)
	# fid between images1 and images2
	fid = calculate_fid(model, images1, images2)
	print('FID (different): %.3f' % fid)
	return fid

if __name__ == "__main__":
	original = 'original/*'
	b1 = 'batch1/*'
	b2 = 'batch2/*'
	b3 = 'batch3/*'

	num_imgs = 10
	model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

	orig = [imread(glob(original)[idx]) for idx in range(0,num_imgs)]
	b1 = [imread(glob(b1)[idx]) for idx in range(0,num_imgs)]
	b2 = [imread(glob(b2)[idx]) for idx in range(0,num_imgs)]
	b3 = [imread(glob(b3)[idx]) for idx in range(0,num_imgs)]

	all_images = [original, b1, b2, b3]

	data = np.zeros((1, 4))

	f1 = predict(orig, b1)
	f2 = predict(orig, b2)
	f3 = predict(orig, b3)


	# print(sorted(glob(original)), sorted(glob(b1)))
	# for i, idx in enumerate(range(0,len(glob(original)))):
	# 	orig = imread(sorted(glob(original))[idx], plugin='matplotlib')
	# 	i1 = imread(sorted(glob(b1))[idx], plugin='matplotlib')
	# 	i2 = imread(sorted(glob(b2))[idx], plugin='matplotlib')
	# 	i3 = imread(sorted(glob(b3))[idx], plugin='matplotlib')
		
	f1 = predict(orig, i1)
	print('OUTPUT 1: %.3f' %f1)

	f2 = predict(orig, i2)
	print('OUTPUT 2: %.3f' %f2)

	f3 = predict(orig, i3)
	print('OUTPUT 3: %.3f' %f3)
	# 	# convert integer to floating point values

	data[0,0] = idx
	data[0,1] = f1
	data[0,2] = f2
	data[0,3] = f3

	np.save('fid_scores.npy', data)

		
