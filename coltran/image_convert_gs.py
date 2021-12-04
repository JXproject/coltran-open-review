import cv2
import os

path="./pic_gt/"
output_path='./trial_images/'
name='pic'

file_list=os.listdir(path)

for it,p in enumerate(file_list):
	pic_path=path+p
	output_dir=output_path+name+str(it)+".jpg"
	img = cv2.imread(pic_path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cv2.imwrite(output_dir, gray)
