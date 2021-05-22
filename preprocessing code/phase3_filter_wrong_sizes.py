import os
from simplejpeg import decode_jpeg
import numpy as np
from natsort import natsorted,ns
import cv2
from preprocess_config import PreprocessConfig
config = PreprocessConfig()

#kod za provjeru velicine patcheva

def execute(ext1, ext2):
	patch_size = config.patch_size

	'''
	mask_dir = r"T:\\Dataset_vascular\\"+ext1+r"_images_patch_sampled_period\\"+ext2+"_masks\\"
	img_dir = r"T:\\Dataset_vascular\\"+ext1+r"_images_patch_sampled_period\\"+ext2+"\\"
	'''
	path = config.path_short + ext1 + "_images"
	img_dir = path + "/resampled_" + ext2 + "/"
	mask_dir = path + "/resampled_" + ext2 + "_masks/"


	img_list = natsorted(os.listdir(img_dir), alg=ns.IGNORECASE)
	mask_list = natsorted(os.listdir(mask_dir), alg=ns.IGNORECASE)

	print("\nChecking file sizes...")
	for i in range(0,len(img_list)):
		with open(os.path.join(img_dir, img_list[i]), 'rb') as f1:
			image = decode_jpeg(f1.read())[:, :, 1]
			#print("IMAGE ", img_list[i]," SIZE: ", image.shape[0], "x", image.shape[1]) # !!

		if image.shape[0]!=patch_size or image.shape[1]!=patch_size:
			print("Removing " ,img_list[i])
			os.remove(os.path.join(img_dir, img_list[i])) #

	#evtl spojiti u jednu petlju radi brzine

	for i in range(0,len(mask_list)):
		with open(os.path.join(mask_dir, mask_list[i]), 'rb') as f1:
			image = decode_jpeg(f1.read())[:, :, 1]
			#print("IMAGE ", mask_list[i]," SIZE: ", image.shape[0], "x", image.shape[1]) # !!

		if image.shape[0]!=patch_size or image.shape[1]!=patch_size:
			print("Removing " ,mask_list[i])
			os.remove(os.path.join(mask_dir, mask_list[i]))
	print("Checking file sizes complete.")
