import os
from natsort import natsorted, ns
import shutil
from preprocess_config import PreprocessConfig
config = PreprocessConfig()



def execute(ext1, ext2):
	#kod gdje uzima svaku petu sliku (no varijabla) iz originalnog dataseta i kopira ju u novi folder
	# svaku petu sliku sam uzimala da bi se smanjio broj medusobno slicnih slika

	no = config.resample_rate

	'''
	img_dir = r"T:\\Dataset_vascular\\"+ext1+r"_images_full\\" + ext2 + "\\"
	mask_dir = r"T:\\Dataset_vascular\\"+ext1+r"_images_full\\" + ext2 + "_masks\\"
	dst_img = r"T:\\Dataset_vascular\\"+ext1+r"_images_full_sampled\\"+ext2+"\\"
	dst_mask = r"T:\\Dataset_vascular\\"+ext1+r"_images_full_sampled\\"+ext2+"_masks\\"
	'''

	path = config.path_short + ext1 + "_images"
	img_dir = path + "/extracted_" + ext2 + "/"
	mask_dir = path + "/extracted_" + ext2 + "_masks/"
	dst_img = path + "/resampled_" + ext2 + "/"
	dst_mask = path + "/resampled_" + ext2 + "_masks/"

	if not os.path.exists(dst_img):
		os.makedirs(dst_img)
	if not os.path.exists(dst_mask):
		os.makedirs(dst_mask)

	img_list = natsorted(os.listdir(img_dir), alg=ns.IGNORECASE)
	mask_list = natsorted(os.listdir(mask_dir), alg=ns.IGNORECASE)

	print("Resampling...")
	idx = range(0,len(mask_list),no)
	for i in idx:
		shutil.copy(mask_dir + mask_list[i], dst_mask + mask_list[i])
		shutil.copy(img_dir + img_list[i], dst_img + img_list[i])
	print("Resampling complete.")
