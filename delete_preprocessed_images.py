import os, shutil
from os import path
from os.path import basename
from configure import Config
config = Config()

folders = []

# Quick N Dirty - non-configurable
folders.append("/content/bach-thes/notebooks/UNet/Train_images/train")
folders.append("/content/bach-thes/notebooks/UNet/Train_images/train_masks")
folders.append("/content/bach-thes/notebooks/UNet/Train_images/resampled_train")
folders.append("/content/bach-thes/notebooks/UNet/Train_images/resampled_train_masks")
folders.append("/content/bach-thes/notebooks/UNet/Train_images/extracted_train")
folders.append("/content/bach-thes/notebooks/UNet/Train_images/extracted_train_masks")

folders.append("/content/bach-thes/notebooks/UNet/Validation_images/validation")
folders.append("/content/bach-thes/notebooks/UNet/Validation_images/validation_masks")
folders.append("/content/bach-thes/notebooks/UNet/Validation_images/resampled_validation")
folders.append("/content/bach-thes/notebooks/UNet/Validation_images/resampled_validation_masks")
folders.append("/content/bach-thes/notebooks/UNet/Validation_images/extracted_validation")
folders.append("/content/bach-thes/notebooks/UNet/Validation_images/extracted_validation_masks")

folders.append("/content/bach-thes/notebooks/UNet/Test_images/test")
folders.append("/content/bach-thes/notebooks/UNet/Test_images/test_masks")
folders.append("/content/bach-thes/notebooks/UNet/Test_images/resampled_test")
folders.append("/content/bach-thes/notebooks/UNet/Test_images/resampled_test_masks")
folders.append("/content/bach-thes/notebooks/UNet/Test_images/extracted_test")
folders.append("/content/bach-thes/notebooks/UNet/Test_images/extracted_test_masks")

for folder in folders:
	for filename in os.listdir(folder):
		file_path = os.path.join(folder, filename)
		try:
			if os.path.isfile(file_path) or os.path.islink(file_path):
				os.unlink(file_path)
			elif os.path.isdir(file_path):
				shutil.rmtree(file_path)
		except Exception as e:
			print('Failed to delete %s. Reason: %s' % (file_path, e))
	print("Deleted contents of folder {}".format(folder))
