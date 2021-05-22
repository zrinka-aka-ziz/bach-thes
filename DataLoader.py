import glob, os, random
import numpy as np
from simplejpeg import decode_jpeg
import albumentations as A
from natsort import natsorted,ns

import torch
import torch.utils.data as d

from configure import Config
config = Config()

np.random.seed(config.seed_value) # cpu vars
torch.manual_seed(config.seed_value) # cpu  vars
torch.cuda.manual_seed(config.seed_value)
torch.cuda.manual_seed_all(config.seed_value) # gpu vars
torch.backends.cudnn.deterministic = True  #needed
torch.backends.cudnn.benchmark = False

light_aug = A.Compose([
	A.OneOf([
		#A.Flip(),
		#A.Rotate(limit=180, border_mode=0, value=0, mask_value=0),
		#A.ElasticTransform(border_mode=0, value=0),

		#A.GaussNoise(var_limit=(50,100), mean=0),
		#A.Downscale(scale_min=0.8, scale_max=0.99),

		#A.GaussianBlur(),
		#A.MotionBlur(),
		#A.MedianBlur(blur_limit=(3,5)),

		A.RandomBrightnessContrast(brightness_limit=[0.3,0.3], contrast_limit=[0.3,0.3]),
		#A.CLAHE(),


	]),
], p=1)

medium_aug = A.Compose([
	A.OneOf([
		A.Flip(),
		A.Rotate(limit=180, border_mode=0, value=0, mask_value=0),
		A.ElasticTransform(border_mode=0, value=0, p=1),
		]),
	A.OneOf([
		A.GaussNoise(var_limit=(50,100), mean=0),
		A.Downscale(scale_min=0.8, scale_max=0.99),
		]),
	A.OneOf([
		A.GaussianBlur(),
		A.MotionBlur(),
		A.MedianBlur(blur_limit=(3,5)),
		]),
	A.OneOf([
		A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
		A.CLAHE(),
	]),
], p=0.5)

heavy_aug = A.Compose([
	A.OneOf([
		A.Flip(),
		A.Rotate(limit=180, border_mode=0, value=0, mask_value=0),
		A.ElasticTransform(border_mode=0, value=0),
		]),
	A.OneOf([
		A.GaussNoise(var_limit=(50,100), mean=0),
		A.Downscale(scale_min=0.8, scale_max=0.99),
		]),
	A.OneOf([
		A.GaussianBlur(),
		A.MotionBlur(),
		A.MedianBlur(blur_limit=(3,5)),
		]),
	A.OneOf([
		A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
		A.CLAHE(),
	]),
], p=1)

res = A.Resize(config.imgsize[0], config.imgsize[1], interpolation=2, always_apply=True, p=1)

class ImageDataset(d.Dataset):
	def __init__(self, datapath):
		self.datapath = datapath
		self.path_img = self.datapath + '/'
		self.path_mask = self.datapath + '_masks/'

		self.imglist = natsorted(glob.glob(self.path_img + '*.jpeg'), alg=ns.IGNORECASE)
		self.masklist = natsorted(glob.glob(self.path_mask + '*.jpeg'), alg=ns.IGNORECASE)
		#if(center==True):
		#	self.img_mean = self.center_data(imglist)

	def __len__(self):
		return len(os.listdir(self.path_img))

	def __getitem__(self, idx):
		#print("IMGLIST SIZE: ", len(self.imglist))
		#print("MASKLIST SIZE: ", len(self.masklist))
		#print("CURRENT INDEX: ", idx)

		with open(os.path.join(self.path_img, self.imglist[idx]), 'rb') as f1:
			image = decode_jpeg(f1.read())[:,:,1]
		with open(os.path.join(self.path_mask, self.masklist[idx]), 'rb') as f2:
			mask = decode_jpeg(f2.read())[:,:,1]

		#if(center==True):
		#	image = image - self.img_mean

		if(config.extension=="model12"):
			augmented = light_aug(image=image, mask=mask)
			image = augmented['image']
			mask = augmented['mask']
		if(config.extension=="model13"):
			augmented = medium_aug(image=image, mask=mask)
			image = augmented['image']
			mask = augmented['mask']
		if(config.extension=="model14"):
			augmented = heavy_aug(image=image, mask=mask)
			image = augmented['image']
			mask = augmented['mask']

		#resized = res(image=image, mask=mask)
		#image = resized['image']
		#mask = resized['mask']
		sample = {'image': image, 'mask': mask}  

		sample['image'] = np.expand_dims(sample['image'], axis=0)
		sample['mask'] = np.expand_dims(sample['mask'], axis=0)
		sample['image'].astype(float)
		sample['mask'].astype(float)

		sample['image'] = sample['image']/255.0 #image being rescaled to contain values between 0 to 1 for BCE Loss
		if sample['mask'].max()==255:
		    sample['mask'] = sample['mask']/255.0

		#sample['image'] = (sample['image']/255)*2 -1 #image being rescaled to contain values between -1 to 1 for BCE Loss
		#sample['mask'] = (sample['mask']/255)*2 -1

		return sample

	def center_data(self, list):
		num_of_img = len(list)
		mean_img = np.zeros((config.imgsize[0], config.imgsize[1]), dtype=float64)
		for i, file in enumerate(list):
			img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
			mean_img += img / num_of_img
		return mean_img

	def normalize_data_std(self, list):
		num_of_img = len(list)
		std_img = np.zeros((config.imgsize[0], config.imgsize[1]), dtype=float64)
		for i, file in enumerate(list):
			img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
			img = cv2.resize(img, (config.imgsize[0], config.imgsize[1]), interpolation = cv2.INTER_LINEAR)
			std_img += (img - self.mean_img)^2
		return sqrt(std_img / num_of_img)