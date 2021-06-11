import os
from simplejpeg import decode_jpeg
import numpy as np
from natsort import natsorted,ns
import shutil
import cv2
from os import path
from os.path import basename

def Merging():

    print("Merging train1 and train2 dataset...")
    path_train2 = "/content/bach-thes/notebooks/UNet/Train_images/train2/"

    path_src1 = "/content/bach-thes/notebooks/UNet/Train_images/train/"
    path_src2 = "/content/bach-thes/notebooks/UNet/Train_images/train_masks/"
    

    dst_img = "/content/bach-thes/notebooks/UNet/Train_images/train_merged/"
    dst_mask = "/content/bach-thes/notebooks/UNet/Train_images/train_merged_masks/"
    folders = []
    folders.append(dst_img)
    folders.append(dst_mask)

    if not os.path.exists(dst_img):
        os.makedirs(dst_img)
        first = True
    if not os.path.exists(dst_mask):
        os.makedirs(dst_mask)

    folders = []
    folders.append(dst_img)
    folders.append(dst_mask)
    
    img_list1 = natsorted(os.listdir(path_src1), alg=ns.IGNORECASE)
    mask_list2 = natsorted(os.listdir(path_src2), alg=ns.IGNORECASE)
    img_train2 = natsorted(os.listdir(path_train2), alg=ns.IGNORECASE)

    #ocisti merged foldere
    if not first:
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
	#print("Deleted contents of folder {}".format(folder))

    #kopiraj novo u merged, prvo train1 pa onda tek train2
    for file in img_list1:
        shutil.copy(file, dst_img)
    for file in img_train2:
        shutil.copy(file, dst_img)
    print("Merging complete.")
        #kopiranje slika
    print("Copying train1 masks...")
        #kopiranje maski
    for file in mask_list2: #kopit+raj train maske
        shutil.copy(file, dst_mask)
    print("Copying complete.")
