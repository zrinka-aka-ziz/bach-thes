import torch
import torch.nn as nn
import os, glob
from torch.autograd import Variable
import numpy as np
from simplejpeg import decode_jpeg
import albumentations as A
from natsort import natsorted,ns
import cv2

from configure import Config
config = Config()

res1 = A.Resize(848,848,interpolation=2, always_apply=True, p=1)
res2 = A.Resize(1184,1184, interpolation=2, always_apply=True, p=1)

checkpoints = natsorted(os.listdir(config.checkpoints), alg=ns.IGNORECASE)[-2]
imglist = natsorted(os.listdir(config.test_orig+'/'), alg=ns.IGNORECASE)
masklist = natsorted(os.listdir(config.test_orig+'_masks/'), alg=ns.IGNORECASE)

if not os.path.exists(config.test_results):
    os.mkdir(config.test_results)

f11 = open(os.path.join(config.test_results, 'IoU.txt'), 'w')
f22 = open(os.path.join(config.test_results, 'DICE.txt'), 'w')

for i in range(0,len(imglist),1):
    with open(os.path.join(config.test_orig + '/' + imglist[i]), 'rb') as f1:
        image = decode_jpeg(f1.read())[:,:,1]

    with open(os.path.join(config.test_orig + '_masks/' + masklist[i]), 'rb') as f2:
        mask_or = decode_jpeg(f2.read())[:,:,1]
  
    ##resized = res1(image=image, mask=mask_or)
    #input_unet = resized['image']
    #mask = resized['mask']
    input_unet = image
    mask = mask_or

    #mask=mask_or
    #input_unet = image #- za full size
    input_unet = np.expand_dims(input_unet, axis=0)
    input_unet = np.expand_dims(input_unet, axis=0)
    mask = np.expand_dims(mask, axis=0)
    input_unet.astype(float)
    input_unet = input_unet / 255.0
    input_unet = torch.from_numpy(input_unet)
    input_unet = input_unet.type(torch.FloatTensor)
    input_unet = Variable(input_unet.cuda())

    model_unet = torch.load(os.path.join(config.checkpoints+checkpoints))
    model_unet.eval()
    model_unet.cuda()
    out_unet = model_unet(input_unet)
    out_unet = out_unet.cpu().data.numpy()
    out_unet = out_unet * 255
    out_unet = out_unet.transpose((2, 3, 0, 1))
    out_unet = out_unet[:,:,:,0]
    
    #resized = res2(image=image,mask=out_unet)
    #out_unet  = resized['mask']

    out_unet[out_unet <= 127] = 0
    out_unet[out_unet > 127] = 255

    out_unet.astype('uint8')
    cv2.imwrite(os.path.join(config.test_results + imglist[i]), out_unet)
    
    mask_or = mask[0,:,:]
    mask_pred = out_unet[:,:,0]
    intersection = np.logical_and(mask_or, mask_pred)
    union = np.logical_or(mask_or, mask_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    dice_score = 2 * iou_score / (1 + iou_score)
    f11.write("%s %s %s \n" % (imglist[i].rsplit(".",1)[0], checkpoints.rsplit(".",1)[0], str(iou_score)))
    f22.write("%s %s %s \n"% (imglist[i].rsplit(".",1)[0], checkpoints.rsplit(".",1)[0], str(dice_score)))
    print("Image " + imglist[i].rsplit(".",1)[0] + ", checkpoint " + checkpoints.rsplit(".",1)[0] + ", IoU score: " + str(round(iou_score, 6)))

f11.close()
f22.close()
