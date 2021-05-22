import os
from natsort import natsorted,ns

img_dir = r"T:\\Dataset_vascular\\proba\\"
mask_dir = r"T:\\Dataset_vascular\\proba_masks\\"

img_list = natsorted(os.listdir(img_dir), alg=ns.IGNORECASE)
mask_list = natsorted(os.listdir(mask_dir), alg=ns.IGNORECASE)

for image in img_list:
    if '{}_mask_vasc_{}_{}'.format(image.split('_')[0],image.split('_')[1],image.split('_')[2]) not in mask_list:
        print('Going to remove %s' % image)
        os.remove(os.join.path(img_dir,'%s') % image)