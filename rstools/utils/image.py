import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()
import cv2
import numpy as np
from rstools.utils.tif import read_RSTif, strentch_img

def read_image(img_path,norm_mode=1,im_w=None,im_h=None,im_c=None,dtype=None):
    JPGext = ['png','jpg','jpeg','PNG','JPG','JPEG']
    TIFext = ['tif','tiff','TIF','TIFF']
    RAWext = ['raw', 'RAW']
    if img_path.endswith(tuple(JPGext)):
        img = cv2.imread(img_path)
    elif img_path.endswith(tuple(TIFext)):
        img = read_RSTif(img_path,norm=True,norm_mode=norm_mode)
    elif img_path.endswith(tuple(RAWext)):
        img = np.fromfile(img_path, dtype=dtype)
        img = img.reshape(im_w, im_h, im_c)
        img = strentch_img(img,mode=norm_mode)
    else:
        raise TypeError
    return img


def resize_image(img, scale=1.0):
    if (scale != 1.0):
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img


def blur_image(imgarr, mode='g'):
    if mode == 'g':
        imgarr = cv2.GaussianBlur(imgarr,(1,1),0)
    elif mode == 'b':
        imgarr = cv2.bilateralFilter(imgarr,1,1,1)
    elif mode == 'm':
        imgarr = cv2.medianBlur(imgarr,1)
    else:
        imgarr = cv2.blur(imgarr,(1,1))
    return imgarr
