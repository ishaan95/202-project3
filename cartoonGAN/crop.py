import os
import cv2
import random

def cropDataFromPath(count, path, newpath):
  for (_, _, fnames) in os.walk(path):
    random.shuffle(fnames)
    for fname in fnames[:count]:
      fpath = path + "/" + fname
      newfpath = newpath + "/" + fname
      im = cv2.imread(fpath)
      if im.shape[1] > im.shape[0]:
        width = int(256/im.shape[0]*im.shape[1])
        im = cv2.resize(im, (width, 256)) # resize
        crop_index = int(im.shape[1]/2 - 256/2) # crop index so width is 256
        im = im[:, crop_index:crop_index+256] # crop
      else:
        height = int(256/im.shape[1]*im.shape[0])
        im = cv2.resize(im, (256, height)) # resize
        crop_index = int(im.shape[0]/2 - 256/2) # crop index so height is 256
        im = im[crop_index:crop_index+256, :] # crop
      cv2.imwrite(newfpath, im)

cropDataFromPath(6000, 'flickr_train', 'flickr_train256x256')
