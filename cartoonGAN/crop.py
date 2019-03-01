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
      width = int(256/im.shape[0]*im.shape[1]) # calculate goal width, assuming width>height
      im = cv2.resize(im, (width, 256)) # resize
      crop_index = int(im.shape[1]/2 - 256/2) # crop index so width is 256
      im = im[:, crop_index:crop_index+256] # crop
      cv2.imwrite(newfpath, im)

cropDataFromPath(6000, 'flickr_train', 'flickr_train256x256')
