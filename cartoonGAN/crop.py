import os
import cv2
import random

def cropDataFromPath(path, newpath):
  for (_, _, fnames) in os.walk(path):
    random.shuffle(fnames)
    count = 0
    for fname in fnames:
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
      count +=1

cropDataFromPath('dataset/spirit_train', 'dataset/spirit_train256x256')
