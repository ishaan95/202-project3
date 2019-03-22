import os
import cv2
import numpy as np

def fuzzDataFromPath(path, newpath):
  for (_, _, fnames) in os.walk(path):
    for fname in fnames:
      fpath = path + "/" + fname
      newfpath = newpath + "/" + fname
      im = cv2.imread(fpath)
      kernel = np.ones((7, 7), np.uint8)
      mask = cv2.Canny(im, 100, 200)
      mask = cv2.dilate(mask, kernel, iterations=1)
      mask_background = cv2.bitwise_not(mask)
      blur = cv2.GaussianBlur(im, (3, 3), 0)
      blur = cv2.GaussianBlur(blur, (5, 5), 0)
      blur = cv2.GaussianBlur(blur, (7, 7), 0)
      im_foreground = cv2.bitwise_and(blur, blur, mask=mask)
      im_background = cv2.bitwise_and(im, im, mask=mask_background)
      im2 = cv2.bitwise_or(im_foreground, im_background)
      cv2.imwrite(newfpath, im2)

fuzzDataFromPath('dataset/spirit_train256x256', 'dataset/spirit_fuzzy_train256x256')
