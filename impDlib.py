

import sys
import os
import dlib
import glob
import cv2 as cv
import numpy as np
import imutils

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
 
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
 
    # return the list of (x, y)-coordinates
    return coords

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    padding = 100
    x = rect.left() - padding
    y = rect.top() - padding
    w = rect.right() - x + 2*padding
    h = rect.bottom() - y + 2*padding
 
    # return a tuple of (x, y, w, h)
    return (x , y , w , h)


print(dlib.__version__)

predictor_path = 'shape_predictor_68_face_landmarks.dat'   #sys.argv[1]
faces_folder_path =  'C:\\Users\\abjaw\\Documents\\GitHub\\202-project3\\image'  #sys.argv[2]
output_path = 'C:\\Users\\abjaw\\Documents\\GitHub\\202-project3\\output\\'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
#win = dlib.image_window()
i = 0
for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):

    print("Processing file: {}".format(f))
    img = cv.imread(f)

    # img = imutils.resize(img, width=500)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    dets = detector(gray, 1)

    print("Number of faces detected: {}".format(len(dets)))
    for k, rect in enumerate(dets):
        shape = predictor(img, rect)
        shape = shape_to_np(shape)
        # cv.imshow("some.jpg", img)
        x, y, w, h = rect_to_bb(rect)
        # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv.putText(img, "Face #{}".format(k + 1), (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # for (x, y) in shape:
        #     cv.circle(img, (x, y), 1, (0, 0, 255), -1)
        img = img[y:y+h, x:x+w]
        
        print("cropping...")
        # cv.imwrite('input.jpg', img)
        # print("Part 0: {}, Part 1: {} ...".format(shape.part(0),
                                                  # shape.part(1)))
        # Draw the face landmarks on the screen.
        #win.add_overlay(shape)
    # cv.imshow("Output", img)
    output = 'img' + str(i) + '.jpg'
    i += 1
    cv.imwrite(os.path.join(output_path , output),img)
    cv.waitKey(0)
    #win.add_overlay(dets)
    dlib.hit_enter_to_continue()