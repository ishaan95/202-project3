# Haar Cascade Anime Face Detection
import cv2
import numpy as np
import glob
import os
import numpy
from PIL import Image


# loading in the cascades.
face_cascade = cv2.CascadeClassifier('C:\\Users\\abjaw\\Documents\\GitHub\\202-project3\\lbpcascade_animeface.xml')
faces_folder_path =  'C:\\Users\\abjaw\\Documents\\GitHub\\202-project3\\output'
output_path = 'C:\\Users\\abjaw\\Documents\\GitHub\\202-project3\\crop\\'
# video_link = ''

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    # print(imageA.shape, imageB.shape)
    if imageA.shape != imageB.shape:
        return 100
    else:
        err = np.sum((imageA.astype("uint8") - imageB.astype("uint8")) ** 2)
        err /= float(imageA.shape[0] * imageA.shape[1])
        
        # return the MSE, the lower the error, the more "similar"
        # the two images are
        return err


def get_from_video(filename, output_path):
    cap = cv2.VideoCapture(filename)
    i = 0
    # for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    # currentFrame = 0
    img_list = np.zeros((0, 0, 0))
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # print("Processing.... ", ret, frame)

        img = frame #cv2.imread(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # print("face detected... " , len(faces))
        for (x, y, w, h) in faces:
            # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            temp_image = img[y:y+h, x:x+w]
            print(mse(img_list, temp_image))
            if mse(img_list, temp_image) >= 100:
                print(mse(img_list, temp_image))
                img_list = temp_image
                # print(len(img_list))
                output = 'imgcrop' + str(i) + '.jpg'
                i += 1
                cv2.imwrite(os.path.join(output_path , output),temp_image)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

    cap.release()
    cv2.destroyAllWindows()

def get_from_folder(foldername, output_path):
    i = 0
    for f in glob.glob(os.path.join(foldername, "*.jpg")):
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        print("face detected... " , len(faces), f)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            output = 'img_from_folder' + str(i) + '.jpg'
            i += 1
            cv2.imwrite(os.path.join(output_path , output),img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break


# get_from_folder(faces_folder_path, output_path)

def change_background(Human_foldername, anime_folder, output_path):
    i = 0
    for f in glob.glob(os.path.join(foldername, "*.jpg")):
        img = cv2.imread(f)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # print(img.shape)
        img = cv2.resize( img, (256, 256))
        # print("new shape ", img.shape)
        output = 'img_from_folder' + str(i) + '.jpg'
        i += 1
        cv2.imwrite(os.path.join(output_path , output),img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break


def overlap_image(actual, transparent, dest_folder):
    # Read the images
    print(actual.shape, transparent.shape)
    foreground = transparent
    background = actual
    
    for row in range(256):
        for col in range(256):
            if transparent[row][col][3] == 0:
                transparent[row][col][0] = actual[row][col][0]  
                transparent[row][col][1] = actual[row][col][1]
                transparent[row][col][2] = actual[row][col][2]
                transparent[row][col][3] = 1
                # print("Inside here")

    outImage = cv2.add(foreground, transparent)
     
    # Display image
    cv2.imshow("outImg", foreground)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def make_transparent(sourceImage):
    threshold = 230
    img = cv2.imread(sourceImage)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
    i, j, k = img.shape
    print(i, j, k)
    for row in range(i):
        for col in range(j):
            # print(img[row][col][0], img[row][col][1], img[row][col][2])
            if img[row][col][0] >= threshold and img[row][col][1] >= threshold and img[row][col][2] >= threshold:
                img[row][col][0] = 0
                img[row][col][1] = 0
                img[row][col][2] = 0
                img[row][col][3] = 0
            else:
                break

        for col in range (j - 1, 0, -1):
            if img[row][col][0] >= threshold and img[row][col][1] >= threshold and img[row][col][2] >= threshold:
                img[row][col][0] = 0
                img[row][col][1] = 0
                img[row][col][2] = 0
                img[row][col][3] = 0
            else:
                break
        
    return img    



anime_source_folder = r'C:\Users\abjaw\Documents\GitHub\202-project3\output'
human_source_folder = r'C:\Users\abjaw\Documents\GitHub\202-project3\image'

dest_folder = r'C:\Users\abjaw\Documents\GitHub\202-project3\crop'
human_source_image = r'C:\Users\abjaw\Documents\GitHub\202-project3\image\img_from_folder4.jpg'
source_image = r'C:\Users\abjaw\Documents\GitHub\202-project3\output\img_from_folder4.jpg'
# change_background(human_source_folder, dest_folder)

# cv2.imshow('res',make_transparent(source_image))
# cv2.waitKey(0)
# cv2.destroyAllWindows()
overlap_image(cv2.imread(human_source_image), make_transparent(source_image), dest_folder)