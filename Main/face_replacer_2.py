import cv2 as cv
import numpy as np
import glob
import os
import numpy
from PIL import Image
import keyboard
from Human_face_detector import human_face_detector
from face_replacer import anime_face_crop



cwd = os.getcwd()
anime_image_path = r'\output'
main_image_path = r'\input\input.jpg'
temp_path = r'\temp\\'
top_image = r'\mask.png'
output_path = r'\CartoonGAN-Test-Pytorch-Torch-master\CartoonGAN-Test-Pytorch-Torch-master\input\\'
predictor_path = r'shape_predictor_68_face_landmarks.dat'
face_cascade = cv.CascadeClassifier(cwd + r'\lbpcascade_animeface.xml')
anime_image_path = cwd + anime_image_path
main_image_path = cwd + main_image_path
temp_path = cwd + temp_path
top_image = cwd +  top_image
output_path = cwd + output_path


top_head_part = cv.imread(top_image)

img = cv.imread(main_image_path)

human_temp = human_face_detector(img, predictor_path)
human_temp.detect_human_face()
# human_temp.image_write(temp_path)
# human_temp.faces[0].showface()

# while True:
# 	if keyboard.is_pressed('q'):
# 		break

anime_temp = anime_face_crop(anime_image_path, face_cascade)
# human_temp.replace_main_image()

for i in range(len(anime_temp.anime_faces)):
	print("..... ",anime_temp.anime_faces[i].name, human_temp.faces[i].id)
	if anime_temp.anime_faces[i].name.find(str(human_temp.faces[i].id)) != -1:
		human_image = human_temp.faces[i]
		print("human temp faces h w", human_image.h, human_image.w)
		temp_img = human_image.face
		temp_img = cv.resize( temp_img, (256, 256))
		human_temp.faces[i].face = anime_temp.make_composit_image(human_image, anime_temp.anime_faces[i], top_head_part)



	
# 	# cv.imshow(anime_temp.create_face_mask(anime_temp.anime_faces[i]))
human_temp.replace_main_image(output_path, "out.jpg")
