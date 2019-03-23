import cv2 as cv
import numpy as np
import glob
import os
import numpy
from PIL import Image
import keyboard
from Human_face_detector import human_face_detector


class anime_face_crop:
	#get the image without clothings

	def __init__(self, folder_path, face_cascade):
		self.anime_faces = []
		for f in glob.glob(os.path.join(folder_path, "*.jpg")):

			img = cv.imread(f)
			gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(gray, 1.3, 5)
			print("name ", f[57:], faces)
			temp = face_object(img, f[57:], faces[0])
			self.anime_faces.append(temp)

	def show_image(self, name ,img):
		cv.imshow(name, img)
		cv.waitKey(0)


	def make_transparent(self, face):
		img = face.face
		threshold = 20
		img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
		i, j, k = img.shape
		print(i, j, k)
		for row in range(i):
			for col in range(j):
				if img[row][col][0] <= threshold and img[row][col][1] <= threshold and img[row][col][2] <= threshold:
					img[row][col][0] = 0
					img[row][col][1] = 0
					img[row][col][2] = 0
					img[row][col][3] = 0
		# self.show_image('transparent', img)


	def make_composit_image(self, human_object, anime, mask_image):
		low_y = human_object.low_y
		# low_y = int(low_y * 256 / human_object.h)
		# anime.
		padding = max(20, low_y - human_object.h)
		print("padding ", low_y, human_object.h)
		human = human_object.face
		# cv.imshow("Huamn ", human)
		# cv.waitKey(0)
		
		# human = cv.resize( human_object.face, (256, 256))
		print(" human h w and shape",human_object.h, human_object.w, human_object.face.shape)
		anime = self.create_face_mask(anime, human_object.h, human_object.w)
		cv.imshow("anime ", anime)
		cv.waitKey(0)
		print(" anime h w",anime.shape[0], anime.shape[1])
		# mask_image = cv.cvtColor(mask_image, cv.COLOR_RGB2GRAY)
		# ret, mask_image = cv.threshold(mask_image, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
		# after_mask_surr = cv.bitwise_and(human, human, mask = mask_image)

		# main_mask_image = cv.bitwise_not(mask_image)
		# print(mask.shape)
		for row in range(padding, human_object.w):
			for col in range(human_object.h):
				if anime[row - padding][col][3] != 0:
					human[row][col][0] = anime[row - padding][col][0]
					human[row][col][1] = anime[row - padding][col][1] 
					human[row][col][2] = anime[row - padding][col][2]
					# anime[row][col][3] = 1
		# kernel = np.ones((3, 3), np.uint8)
		# closing = cv.morphologyEx(human, cv.MORPH_CLOSE, kernel, iterations = 2) 
		# after_mask_main = cv.bitwise_and(human, human, mask = main_mask_image)
		# after_mask_surr = cv.bitwise_and(human, human, mask = mask_image)
		# after_mask = after_mask_surr  + after_mask_main
		# cv.imshow("Transparent", human)
		# self.show_image("composite", human)
		return human





	def create_face_mask(self, face, height, width):
		point_thres = 1.5
		hw_thres = 0.75
		threshold_color = 220
		temp_face_object = face
		x = int(face.x * point_thres)
		y = int(face.y * point_thres)
		w = int(face.w * hw_thres)
		h = int(face.h * hw_thres)
		print(x, x+w , y, y+h)
		img = face.face #face.face[y:y+h, x:x+h]
		# print(img.shape)
		bg = np.array([[[0] * 3] * img.shape[0]] * img.shape[1], np.uint8)
		bg = np.full_like(bg, [threshold_color, threshold_color, threshold_color])
		temp_img = img - bg
		# self.show_image('Subtraction', temp_img)
		# making gray
		gray = cv.cvtColor(temp_img, cv.COLOR_BGR2GRAY)
		ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU) 
		# self.show_image('THRESH_OTSU', thresh)

		kernel = np.ones((3, 3), np.uint8)
		closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel, iterations = 1)

		# self.show_image('MorphologyEX', closing)

		fg = cv.dilate(closing, kernel, iterations = 1)
		dist_transform = cv.distanceTransform(closing, cv.DIST_L1, 0)

		# self.show_image("distanceTransform", dist_transform)

		ret, fg = cv.threshold(dist_transform, 0.05 * dist_transform.max(), 255, 0)
		fg = np.uint8(fg)
		fg = cv.bitwise_not(fg)

		# self.show_image('foreground', fg)
		
		after_mask = cv.bitwise_and(img, img, mask = fg)

		# self.show_image("Bitwise and", after_mask)

		after_mask =  cv.cvtColor(after_mask, cv.COLOR_BGR2BGRA)
		for row in range(y, y+h):
			for col in range(x, x+w):
				if img[row][col][0] >= threshold_color and img[row][col][1] >= threshold_color and img[row][col][2] >= threshold_color:
					img[row][col][0] = 0
					img[row][col][1] = 0
					img[row][col][2] = 0
				else:
					break
			for col in range (x+w, x, -1):
				if img[row][col][0] >= threshold_color and img[row][col][1] >= threshold_color and img[row][col][2] >= threshold_color:
					img[row][col][0] = 0
					img[row][col][1] = 0
					img[row][col][2] = 0
					
				else:
					break
		# cv.imshow("main_face_part", img)
		# cv.waitKey(0)
		for i in range(x, x+w):
			for j in range(y, y+h):
				after_mask[j][i][0] = img[j][i][0]
				after_mask[j][i][1] = img[j][i][1]
				after_mask[j][i][2] = img[j][i][2]

		
		
		for i in range(256):
			for j in range(y+h, 256):
				# print("Here i am")
				after_mask[j][i][0] = 0
				after_mask[j][i][1] = 0
				after_mask[j][i][2] = 0

		# cv.imshow("Lower_part_removal", fg)
		# cv.waitKey(0)

		img = after_mask
		threshold = 0
		img = cv.cvtColor(img, cv.COLOR_RGB2RGBA)
		i, j, k = img.shape
		print(i, j, k)
		for row in range(i):
			for col in range(j):
				if img[row][col][0] <= threshold and img[row][col][1] <= threshold and img[row][col][2] <= threshold:
					img[row][col][0] = 0
					img[row][col][1] = 0
					img[row][col][2] = 0
					img[row][col][3] = 0
		img = cv.resize(img, (height, width))
		# cv.imshow("Main_mask", img)
		# cv.waitKey(0)
		return img
		# new_img = make_transparent(img)


	
#create mask
#add image



	#save without cloth image
class face_object:
	def __init__(self, face_image, name, coordinate):
		self.face = face_image
		self.name = name
		x, y, h, w = coordinate
		self.x = x
		self.y = y
		self.h = h
		self.w = w


	def showface(self):
		cv.imshow("Output", self.face)
		cv.waitKey(0)


cwd = os.getcwd()
anime_image_path = r'\output'
main_image_path = r'\input\input.jpg'
temp_path = r'\temp\\'
top_image = r'\mask.png'

predictor_path = r'shape_predictor_68_face_landmarks.dat'
face_cascade = cv.CascadeClassifier(cwd + r'\lbpcascade_animeface.xml')
anime_image_path = cwd + anime_image_path
main_image_path = cwd + main_image_path
temp_path = cwd + temp_path
top_image = cwd +  top_image


top_head_part = cv.imread(top_image)

img = cv.imread(main_image_path)

human_temp = human_face_detector(img, predictor_path)
human_temp.detect_human_face()
human_temp.image_write(temp_path)
# human_temp.faces[0].showface()

# while True:
# 	if keyboard.is_pressed('q'):
# 		break

# anime_temp = anime_face_crop(anime_image_path, face_cascade)
# # human_temp.replace_main_image()

# for i in range(len(anime_temp.anime_faces)):
# 	print("..... ",anime_temp.anime_faces[i].name, human_temp.faces[i].id)
# 	if anime_temp.anime_faces[i].name.find(str(human_temp.faces[i].id)) != -1:
# 		human_image = human_temp.faces[i]
# 		print("human temp faces h w", human_image.h, human_image.w)
# 		temp_img = human_image.face
# 		temp_img = cv.resize( temp_img, (256, 256))
# 		human_temp.faces[i].face = anime_temp.make_composit_image(human_image, anime_temp.anime_faces[i], top_head_part)



	
# # 	# cv.imshow(anime_temp.create_face_mask(anime_temp.anime_faces[i]))
# human_temp.replace_main_image()
