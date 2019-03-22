import sys
import os
import dlib
import glob
import cv2 as cv
import numpy as np
import imutils

padding_threshold_h = 1.5
padding_threshold_w = 0.7
class human_face_detector:
	def __init__(self, input_image, predictor_path):
		print("Initializing...")
		self.input_image = input_image #cv.cvtColor(input_image, cv.COLOR_BGR2GRAY) 
		self.predictor_path = predictor_path
		self.faces = []
		


	def replace_main_image(self):
		cv.imshow("mainImage", self.input_image)
		cv.waitKey(0)
		for i in range(len(self.faces)):
			x = self.faces[i].x
			y = self.faces[i].y
			h = self.faces[i].face.shape[0]
			w = self.faces[i].face.shape[1]
			print(" H/ W ",h, w)
			temp_img = cv.resize(self.faces[i].face, (h, w))
			print(temp_img.shape[0], temp_img.shape[1])
			
			# print(" Shape temp " ,temp_img.shape)
			# self.input_image[y: y+h, x:x+w] = temp_img[0:h, 0:w]
			# cv.imshow("temp image ", temp_img)
			# cv.waitKey(0)
			# cv.imshow("main image", self.input_image[y: y+h, x:x+w])
			# cv.waitKey(0)
		# cv.imshow("mainImage", self.input_image)
		# cv.waitKey(0)

	def rect_to_bb(self, rect):
	    padding_h = int((rect.bottom() - rect.top()) * padding_threshold_h)
	    padding_w = int((rect.right() - rect.left()) * padding_threshold_w)
	    x = int(rect.left() - 0.5 * padding_w)
	    y = max(0, int(rect.top() - 0.5 * padding_h))
	    w =  (rect.right() - rect.left()) + padding_w
	    h = (rect.bottom() - rect.top()) + padding_h
	    # print("Padding val h w..left..right.. ",padding_h, padding_w, rect.left(), rect.right())
	    # x = max(0, rect.left() - padding) 
	    # y = max(0, rect.top() - 2 * padding)
	    # w = int((rect.right() - rect.left()) * padding_threshold + (rect.right() - rect.left()))
	    # h = int((rect.bottom() - rect.top()) * padding_threshold + (rect.bottom() - rect.top()))
	    print("rect_to_bb ", h, w)
	    return (x , y , w , h)

	def detect_human_face(self):
		print("detect_human_face....")
		detector = dlib.get_frontal_face_detector()
		predictor = dlib.shape_predictor(self.predictor_path)
		dets = detector(self.input_image, 1)

		print("detected number of face ", len(dets))
		for k, rect in enumerate(dets):
			print("Rect", rect)
			x, y, w, h = self.rect_to_bb(rect)
			temp_img = self.input_image[int(y/2):y+h, x:x+w]
			y = int(y/2)
			h = y + h
			# cv.imshow("circle output",temp_img)
			# cv.waitKey(0)
			temp_dets = detector(temp_img, 1)
			print("temp dets ", temp_dets[0])
			shape = predictor(temp_img, temp_dets[0])
			_tuple = (x, y, w, h, shape.part(8).x, shape.part(8).y)
			# cv.circle(temp_img, (shape.part(8).x, shape.part(8).y), 10, (0, 255, 0), -1)
			
			temp_face_object = face_object(temp_img, k, _tuple)
			self.faces.append(temp_face_object)

	def image_write(self, folder_path):
		print("Write Image....", folder_path)
		for i in range(len(self.faces)):
   			output = str(self.faces[i].id) + r'.jpg'
   			print(folder_path + output)
   			cv.imwrite(folder_path + output, self.faces[i].face)
   			k = cv.waitKey(30) & 0xff
   			if k == 27:
   				break


class face_object:
	def __init__(self, face, _id, coordinate):
		self.face = face
		self.id = _id
		x, y, h, w, low_x, low_y = coordinate
		self.x = x
		self.y = y
		self.h = h
		self.w = w
		self.low_x = low_x
		self.low_y = low_y

	def showface(self):
		cv.imshow("Output", self.face)
		cv.waitKey(0)



# predictor_path = r'shape_predictor_68_face_landmarks.dat'
# image_path = r'\input\input.jpg'
# temp_path = r'\temp\\'
# cwd = os.getcwd()
# image_path = cwd + image_path
# print(image_path)
# img = cv.imread(image_path)
# # img = np.uint8(img)
# temp_object = human_face_detector(img, predictor_path)
# temp_object.detect_human_face()
# # for i in range(3):
# # 	print(temp_object.faces[i].face.shape)
# print(cwd + temp_path)
# temp_object.image_write(cwd + temp_path)