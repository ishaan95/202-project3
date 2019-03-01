import cv2

def parseFramesFromVid(path):
   vidcap = cv2.VideoCapture(path)
   success,image = vidcap.read()
   count = 0
   while success:
     cv2.imwrite("sailor_moon_train/%d.jpg" % count, image)
     success,image = vidcap.read()
     count += 1

parseFramesFromVid('sailor_moon.mp4')
