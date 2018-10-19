import cv2
import numpy as np
import threading as th

class VideoStream:
	def __init__(self):
		self.cap = cv2.VideoCapture(0)

	def get_frame(self):
		for i in range(4):
			self.cap.grab()
		ret,frame = self.cap.read()
		return(frame)

	def release(self):
		self.cap.release()

class VideoStreamThread:
	#https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
	def __init__(self,device=0):
		self.device = device
		#self.stopped = False

	def start(self):
		self.stopped = False
		self.stream = cv2.VideoCapture(self.device)
		th.Thread(target=self.main_loop,args=()).start()
		return(self)

	def main_loop(self):
		while True:
			if self.stopped:
				break

			self.grabbed,self.frame = self.stream.read()

	def get_frame(self):
		return(self.frame)

	def stop(self):
		self.stopped = True
		self.stream.release()

class VideoStreamThread2:
	#https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
	def __init__(self,device=0):
		self.stream = cv2.VideoCapture(device)
		self.stopped = False
		self.init = False

	def start(self):
		self.stopped = False

		if self.init==False:
			th.Thread(target=self.main_loop,args=()).start()
			self.init = True
		
		return(self)

	def main_loop(self):
		while True:
			if self.stopped:
				return()

			self.grabbed,self.frame = self.stream.read()

	def get_frame(self):
		return(self.frame)

	def stop(self):
		self.stopped = True