import cv2
import numpy as np
import multiprocessing as mp




class VideoStream2:
	def __init__(self):
		self.queue = mp.Queue()
		self.cap = cv2.VideoCapture(0)

		self.cam_process = mp.Process(target=self.loop,args=(self.queue))
		self.cam_process.start()

	def loop(self):
		ret,frame = self.cap.read()
		self.queue.put(frame)

	def get_frame(self):
		if self.queue.empty():
			return(None)

		frame = self.queue.get()
		return(frame)

	def release(self):
		pass


import threading as th
from queue import Queue


class VideoStream3:
	def __init__(self,queueSize=1):
		self.stream = cv2.VideoCapture(0)
		self.Q = Queue(maxsize=queueSize)
		self.stopped = False

	def start(self):
		# Start the thread to read frames
		t = th.Thread(target=self.update,args=())
		t.deamon = True
		t.start()
		return(self)

	def update(self):
		while True:
			if self.stopped:
				return()

			if not self.Q.full():
				grabbed,frame = self.stream.read()

				self.Q.put(frame)

	def get_frame(self):
		frame = self.Q.get()
		return(frame)

	def stop(self):
		self.stopped = True


class VideoStream:
	def __init__(self,device=0):
		self.stream = cv2.VideoCapture(device)
		self.stopped = False

	def start(self):
		th.Thread(target=self.update,args=()).start()
		return(self)

	def update(self):
		while True:
			if self.stopped:
				return()

			self.grabbed,self.frame = self.stream.read()

	def get_frame(self):
		return(self.frame)

	def stop(self):
		self.stopped = True