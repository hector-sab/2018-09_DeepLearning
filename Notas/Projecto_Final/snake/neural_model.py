import numpy as np
from tqdm import tqdm
import tensorflow as tf

IM_SHAPE = (64,64) # Height, Width
CLS = {0:'Top',1:'Right',2:'Left',3:'Bottom'}

class Data:
	def __init__(self,ims,lbs,im_h=64,im_w=64):
		self.IM_H = im_h
		self.IM_W = im_w

		self.ims = ims
		self.lbs = lbs

		self.total_it = self.ims.shape[0]
		self.current_item = 0

	def next_batch(self,bs=0):
		"""
		Returns the next batch of images and labels.
		Args:
			bs (int): Size of the batch
		"""
		
		ims = self.ims[self.current_item:self.current_item+bs]
		lbs = self.lbs[self.current_item:self.current_item+bs]

		ims = np.reshape(ims,(-1,self.ims.shape[1],self.ims.shape[2],self.ims.shape[3]))
		lbs = np.reshape(lbs,(-1,1))

		self.current_item += bs

		if self.current_item>=self.total_it:
			self.current_item = 0
		
		return(ims,lbs)

	def reset_ind(self):
		self.current_item = 0

class Model:
	def __init__(self,init=False):
		self.sess = tf.Session()

		self.IM_H = IM_SHAPE[0]
		self.IM_W = IM_SHAPE[1]
		self.NUM_C =len(CLS)

		self.ims_inp = tf.placeholder(tf.float32,shape=[None,self.IM_H,self.IM_W,3])
		#self.lbs_inp = tf.placeholder(tf.float32,shape=[None,1])
		#self.lbs_onehot = tf.one_hot(tf.cast(self.lbs_inp,tf.int32),depth=len(self.NUM_C))

		self.__model()

		if init:
			self.init_variables()

	def __model(self):
		self.conv1 = tf.layers.conv2d(inputs=self.ims_inp,filters=32,kernel_size=3,activation=tf.nn.relu)
		self.pool1 = tf.layers.max_pooling2d(self.conv1,pool_size=2,strides=2)

		self.conv2 = tf.layers.conv2d(inputs=self.pool1,filters=32,kernel_size=3,activation=tf.nn.relu)
		self.pool2 = tf.layers.max_pooling2d(self.conv2,pool_size=2,strides=2)

		self.conv3 = tf.layers.conv2d(inputs=self.pool2,filters=32,kernel_size=3,activation=tf.nn.relu)
		self.pool3 = tf.layers.max_pooling2d(self.conv3,pool_size=2,strides=2)

		shape = self.pool3.get_shape()
		self.flat1 = tf.reshape(self.pool3,[-1,shape[1].value*shape[2].value*shape[3].value])
		
		self.fc1 = tf.layers.dense(self.flat1,units=256,activation=tf.nn.relu)
		self.fc2 = tf.layers.dense(self.fc1,units=256,activation=tf.nn.relu)
		self.fc3 = tf.layers.dense(self.flat1,units=self.NUM_C)
		self.last_layer = tf.identity(self.fc3)

		pred = tf.nn.softmax(self.fc3,axis=-1)
		self.pred = tf.argmax(pred,axis=-1)

	def predict(self,inputs):
		fd = {self.ims_inp:inputs}
		pred = self.sess.run(self.pred,feed_dict=fd)
		return(pred)

	def init_variables(self):
		self.sess.run(tf.global_variables_initializer())

class Trainer:
	def __init__(self,model):
		#self.train_set = train_set

		self.model = model
		self.sess = self.model.sess

		self.ims_inp = self.model.ims_inp
		self.lbs_inp = tf.placeholder(tf.float32,shape=[None,1])
		self.lbs_onehot = tf.one_hot(tf.cast(self.lbs_inp,tf.int32),depth=self.model.NUM_C)

		self.loss = self.__loss()
		self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)

		self.init_variables()

	def __loss(self):
		loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.lbs_onehot,logits=self.model.last_layer)
		loss = tf.reduce_mean(loss)
		return(loss)

	def train(self,its=1,bs=1):
		pbar = tqdm(range(its))
		for it in pbar:
			ims,lbs = self.train_set.next_batch(bs)
			fd = {self.ims_inp:ims,self.lbs_inp:lbs}

			self.sess.run(self.optimizer,feed_dict=fd)

	def init_variables(self):
		self.sess.run(tf.global_variables_initializer())

	def set_train_set(self,train_set):
		self.train_set = train_set
