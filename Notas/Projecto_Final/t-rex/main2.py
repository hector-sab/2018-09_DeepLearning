import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

import utils as ut

# TODO: Complete Model and Trainer classes, and Training method of the UI

IM_SHAPE = (64,64) # Height, Width
CLS = {0:'Center',1:'Top',2:'Right',3:'Bottom',4:'Left'}

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

		self.pred = tf.nn.softmax(self.fc3,axis=-1)

	def predict(self,inputs):
		fd = {self.ims_inp:inputs}
		pred = self.sess.run(self.pred,feed_dict=fc)
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


class DataCollection:
	def __init__(self,cam,root,name='Collection'):
		self.root = root
		self.cam = cam
		self.window = tk.Toplevel(self.root)
		self.__content()
		self.window.protocol("WM_DELETE_WINDOW",self._on_closing)
		self.window.withdraw()

	def display_window(self):
		self.window.deiconify()
		self.cam.start()

	def __content(self):
		self.tn_shape = (100,100) # Thumbnail (Height,Width)
		self.ims_set_shape = IM_SHAPE # Tamaño de imagenes de la base de dato

		# Contendra todas las imagenes tomadas
		self.ims_set_top = None
		self.ims_set_bottom = None
		self.ims_set_center = None
		self.ims_set_left = None
		self.ims_set_right = None

		# Etiquetas
		self.lbl_top = tk.Label(self.window,text="Arriba")
		self.lbl_left = tk.Label(self.window,text="Izquierda")
		self.lbl_center = tk.Label(self.window,text="Neutral")
		self.lbl_right = tk.Label(self.window,text="Derecha")
		self.lbl_bottom = tk.Label(self.window,text="Abajo")

		# Posicion de las etiquetas
		self.lbl_top.grid(column=1,row=0)
		self.lbl_left.grid(column=0,row=3)
		self.lbl_center.grid(column=1,row=3)
		self.lbl_right.grid(column=2,row=3)
		self.lbl_bottom.grid(column=1,row=6)

		# Canvas de los thumbnails
		self.tn_cvs_top = tk.Canvas(self.window,width=self.tn_shape[1],height=self.tn_shape[0])
		self.tn_cvs_bottom = tk.Canvas(self.window,width=self.tn_shape[1],height=self.tn_shape[0])
		self.tn_cvs_center = tk.Canvas(self.window,width=self.tn_shape[1],height=self.tn_shape[0])
		self.tn_cvs_left = tk.Canvas(self.window,width=self.tn_shape[1],height=self.tn_shape[0])
		self.tn_cvs_right = tk.Canvas(self.window,width=self.tn_shape[1],height=self.tn_shape[0])

		# Coloca a los thumbnails en sus posiciones
		self.tn_cvs_top.grid(column=1,row=1)
		self.tn_cvs_left.grid(column=0,row=4)
		self.tn_cvs_center.grid(column=1,row=4)
		self.tn_cvs_right.grid(column=2,row=4)
		self.tn_cvs_bottom.grid(column=1,row=7)

		# Inicializa los thumbnails con imagenes negras
		bk_im = ImageTk.PhotoImage(image=Image.fromarray(np.zeros((self.tn_shape[1],self.tn_shape[0],3)).astype(np.uint8)))
		self.tn_im_top = self.tn_cvs_top.create_image(self.tn_shape[1]//2,self.tn_shape[0]//2,image=bk_im)
		self.tn_im_bottom = self.tn_cvs_bottom.create_image(self.tn_shape[1]//2,self.tn_shape[0]//2,image=bk_im)
		self.tn_im_center = self.tn_cvs_center.create_image(self.tn_shape[1]//2,self.tn_shape[0]//2,image=bk_im)
		self.tn_im_left = self.tn_cvs_left.create_image(self.tn_shape[1]//2,self.tn_shape[0]//2,image=bk_im)
		self.tn_im_right = self.tn_cvs_right.create_image(self.tn_shape[1]//2,self.tn_shape[0]//2,image=bk_im)

		# Contenedor de las imagenes en los thumbnails
		self.im_top = None
		self.im_bottom = None
		self.im_center = None
		self.im_left = None
		self.im_right = None

		# Acciona la recoleccion de imagenes
		self.btn_tn_top = tk.Button(self.window,text='Get',command=lambda:self.__get_image('top'))
		self.btn_tn_bottom = tk.Button(self.window,text='Get',command=lambda:self.__get_image('bottom'))
		self.btn_tn_center = tk.Button(self.window,text='Get',command=lambda:self.__get_image('center'))
		self.btn_tn_left = tk.Button(self.window,text='Get',command=lambda:self.__get_image('left'))
		self.btn_tn_right = tk.Button(self.window,text='Get',command=lambda:self.__get_image('right'))

		# Coloca los botones en posicion
		self.btn_tn_top.grid(column=1,row=2)
		self.btn_tn_left.grid(column=0,row=5)
		self.btn_tn_center.grid(column=1,row=5)
		self.btn_tn_right.grid(column=2,row=5)
		self.btn_tn_bottom.grid(column=1,row=8)

	def __get_image(self,tn):
		# Obtiene el cuadro actual
		frame = self.cam.get_frame()
		# Reorganiza el orden de los colores y la posicion de las columnas
		frame[:,::-1,::-1] = frame
		# Redimensiona las imagenes para el thumbnail y la base de datos
		tn_im = cv2.resize(frame,self.tn_shape)
		frame = cv2.resize(frame,self.ims_set_shape)

		# Acondiciona el formato del thumbnail
		im = Image.fromarray(tn_im)

		frame = np.expand_dims(frame,axis=0)
		
		if tn=='top':
			self.im_top = ImageTk.PhotoImage(image=im)
			self.tn_cvs_top.itemconfig(self.tn_im_top,image=self.im_top)
			
			# Guarda la imagen en la base de datos
			if self.ims_set_top is None:
				self.ims_set_top = frame
			else:
				self.ims_set_top = np.concatenate([self.ims_set_top,frame],axis=0)
			# Indica el numero de imagenes tomadas en el boton
			self.btn_tn_top.config(text=str(self.ims_set_top.shape[0]))
			print('top',self.ims_set_top.shape)

		elif tn=='bottom':
			self.im_bottom = ImageTk.PhotoImage(image=im)
			self.tn_cvs_bottom.itemconfig(self.tn_im_bottom,image=self.im_bottom)
			
			if self.ims_set_bottom is None:
				self.ims_set_bottom = frame
			else:
				self.ims_set_bottom = np.concatenate([self.ims_set_bottom,frame],axis=0)
			self.btn_tn_bottom.config(text=str(self.ims_set_bottom.shape[0]))
			print('bottom',self.ims_set_bottom.shape)

		elif tn=='center':
			self.im_center = ImageTk.PhotoImage(image=im)
			self.tn_cvs_center.itemconfig(self.tn_im_center,image=self.im_center)
			
			if self.ims_set_center is None:
				self.ims_set_center = frame
			else:
				self.ims_set_center = np.concatenate([self.ims_set_center,frame],axis=0)
			self.btn_tn_center.config(text=str(self.ims_set_center.shape[0]))
			print('center',self.ims_set_center.shape)

		elif tn=='left':
			self.im_left = ImageTk.PhotoImage(image=im)
			self.tn_cvs_left.itemconfig(self.tn_im_left,image=self.im_left)
			
			if self.ims_set_left is None:
				self.ims_set_left = frame
			else:
				self.ims_set_left = np.concatenate([self.ims_set_left,frame],axis=0)
			self.btn_tn_left.config(text=str(self.ims_set_left.shape[0]))
			print('left',self.ims_set_left.shape)

		elif tn=='right':
			self.im_right = ImageTk.PhotoImage(image=im)
			self.tn_cvs_right.itemconfig(self.tn_im_right,image=self.im_right)

			if self.ims_set_right is None:
				self.ims_set_right = frame
			else:
				self.ims_set_right = np.concatenate([self.ims_set_right,frame],axis=0)
			self.btn_tn_right.config(text=str(self.ims_set_right.shape[0]))
			print('right',self.ims_set_right.shape)

	def _on_closing(self):
		self.cam.stop()
		#self.window.destroy()
		self.window.withdraw()
		self.root.deiconify()


class TrainModel:
	def __init__(self,root):
		self.root = root
		self.window = tk.Toplevel(self.root)
		self.__content()
		self.window.protocol("WM_DELETE_WINDOW",self._on_closing)
		self.window.withdraw()

	def __content(self):
		self.btn_start_train = tk.Button(self.window,text="Iniciar Entrenamiento",
			command=self.__train_model)
		self.btn_start_train.grid(column=0,row=0)

		self.lbl_lr = tk.Label(self.window,text="Learning Rate:")
		self.lbl_bs = tk.Label(self.window,text="Batch Size:")
		self.lbl_its = tk.Label(self.window,text="Iterations:")

		self.lbl_lr.grid(column=1,row=0)
		self.lbl_bs.grid(column=3,row=0)
		self.lbl_its.grid(column=5,row=0)

		self.inp_lr = tk.Entry(self.window)
		self.inp_bs = tk.Entry(self.window)
		self.inp_its = tk.Entry(self.window)

		self.inp_lr.grid(column=2,row=0)
		self.inp_bs.grid(column=4,row=0)
		self.inp_its.grid(column=6,row=0)

		self.pbar = ttk.Progressbar(self.window,orient=tk.HORIZONTAL)#,length=100,mode='determinate')
		self.pbar.grid(column=0,row=1,columnspan=5,sticky=tk.W+tk.E+tk.N+tk.S)

	def set_models(self,model,trainer):
		self.model = model
		self.trainer = trainer

	def __train_model(self):
		lr = self.inp_lr.get()
		bs = self.inp_bs.get()
		its = self.inp_its.get()

		if lr!="" and bs!="" and its!="":
			print('Training')
			lr = float(lr)
			bs = int(bs)
			its = int(its)

			self.trainer.train(its=its,bs=bs)
			print("¡Entrenamiento Completado!")
		
		# TODO: Fix tkinter pbar progress indicator
		#self.pbar.step(99)
		#for i in range(100):
		#	self.pbar.step(i)

	def display_window(self):
		self.window.deiconify()

	def _on_closing(self):
		self.window.withdraw()
		self.root.deiconify()


class Main:
	def __init__(self,model=None,trainer=None,cam=None,name='Main'):
		if cam is None:
			self.cam = ut.VideoStreamThread()
		else:
			self.cam = cam
		self.root = tk.Tk()
		#self.root.geometry('600x200')

		self.__content()

		self.dataCollection = DataCollection(cam=self.cam,root=self.root)
		self.trainModel = TrainModel(root=self.root)

		self.model_initialized = False
		

		self.root.mainloop()

	def __content(self):
		size = (20,20) # height, width
		self.btn_create_data = tk.Button(self.root,text="Recolectar Datos",command=self.__data_collection)
		self.btn_train_model = tk.Button(self.root,text="Entrenar Modelo",command=self.__train_model)
		self.btn_run_game = tk.Button(self.root,text="Iniciar Juego",command=self.__run_game)

		self.btn_create_data.config(height=size[0],width=size[1])
		self.btn_train_model.config(height=size[0],width=size[1])
		self.btn_run_game.config(height=size[0],width=size[1])

		self.btn_create_data.grid(column=0,row=0)
		self.btn_train_model.grid(column=1,row=0)
		self.btn_run_game.grid(column=2,row=0)

	def __data_collection(self):
		print('Data Collection')
		self.root.withdraw()
		self.dataCollection.display_window()

	def __train_model(self):
		print('Train Model')
		self.root.withdraw()
		
		if not self.model_initialized:
			self.__init_models()
			self.trainModel.set_models(model=self.model,
				trainer=self.trainer)
			self.model_initialized = True
		else:
			self.__set_new_training_data()

		self.trainModel.display_window()

	def __init_models(self):
		self.model = Model()
		self.trainer = Trainer(model=self.model)
		self.__set_new_training_data()

	def __set_new_training_data(self):
		ims = np.zeros((0,self.dataCollection.ims_set_shape[0],
			self.dataCollection.ims_set_shape[1],3))
		lbs = np.zeros((0,1))

		ims_center = self.dataCollection.ims_set_center
		if ims_center is not None:
			lbs_center = np.ones((ims_center.shape[0],1))*0
			ims = np.concatenate([ims,ims_center],axis=0)
			lbs = np.concatenate([lbs,lbs_center],axis=0)

		ims_top = self.dataCollection.ims_set_top
		if ims_top is not None:
			lbs_top = np.ones((ims_top.shape[0],1))*1
			ims = np.concatenate([ims,ims_top],axis=0)
			lbs = np.concatenate([lbs,lbs_top],axis=0)

		ims_right = self.dataCollection.ims_set_right
		if ims_right is not None:
			lbs_right = np.ones((ims_right.shape[0],1))*2
			ims = np.concatenate([ims,ims_right],axis=0)
			lbs = np.concatenate([lbs,lbs_right],axis=0)

		ims_bottom = self.dataCollection.ims_set_bottom
		if ims_bottom is not None:
			lbs_bottom = np.ones((ims_bottom.shape[0],1))*3
			ims = np.concatenate([ims,ims_bottom],axis=0)
			lbs = np.concatenate([lbs,lbs_bottom],axis=0)

		ims_left = self.dataCollection.ims_set_left
		if ims_left is not None:	
			lbs_left = np.ones((ims_left.shape[0],1))*4
			ims = np.concatenate([ims,ims_left],axis=0)
			lbs = np.concatenate([lbs,lbs_left],axis=0)


		print('Dataset shape:',ims.shape,lbs.shape)
		
		train_set = Data(ims=ims,lbs=lbs)
		self.trainer.set_train_set(train_set=train_set)

	def __run_game(self):
		print('Run Game')



if __name__=='__main__':
	#model = Model()
	#cam = ut.VideoStreamThread()
	
	#cam.start()
	main = Main(model=Model,trainer=Trainer)
	#cam.stop()