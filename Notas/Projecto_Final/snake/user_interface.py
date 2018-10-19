import os
import cv2
import numpy as np

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

import ui_utils as ut
from neural_model import Data

import game as ob

IM_SHAPE = (64,64) # Height, Width

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
		#self.lbl_center = tk.Label(self.window,text="Neutral")
		self.lbl_right = tk.Label(self.window,text="Derecha")
		self.lbl_bottom = tk.Label(self.window,text="Abajo")

		# Posicion de las etiquetas
		self.lbl_top.grid(column=1,row=0)
		self.lbl_left.grid(column=0,row=3)
		#self.lbl_center.grid(column=1,row=3)
		self.lbl_right.grid(column=2,row=3)
		self.lbl_bottom.grid(column=1,row=6)

		# Canvas de los thumbnails
		self.tn_cvs_top = tk.Canvas(self.window,width=self.tn_shape[1],height=self.tn_shape[0])
		self.tn_cvs_bottom = tk.Canvas(self.window,width=self.tn_shape[1],height=self.tn_shape[0])
		#self.tn_cvs_center = tk.Canvas(self.window,width=self.tn_shape[1],height=self.tn_shape[0])
		self.tn_cvs_left = tk.Canvas(self.window,width=self.tn_shape[1],height=self.tn_shape[0])
		self.tn_cvs_right = tk.Canvas(self.window,width=self.tn_shape[1],height=self.tn_shape[0])

		# Coloca a los thumbnails en sus posiciones
		self.tn_cvs_top.grid(column=1,row=1)
		self.tn_cvs_left.grid(column=0,row=4)
		#self.tn_cvs_center.grid(column=1,row=4)
		self.tn_cvs_right.grid(column=2,row=4)
		self.tn_cvs_bottom.grid(column=1,row=7)

		# Inicializa los thumbnails con imagenes negras
		bk_im = ImageTk.PhotoImage(image=Image.fromarray(np.zeros((self.tn_shape[1],self.tn_shape[0],3)).astype(np.uint8)))
		self.tn_im_top = self.tn_cvs_top.create_image(self.tn_shape[1]//2,self.tn_shape[0]//2,image=bk_im)
		self.tn_im_bottom = self.tn_cvs_bottom.create_image(self.tn_shape[1]//2,self.tn_shape[0]//2,image=bk_im)
		#self.tn_im_center = self.tn_cvs_center.create_image(self.tn_shape[1]//2,self.tn_shape[0]//2,image=bk_im)
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
		#self.btn_tn_center = tk.Button(self.window,text='Get',command=lambda:self.__get_image('center'))
		self.btn_tn_left = tk.Button(self.window,text='Get',command=lambda:self.__get_image('left'))
		self.btn_tn_right = tk.Button(self.window,text='Get',command=lambda:self.__get_image('right'))

		# Coloca los botones en posicion
		self.btn_tn_top.grid(column=1,row=2)
		self.btn_tn_left.grid(column=0,row=5)
		#self.btn_tn_center.grid(column=1,row=5)
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
		self.__model_class = model
		self.__trainer_class = trainer

		if cam is None:
			self.cam = ut.VideoStreamThread2()
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
		self.btn_save_data = tk.Button(self.root,text="Guardar Datos",command=self.__data_save)
		self.btn_load_data = tk.Button(self.root,text="Recargar Datos",command=self.__data_load)
		self.btn_train_model = tk.Button(self.root,text="Entrenar Modelo",command=self.__train_model)
		self.btn_run_game = tk.Button(self.root,text="Iniciar Juego",command=self.__run_game)

		self.btn_create_data.config(height=size[0],width=size[1])
		self.btn_save_data.config(height=size[0],width=size[1])
		self.btn_load_data.config(height=size[0],width=size[1])
		self.btn_train_model.config(height=size[0],width=size[1])
		self.btn_run_game.config(height=size[0],width=size[1])

		self.btn_create_data.grid(column=0,row=0)
		self.btn_save_data.grid(column=1,row=0)
		self.btn_load_data.grid(column=2,row=0)
		self.btn_train_model.grid(column=3,row=0)
		self.btn_run_game.grid(column=4,row=0)

	def __data_collection(self):
		print('Data Collection')
		self.root.withdraw()
		self.dataCollection.display_window()

	def __data_save(self):
		ims = np.zeros((0,self.dataCollection.ims_set_shape[0],
			self.dataCollection.ims_set_shape[1],3))
		lbs = np.zeros((0,1))

		ims_top = self.dataCollection.ims_set_top
		if ims_top is not None:
			lbs_top = np.ones((ims_top.shape[0],1))*0
			ims = np.concatenate([ims,ims_top],axis=0)
			lbs = np.concatenate([lbs,lbs_top],axis=0)

		ims_right = self.dataCollection.ims_set_right
		if ims_right is not None:
			lbs_right = np.ones((ims_right.shape[0],1))*1
			ims = np.concatenate([ims,ims_right],axis=0)
			lbs = np.concatenate([lbs,lbs_right],axis=0)

		ims_bottom = self.dataCollection.ims_set_bottom
		if ims_bottom is not None:
			lbs_bottom = np.ones((ims_bottom.shape[0],1))*2
			ims = np.concatenate([ims,ims_bottom],axis=0)
			lbs = np.concatenate([lbs,lbs_bottom],axis=0)

		ims_left = self.dataCollection.ims_set_left
		if ims_left is not None:
			lbs_left = np.ones((ims_left.shape[0],1))*3
			ims = np.concatenate([ims,ims_left],axis=0)
			lbs = np.concatenate([lbs,lbs_left],axis=0)

		np.save('saved_ims',ims)
		np.save('saved_lbs',lbs)

		print('Dataset shape:',ims.shape,lbs.shape)

	def __data_load(self):
		if os.path.isfile('saved_ims.npy') and os.path.isfile('saved_lbs.npy'):
			ims = np.load('saved_ims.npy')
			lbs = np.load('saved_lbs.npy')

			lbs = lbs.reshape(-1)
			self.dataCollection.ims_set_top = ims[lbs==1]
			self.dataCollection.ims_set_center = ims[lbs==0]
			self.dataCollection.ims_set_bottom = ims[lbs==2]
			print('Archivos cargados.')
		else:
			print('No se encontraron archivos que cargar.')


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
		self.model = self.__model_class()
		self.trainer = self.__trainer_class(model=self.model)
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
		self.root.withdraw()
		#ims,lbs = self.trainer.train_set.next_batch(10)
		#pred = self.model.predict(ims)
		print('Run Game')
		#print(lbs.reshape(-1))
		#print(pred.reshape(-1))
		self.cam.start()
		#tmp = self.cam.get_frame()
		#print(tmp.shape)
		game = ob.SnakeGame(model=self.model,cam=self.cam,im_shape=IM_SHAPE)
		game.start()
