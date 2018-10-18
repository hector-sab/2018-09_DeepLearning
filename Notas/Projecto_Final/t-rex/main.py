import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

import utils as ut

class ControllsGUI:
	def __init__(self,viz,name='Controles'):
		self.tn_shape = (100,100) # Thumbnail (Height,Width)
		self.ims_set_shape = (100,100) # Tamaño de imagenes de la base de dato
		self.viz = viz

		self.window = tk.Tk()
		self.wind_shape = (900,450) # width,height
		self.window.geometry('{}x{}'.format(self.wind_shape[0],self.wind_shape[1]))
		self.window.title(name)

		self.btn_save_data = tk.Button(self.window,text='Guardar Imagenes')
		self.btn_save_data.grid(column=3,row=0)

		self.btn_train_model = tk.Button(self.window,text='Entrenar')
		self.btn_train_model.grid(column=3,row=1)

		self.btn_start_game = tk.Button(self.window,text='Iniciar')
		self.btn_start_game.grid(column=3,row=2)

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

		self.window.mainloop()

	def __get_image(self,tn):
		# Obtiene el cuadro actual
		frame = self.viz.get_frame()
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




if __name__=='__main__':
	cap = ut.VideoStreamThread()
	cap.start()
	viz = ControllsGUI(viz=cap)
	cap.stop()