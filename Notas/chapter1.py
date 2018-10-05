import numpy as np
import matplotlib.pyplot as plt

def plot_AND():
    X = np.array([[1,1],
              [1,0],
              [0,1],
              [0,0]])
    
    plt.plot(X[1:,0],X[1:,1],'ro')
    plt.plot(X[0,0],X[0,1],'bo')

def plot_pred_AND(X,y_pred):
	y_pred = y_pred.reshape(-1)
	blues = y_pred>=0
	red = y_pred<0
	plt.plot(X[blues,0],X[blues,1],'bo')
	plt.plot(X[red,0],X[red,1],'ro')

def plot_area_AND(W,b):
	X = np.linspace(-10,10,100)
	Y = np.linspace(-10,10,100)
	Gx,Gy = np.meshgrid(X,Y)

	Gx = Gx.reshape(-1,1)
	Gy = Gy.reshape(-1,1)

	inputs = np.hstack([Gx,Gy])

	s = np.dot(inputs,W) + b
	y_pred = (s>=0).astype(np.int32)

	plot_pred_AND(inputs,y_pred)
