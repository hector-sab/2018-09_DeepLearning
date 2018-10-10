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
    blues = y_pred==0
    red = y_pred==1
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
    y_pred = sigmoid(s)
    y_pred = (y_pred>0.5).astype(np.int32)

    plot_pred_AND(inputs,y_pred)

def sigmoid(x):
    # Funcion de activación sigmoide
    # Args:
    #    x (int|list|np.array): Valores a los que se le aplicará
    #        la sigmoide.
    #
    # Salidas:
    #    y (np.array): Valor de sigmoide correspondiente a x.
    y = 1/(1+np.exp(-x))
    return(y)

def train_AND(X,Y,W,b,iters=1000,bs=1,lr=0.01):
    # W has Shape [2,1]
    # b has Shape [1,1]

    i = 0
    for it in range(iters):
        inputs = X[i:i+bs,:] # Shape: [N,2]
        ys = Y[i:i+bs,:] # Shape: [N,1]
        print('y:',ys)

        A = np.dot(inputs,W) # Shape: [N,1]
        s = A + b # Shape: [N,1]
        
        y_pred = (s>=0).astype(np.int32) # Shape: [N,1]
        print('y_pred',y_pred)

        # Función de perdida
        loss = (ys-y_pred).astype(np.int32)
        loss = (loss**2)/bs

        print(loss)

        dL = 1
        dsigma = y_pred * dL
        print(dsigma.shape,inputs.shape,W.shape)
        #dW = np.dot(inputs)

