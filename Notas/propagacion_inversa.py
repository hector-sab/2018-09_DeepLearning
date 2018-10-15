import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def crear_lbs(x,y,z):
    out = []
    for i in range(x.shape[0]):
        txt = 'P({},{},{:0.2f})'.format(x[i],y[i],z[i])
        out.append(txt)

    return(out)

def fxy(x,y):
    x = np.array(x)**2
    y = np.array(y)**2
    out = x*y
    return(out)

def _fxy(x,y):
    # Dados X y Y, regresa el valor de Z
    x = np.array(x)**2
    y = np.array(y)**2
    x = np.reshape(x,[-1,1])
    y = np.reshape(y,[1,-1])
    out = np.matmul(x,y)
    return(out)

init = -4
end = 4
steps = 20

X,Y = np.meshgrid(np.linspace(init,end,steps),np.linspace(init,end,steps))
Z = _fxy(np.linspace(init,end,steps),np.linspace(init,end,steps))


def mostrar_superficie_xy(x,y,z,lbl,pp=False):
    # https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
    # x,y,z (list): Listas que contienen las coordenadas de puntos a dibujar
    # lbl (list): Lista que contiene la leyenda a dibujar con cada punto
    # pp (bool): Plot path
    # Creamos una figura donde se mostrará la superficie 3D
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Dibujamos la superficie
    ax.plot_surface(X,Y,Z,cmap='hot',alpha=0.5)

    ax.scatter(x, y, z, c='r', marker='o')
    for i in range(len(x)):
        ax.text(x[i],y[i],z[i],lbl[i])

    if pp:
        ax.plot(x,y,z)


    ax.set_xlabel('Eje X')
    ax.set_ylabel('Eje Y')
    ax.set_zlabel('Eje Z')