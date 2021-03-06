import numpy as np
import tensorflow as tf

class Data:
    # Clase auxiliar para la óbtencion de los subsets de tamaño bs
    def __init__(self,x,y=None):
        # Args:
        #    x (np.array): Arreglo de tamaño [None,1]
        #    y (np.array): Arreglo de tamaño [None,1]
        self.x = x
        self.y = y
        
        self.total_it = len(self.x)
        self.current_item = 0
    
    def next_batch(self,bs=1):
        # Regresa el sub set de entrenamiento de tamaño bs
        # Args:
        #    bs (int): Tamaño del subset que se desea

        ind = self.current_item
        
        if ind>=self.total_it:
            self.current_item = 0
            ind = 0
        
        x = self.x[ind:ind+bs]
        if self.y is not None:
            y = self.y[ind:ind+bs]
        else:
            y = None
        
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        self.current_item = ind + bs
        return(x,y)
    
    def reset_ind(self):
        # Reinicial el indice de iteracíon
        self.current_item = 0