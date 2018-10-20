import numpy as np

class DataSeg:
  """
  Permite generar la segmentación de los numeros de la base de 
  datos MNIST.
  
  Entradas:
    -Espera que se le den las imágenes de la forma ims = [num_ims,im_flat]
    dónde im_flat = height*weight*channels
    -Las etiquetas deben de estár de la forma lbs = [num_lbs,1]
  """
  def __init__(self,ims,lbs):
    # Reshaped images 
    self.ims = ims.reshape(-1,(ims.shape[1]*ims.shape[2]))
    self.lbs = lbs
    self.seg = self.create_seg_data()
    self.onehotenc = self.one_hot()

    self.batch_size = 1
    self.batch_in = 0
    self.batch_out = 0
  
  def create_seg_data(self):
    seg1 = (self.ims==0)*-1
    seg2 = (self.ims!=0)*self.lbs
    seg = seg1 + seg2
    seg += 1
    return(seg)

  def one_hot(self):
    num_lbs = self.lbs.shape[0]
    num_cls = 11
    onehot = np.zeros((num_lbs,num_cls))
    onehot[np.arange(0,num_lbs),self.lbs.reshape(-1)+1] = 1
    return(onehot)