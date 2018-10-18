### Curso de Deep Learning, CICATA IPN Qro. 2018.
www.cicataqro.ipn.mx


_Notas para los instructores:_

En el siguiente se encuentra la [presentación para el curso](https://docs.google.com/presentation/d/1bH6CFAhBaSgL6xC21K4GCNIy6dnb9gfB2beEpRJX660/edit?usp=sharing).

Carpeta del curso en [Drive](https://drive.google.com/drive/folders/1aMvmaUOgp3H9lPXoWzzPNoRznzgH5tIQ?usp=sharing).

El contenido teórico se encuentra en la [wiki](https://github.com/hector-sab/2018-09_DeepLearning/wiki).


__Objetivo del curso__

El curso de Deep Learning presentado utiliza herramientas básicas de código abierto. Pretende introducir al usuario a este entorno de programación para el entrenamiento, prueba y detección de objetos utilizando redes neuronales en Tensorflow con un enfoque teórico-práctico. 


__Contenido del curso__

* Introducción a Machine Learning.
* Introducción a Deep Learning.
* Introducción a Tensorflow.
* Redes neuronales biológicas vs redes neuronales artificiales.
* Funciones de activación.
* Regresión lineal.
* Modelos de redes neuronales artificiales.
* Función de pérdida.
* Feed-forward y back-propagation.
* Inicialización y actualización de pesos.
* Batch normalization.
* Pooling.
* Dropout.
* Regularización.
* Redes neuronales convolucionales.
_Diversión y mucho más_


__Requisitos__ 

Anaconda (para Windows, iOs o Linux) con Python 2.7+ (https://www.anaconda.com/download/)

Cuenta activa de Github (github.com)

__Instalación de librerías__

conda create -n py36 python=3.6 anaconda

![ima001](https://docs.google.com/drawings/d/e/2PACX-1vTbUqswknPfLuDOWezlqqNNuhZ2hwlSGkxnh-pSieYD3sa_Uh-Yr5-Wq6WVDsJkCGcjHjoaHsw-JsW2/pub?w=753&h=216)

conda activate py36

conda install -c anaconda git jupyter numpy matplotlib cython scikit-image

![ima002](https://docs.google.com/drawings/d/e/2PACX-1vQS5Z2_WR9oDPHOz5g5f0bHot8UpA6meyWwU20HxxsC-h3dDxY4N-o8jRdYI1i8VAbyrThnMMmpwnFx/pub?w=1012&h=307)

conda install anaconda-client

pip install opencv-contrib-python

conda install tensorflow 

conda install -c conda-forge tqdm

conda install -c conda-forge keras

conda install pytictoc -c ecf

conda update --all


# Verificación de instalación

Probablemente te preguntes ¿Cómo puedes comprobar que tu instalación es correcta? Escribe las siguientes líneas de código:

python -V

git --version

jupyter --version

python

import numpy as np

import matplotlib as plt

import cv2

cv2.__version__

![ima003](https://docs.google.com/drawings/d/e/2PACX-1vSeZYvCdT1r0aTybL4pf_IA1frawKi_94KIVfjzFdAoDA4LfHr4vXD2VqHjT0aT1yzWhV9jS2rtE45X/pub?w=1362&h=549)

import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')

sess = tf.Session()

print(sess.run(hello))

exit()




__Instructores__

Ing. Héctor Sánchez Barrera, hsanchezb1600@alumno.ipn.mx

Ing. Dagoberto Pulido, dpulidoa1600@alumno.ipn.mx

M. En TA. Sandra de la Fuente, sdelafuenteb1400@alumno.ipn.mx



__Organizadores__

M. En TA. Raymundo Ramos.

M. En TA. Christian Matilde.



__Referencias__

_Cursos relacionados_

Fei-Fei Li, Andrej Karpathy, Justin Johnson, [“CS231n: Convolutional Neural Networks for Visual Recognition”](http://cs231n.stanford.edu/). Stanford University, Spring 2016.

Mariano Rivera, Alan Reyes, Francisco Gurrola, Ulises Rodríguez, [“XII Taller-Escuela de Procesamiento de Imágenes (PI18). Taller Keras”](http://pi2018.eventos.cimat.mx). CIMAT, 2018. 

Sanja Fidler, [“Deep Learning in Computer Vision”](http://www.cs.toronto.edu/~fidler/teaching/2015/CSC2523.html). University of Toronto, Winter 2016.

Hugo Larochelle, [“Neural Networks”](http://info.usherbrooke.ca/hlarochelle/neural_networks/content.html). Université de Sheerbroke.

Joan Bruna, [“Stats212b: Topics on Deep Learning”](https://github.com/joanbruna/stat212b). Berkeley University. Spring 2016.

Yann LeCun, [“Deep Learning: Nine Lectures at Collège de France”](http://cilvr.nyu.edu/doku.php?id=courses%3Adeeplearning-cdf2016%3Astart). Collège de France, Spring 2016. [Facebook page]

Dhruv Batra, [“ECE 6504: Deep learning for perception”](https://computing.ece.vt.edu/~f15ece6504/). Virginia Tech, Fall 2015.

Vincent Vanhoucke, Arpan Chakraborty, [“Deep Learning”](https://www.udacity.com/course/deep-learning--ud730). Google 2016.

Xavier Giro-i-Nieto, “Deep learning for computer vision: [Image](http://www.slideshare.net/xavigiro/deep-learning-for-computer-vision-14-image-analytics-lasalle-2016), [Object](http://www.slideshare.net/xavigiro/deep-learning-for-computer-vision-24-object-analytics-lasalle-2016), [Video Analytics](http://www.slideshare.net/xavigiro/deep-learning-for-computer-vision-34-video-analytics-lasalle-2016) and [Beyond](http://www.slideshare.net/xavigiro/deep-learning-for-computer-vision-44-beyond-vision-lasalle-2016)”. LaSalle URL. May 2016.

German Ros, Joost van de Weijer, Marc Masana, Yaxing Wang, [“Hands-on Deep Learning with Matconvnet”](http://www.cvc.uab.es/~gros/index.php/hands-on-deep-learning-with-matconvnet/). Computer Vision Center (CVC) 2015.

Niloy J. Mitra, Iasonas Kokkinos, Paul Guerrero, Vladimir Kim, Kostas Rematas, Tobias Ritschel, [“Deep Learning for Graphics”](http://geometry.cs.ucl.ac.uk/dl4g/). Eurographics 2018.

Xavier Giro-i-Nieto,	Elisa Sayrol,	Amaia Salvador,	Jordi Torres,	Eva Mohedano,	Kevin McGuinness, [“Deep Learning for Computer Vision Barcelona”](http://imatge-upc.github.io/telecombcn-2016-dlcv/). UPC ETSETB TelecomBCN 2016.


_Fuentes de información_

Goodfellow, Ian; Bengio, Y.; Courville, A. Deep Learning [on line].    2016 [Consultation: 22/02/2016]. Available on: <http://www.deeplearningbook.org/>. 

