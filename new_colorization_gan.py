import keras
import shutil
import keras
from keras.models import Model,Sequential
from keras.layers import *
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import numpy as np
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from keras.callbacks import Callback,ModelCheckpoint
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K
from sklearn.preprocessing import OneHotEncoder
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from IPython.display import display, Image
from matplotlib.pyplot import imshow
from keras.layers import Conv2D, UpSampling2D, InputLayer
from keras.models import Sequential
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import lab2rgb, rgb2lab
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import os
import random
from keras.layers.advanced_activations import*
from keras.optimizers import Adam
import PIL
from PIL import Image


files = os.listdir('images/Train')

samples = 1000
train = np.empty((samples,256,256,3), 'float32')
lab_train = np.empty((samples,256,256,3),'float32')
black_train = np.empty((samples,256,256,1),'float32')

for i in range(samples):
  image = cv2.imread('images/Train/'+files[i*4])
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  train[i] = image
  lab = rgb2lab((1.0/255)*image)
  lab_train[i] = lab
  black_train[i] = lab[:,:,0].reshape(256,256,1)
  
  
lab_train = lab_train/128
train = train/255.0
black_train = black_train/128

img_shape = (256,256,1)
generator_inp = Input(shape=img_shape)

x1 = Conv2D(64, (3,3), padding='same', strides=1)(generator_inp)
x1 = LeakyReLU(alpha = 0.2)(x1)

x2 = Conv2D(64, (3,3), padding='same', strides=2)(x1)
x2 = BatchNormalization(momentum = 0.5)(x2)
x2 = LeakyReLU(alpha = 0.2)(x2)

x3 = Conv2D(128, (3,3), padding='same', strides=2)(x2)
x3 = BatchNormalization(momentum = 0.5)(x3)
x3 = LeakyReLU(alpha = 0.2)(x3)

x4 = Conv2D(256, (3,3), padding='same', strides=2)(x3)
x4 = BatchNormalization(momentum = 0.5)(x4)
x4 = LeakyReLU(alpha = 0.2)(x4)

x5 = Conv2D(512, (3,3), padding='same', strides=2)(x4)
x5 = BatchNormalization(momentum = 0.5)(x5)
x5 = LeakyReLU(alpha = 0.2)(x5)

x6 = Conv2D(512, (3,3), padding='same', strides=2)(x5)
x6 = BatchNormalization(momentum = 0.5)(x6)
x6 = LeakyReLU(alpha = 0.2)(x6)

x7 = Conv2D(512, (3,3), padding='same', strides=2)(x6)
x7 = BatchNormalization(momentum = 0.5)(x7)
x7 = LeakyReLU(alpha = 0.2)(x7)

x8 = Conv2D(512, (3,3), padding='same', strides=2)(x7)
x8 = BatchNormalization(momentum = 0.5)(x8)
x8 = LeakyReLU(alpha = 0.2)(x8)

x9 = Conv2D(512, (3,3), padding='same', strides=1)(x8)
x9 = UpSampling2D()(x9)
x9 = BatchNormalization(momentum = 0.5)(x9)
x9 = Activation('relu')(x9)
x9 = concatenate([x7,x9])

x10 = Conv2D(512, (3,3), padding='same', strides=1)(x9)
x10 = UpSampling2D()(x10)
x10 = BatchNormalization(momentum = 0.5)(x10)
x10 = Activation('relu')(x10)
x10 = concatenate([x6,x10])

x11 = Conv2D(512, (3,3), padding='same', strides=1)(x10)
x11 = UpSampling2D()(x11)
x11 = BatchNormalization(momentum = 0.5)(x11)
x11 = Activation('relu')(x11)
x11 = concatenate([x5,x11])

x12 = Conv2D(256, (3,3), padding='same', strides=1)(x11)
x12 = UpSampling2D()(x12)
x12 = BatchNormalization(momentum = 0.5)(x12)
x12 = Activation('relu')(x12)
x12 = concatenate([x4,x12])

x13 = Conv2D(128, (3,3), padding='same', strides=1)(x12)
x13 = UpSampling2D()(x13)
x13 = BatchNormalization(momentum = 0.5)(x13)
x13 = Activation('relu')(x13)
x13 = concatenate([x3,x13])

x14 = Conv2D(64, (3,3), padding='same', strides=1)(x13)
x14 = UpSampling2D()(x14)
x14 = BatchNormalization(momentum = 0.5)(x14)
x14 = Activation('relu')(x14)
x14 = concatenate([x2,x14])

x15 = Conv2D(64, (3,3), padding='same', strides=1)(x14)
x15 = UpSampling2D()(x15)
x15 = BatchNormalization(momentum = 0.5)(x15)
x15 = Activation('relu')(x15)
x15 = concatenate([x1,x15])

x16 = Conv2D(2, (3,3), activation = 'tanh', padding='same', strides=1)(x15)
x17 = concatenate([generator_inp,x16])

generator = Model(inputs = generator_inp, outputs = x17)

image_shape_color = (256,256,3)
discriminator_inp = Input(shape=image_shape_color)

d1 = Conv2D(64, (3,3), padding='same', strides=1)(discriminator_inp)
d1 = LeakyReLU(alpha = 0.2)(d1)

d2 = Conv2D(64, (3,3), padding='same', strides=2)(d1)
d2 = BatchNormalization(momentum = 0.5)(d2)
d2 = LeakyReLU(alpha = 0.2)(d2)

d3 = Conv2D(128, (3,3), padding='same', strides=2)(d2)
d3 = BatchNormalization(momentum = 0.5)(d3)
d3 = LeakyReLU(alpha = 0.2)(d3)

d4 = Conv2D(256, (3,3), padding='same', strides=2)(d3)
d4 = BatchNormalization(momentum = 0.5)(d4)
d4 = LeakyReLU(alpha = 0.2)(d4)

d5 = Conv2D(512, (3,3), padding='same', strides=2)(d4)
d5 = BatchNormalization(momentum = 0.5)(d5)
d5 = LeakyReLU(alpha = 0.2)(d5)

d6 = Conv2D(512, (3,3), padding='same', strides=2)(d5)
d6 = BatchNormalization(momentum = 0.5)(d6)
d6 = LeakyReLU(alpha = 0.2)(d6)

d7 = Conv2D(512, (3,3), padding='same', strides=2)(d6)
d7 = BatchNormalization(momentum = 0.5)(d7)
d7 = LeakyReLU(alpha = 0.2)(d7)

d8 = Conv2D(512, (3,3), padding='same', strides=2)(d7)
d8 = BatchNormalization(momentum = 0.5)(d8)
d8 = LeakyReLU(alpha = 0.2)(d8)

d8 = Flatten()(d8)
d8 = Dense(100)(d8)
d8 = LeakyReLU(alpha = 0.2)(d8)
d8 = Dense(1, activation = 'sigmoid')(d8)

discriminator = Model(inputs = discriminator_inp, outputs = d8)

gen_shape = (256,256,1)
dis_shape = (256,256,3)

adam = Adam(lr=2e-4, beta_1=0.5, beta_2=0.999, epsilon=1e-08)

discriminator.compile(loss="binary_crossentropy", optimizer=adam)

discriminator.trainable = False

gan_input = Input(shape = gen_shape)
gen_out = generator(gan_input)
gan_final = discriminator(gen_out) 
gans = Model(inputs=gan_input, outputs=[gen_out,gan_final])
gans.compile(loss=["mse", "binary_crossentropy"], loss_weights=[1., 1e-3], optimizer=adam)


m = train.shape[0]
loss_history = []
batch_size = 25
losg = []
losd = []
for epoch in range(100):
    itera  = int(m/batch_size)
    dis_mean = 0
    gan_mean = 0
    for i in range(itera):
      color = train[i*batch_size:min((i+1)*batch_size,m)]
      lab = lab_train[i*batch_size:min((i+1)*batch_size,m)]
      black = black_train[i*batch_size:min((i+1)*batch_size,m)]
      lab_color_img = generator.predict(black)
      
      
      real = np.ones(color.shape[0]) - np.random.random_sample(color.shape[0])*0.1
      fake = np.random.random_sample(lab_color_img.shape[0])*0.1
      
      dis_loss1 = discriminator.train_on_batch(x = lab,
                                         y = real)
      dis_loss2 = discriminator.train_on_batch(x = lab_color_img,
                                    y = fake)
      
      dis_loss = (dis_loss1 + dis_loss2)*0.5
      
      dis_mean = dis_mean + dis_loss
      
      gan_loss = gans.train_on_batch(x = black,
                                     y = [lab, real])
      gan_loss = gan_loss[0] + gan_loss[1]*1e-3
      
      gan_mean = gan_mean + gan_loss
      
      
      print('Epoch = '+str(epoch)+' batch = '+str(i)+' | discriminator loss = '+str(dis_loss)+' | gan loss = '+str(gan_loss))
    dis_mean = dis_mean/itera
    gan_mean = gan_mean/itera
    print('Epoch = '+str(epoch)+' | mean discriminator loss = '+str(dis_mean)+' | mean gan loss = '+str(gan_mean))
    losg.append(gan_mean)
    losd.append(dis_mean)
    print('------------------------------------------------Epoch '+str(epoch)+' complete-----------------------------------------------')
