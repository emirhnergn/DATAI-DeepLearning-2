#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from keras.layers import Dense, Dropout,Input,ReLU
from keras.models import Model,Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
from warnings import filterwarnings
filterwarnings("ignore")
import json,codecs

import keras
import tensorflow as tf

config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4 } ) 
sess = tf.compat.v1.Session(config=config) 
keras.backend.set_session(sess)
    
#%%

(x_train,y_train),(x_test,y_test) = mnist.load_data()

#%%

plt.imshow(x_train[0])
plt.axis("off")
plt.show()

#%%

x_train = (x_train.astype(np.float32)-127.5)/127.5
print(x_train.shape)
#%%

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
print(x_train.shape)
#%%

def create_generator():
    
    generator = Sequential()
    
    generator.add(Dense(units=512,input_dim=100))
    generator.add(ReLU())
    
    generator.add(Dense(units=512))
    generator.add(ReLU())
    
    generator.add(Dense(units=1024))
    generator.add(ReLU())
    
    generator.add(Dense(units=784,activation="tanh"))
    
    generator.compile(loss = "binary_crossentropy",
                      optimizer=Adam(lr=0.0001,beta_1=0.5))
    
    return generator

g = create_generator()
print(g.summary())

#%%

def create_discriminator():
    
    discriminator = Sequential()
    
    discriminator.add(Dense(units=1024,input_dim=784))
    discriminator.add(ReLU())
    discriminator.add(Dropout(0.4))
    
    discriminator.add(Dense(units=512))
    discriminator.add(ReLU())
    discriminator.add(Dropout(0.4))
    
    discriminator.add(Dense(units=256))
    discriminator.add(ReLU())
    
    discriminator.add(Dense(units=1,activation="sigmoid"))
    
    discriminator.compile(loss="binary_crossentropy",
                          optimizer=Adam(lr=0.0001,beta_1=0.5))
    
    return discriminator

d = create_discriminator()
print(d.summary())

#%%

def create_gan(discriminator,generator):
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs = gan_input,outputs = gan_output)
    gan.compile(loss = "binary_crossentropy",optimizer="adam")
    return gan

gan = create_gan(d,g)
print(gan.summary())


#%%

import time

epochs = 40
batch_size = 256

for e in range(epochs):
    bas = time.time()
    print("Epochs:",e)
    for _ in range(batch_size):
        
        noise = np.random.normal(0,1,[batch_size,100])
        
        generated_images = g.predict(noise)
        
        image_batch = x_train[np.random.randint(low=0,high=x_train.shape[0],
                                                size=batch_size)]
        x = np.concatenate([image_batch,generated_images])
        
        y_dis = np.zeros(batch_size*2)
        y_dis[:batch_size] = 1
        
        d.trainable = True
        d.train_on_batch(x,y_dis)
        
        noise = np.random.normal(0,1,[batch_size,100])
        
        y_gen = np.ones(batch_size)
        
        d.trainable = False
        
        gan.train_on_batch(noise,y_gen)
    print("Epochs: ",e,"Geçen Zaman: ",time.time()-bas)
    



#%%

g.save_weights("gans_deneme.h5")

#%%

noise = np.random.normal(loc=0,scale=1,size=[100,100])
generated_images = g.predict(noise)
generated_images = generated_images.reshape(100,28,28)
plt.imshow(generated_images[15],interpolation="nearest")
plt.axis("off")
plt.show()

#%%























#%%
