#%%

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,Flatten,Conv2D
from keras.models import Sequential
from keras.datasets import cifar10
from warnings import filterwarnings
from keras.utils import to_categorical
filterwarnings("ignore")

#%%

(x_train,y_train),(x_test,y_test) = cifar10.load_data()
print("x_train shape:",x_train.shape)
print("train sample:",x_train.shape[0])

numberOfClass = 10

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

input_shape = x_train.shape[1:]
print(input_shape)

#%%
plt.imshow(x_train[2234].astype(np.uint8))
plt.axis("off")
plt.show()
#%%

deneme = x_train[2234].astype(np.uint8)
deneme = cv2.resize(deneme,(224,224))

plt.imshow(deneme)
plt.axis("off")
plt.show()

#%%

def resize_img(img):
    numberOfImage = img.shape[0]
    new_array = np.zeros((numberOfImage,48,48,3))
    for i in range(numberOfImage):
        new_array[i] = cv2.resize(img[i,:,:,:],(48,48))
    return new_array

#%%

x_train = resize_img(x_train)
x_test = resize_img(x_test)

#%%

vgg = VGG19(include_top = False,weights="imagenet",input_shape=(48,48,3))
print(vgg.summary())

vgg_layer_list = vgg.layers
print(vgg_layer_list)

#%%

model = Sequential()

for i in range(len(vgg_layer_list)):
    model.add(vgg_layer_list[i])
    
for layers in model.layers:
    layers.trainable = False

print(model.summary())

model.add(Flatten())
model.add(Dense(units=128))
model.add(Dense(units=128))
model.add(Dense(units=numberOfClass,activation="softmax"))
model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])

batch_size=200

#%%

train_data = ImageDataGenerator().flow(x=x_train,y=y_train)
test_data = ImageDataGenerator().flow(x=x_test,y=y_test)


#%%

hist = model.fit_generator(train_data,
                           steps_per_epoch=1600//batch_size,
                           epochs=10,
                           validation_data=test_data,
                           validation_steps=800//batch_size)

#%%

model.save_weights("cifar10_transfer_learning.h5")

import json
with open("cifar10_transfer_learning.json","w") as f:
    json.dump(hist.history,f)

#%%
print(hist.history.keys())
plt.plot(hist.history["loss"],label="Train Loss")
plt.plot(hist.history["val_loss"],label="Validation Loss")
plt.legend()
plt.show()

plt.plot(hist.history["acc"],label="Train Acc")
plt.plot(hist.history["val_acc"],label="Validation Acc")
plt.legend()
plt.show()

#%%







































#%%