#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense
from glob import glob
from warnings import filterwarnings
filterwarnings("ignore")

#%%

train_path = "fruits-360/Training/"
test_path =    "fruits-360/Test/"

#%%

img = load_img(train_path + "Avocado/0_100.jpg")
plt.imshow(img)
plt.axes("off")
plt.show()

#%%

x = img_to_array(img)
print(x.shape)

#%%

numberOfClass = len(glob(train_path+"/*"))
print("Number of Class:",numberOfClass)

#%%

vgg = VGG16()
print(vgg.summary())
print(type(vgg))

#%%

vgg_layer_list = vgg.layers
print(vgg_layer_list)

#%%

model = Sequential()
for i in range(len(vgg_layer_list)-1):
    model.add(vgg_layer_list[i])

print(model.summary())

for layers in model.layers:
    layers.trainable = False

model.add(Dense(units=numberOfClass,activation="softmax"))

print(model.summary())

model.compile(loss="categorical_crossentropy",optimizer="rmsprop",metrics=["accuracy"])
#%%

train_data = ImageDataGenerator().flow_from_directory(train_path,target_size=(224,224))
test_data = ImageDataGenerator().flow_from_directory(test_path,target_size=(224,224))

batch_size = 32

#%%

hist = model.fit_generator(train_data,
                           steps_per_epoch=1600//batch_size,
                           epochs=15,
                           validation_data=test_data,
                           validation_steps=800//batch_size)

#%%

model.save_weights("deneme_fruits_transfer_learning.h5")

import json
with open("cnn_fruit_hist_transfer_learning.json","w") as f:
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
import codecs 
with codecs.open("cnn_fruit_hist_transfer_learning.json","r",encoding="utf-8") as f:
    h = json.loads(f.read())
    
plt.plot(h ["loss"],label="Train Loss")
plt.plot(h["val_loss"],label="Validation Loss")
plt.legend()
plt.show()

plt.plot(h["acc"],label="Train Acc")
plt.plot(h["val_acc"],label="Validation Acc")
plt.legend()
plt.show()    










#%%
