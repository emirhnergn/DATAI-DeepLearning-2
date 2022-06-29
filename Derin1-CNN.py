#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D,Activation,Dropout,Flatten,Dense,MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from glob import glob
import cv2
from warnings import filterwarnings
filterwarnings("ignore")

#%%

train_path = "fruits-360/Training/" 
test_path = "fruits-360/Test/" 

#%%

img = load_img(train_path + "Apple Braeburn/0_100.jpg")
plt.imshow(img)
plt.axis("off")
plt.show()

x = img_to_array(img)
print(x.shape)

#%%

className = glob(train_path + '/*')
numberOfClass = len(className)
print("Number of Class:",numberOfClass)

#%%

model = Sequential()

model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(x.shape[0],x.shape[1],x.shape[2])))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=32,kernel_size=(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=64,kernel_size=(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(units=numberOfClass))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

batch_size = 32

#%%

train_datagen = ImageDataGenerator(featurewise_center=False,
                                   samplewise_center=False,
                                   rescale= 1./255,
                                   rotation_range=0.5,
                                   shear_range=0.3,
                                   zoom_range=0.3,
                                   horizontal_flip=True,
                                   )

test_datagen = ImageDataGenerator(rescale= 1./255)

#%%

train_generator = train_datagen.flow_from_directory(train_path,
                                                    target_size=x.shape[0:2],
                                                    batch_size=batch_size,
                                                    color_mode="rgb",
                                                    class_mode="categorical")

test_generator = test_datagen.flow_from_directory(test_path,
                                                  target_size=x.shape[0:2],
                                                  batch_size=batch_size,
                                                  color_mode="rgb",
                                                  class_mode="categorical")

#%%

hist = model.fit_generator(generator=train_generator,
                            steps_per_epoch = 1600//batch_size,
                            epochs=10 ,
                            validation_data=test_generator,
                            validation_steps=800//batch_size)

#%%

model.save_weights("deneme_fruits.h5")

import json
with open("cnn_fruit_hist.json","w") as f:
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
with codecs.open("cnn_fruit_hist.json","r",encoding="utf-8") as f:
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













#%%