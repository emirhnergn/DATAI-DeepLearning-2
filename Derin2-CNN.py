#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical

#%%

mnist_train = pd.read_csv("mnist_train.csv")
mnist_test = pd.read_csv("mnist_test.csv")

#%%

x_train = mnist_train.iloc[:,1:].values
x_test = mnist_test.iloc[:,1:].values
y_train = mnist_train.iloc[:,0:1].values
y_test = mnist_test.iloc[:,0:1].values

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#%%
for i in range(1,6):
    deneme = x_train[i,:].reshape(28,28)
    plt.imshow(deneme)
    title = y_train[i]
    plt.title(title)
    plt.axis("off")
    plt.show()


#%%

from keras.models import Sequential
from keras.layers import MaxPooling2D,Conv2D,Dense,Flatten,Dropout,Activation
from keras.preprocessing.image import ImageDataGenerator

#%%

train_datagen = ImageDataGenerator(rotation_range=0.3,
                                   zoom_range=0.3,
                                   rescale=.1/255,
                                   shear_range=0.3)

test_datagen = ImageDataGenerator(rescale=.1/255)

train_datagen.fit(x_train)
test_datagen.fit(x_test)

#%%
model = Sequential()

model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=(28,28,1)))
model.add(Activation("relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.1))

model.add(Conv2D(filters=32,kernel_size=(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.1))

model.add(Conv2D(filters=32,kernel_size=(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(units=512,kernel_initializer="uniform"))
model.add(Activation("relu"))
model.add(Dropout(0.1))

model.add(Dense(units=512,kernel_initializer="uniform"))
model.add(Activation("relu"))
model.add(Dropout(0.1))

model.add(Dense(units=10))
model.add(Activation("softmax"))

model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])

batch_size = 20
#%%

train_generator = train_datagen.flow(x_train,y_train,batch_size=batch_size)
test_generator = test_datagen.flow(x_test,y_test,batch_size=batch_size)



#%%
hist = model.fit_generator(train_generator,steps_per_epoch=1600//batch_size,
                           epochs=30,
                           validation_data=test_generator,
                           validation_steps=800//batch_size,
                           verbose=2)

#%%

model.save_weights("deneme_mnist.h5")

import json
with open("mnist_hist.json","w") as f:
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
with codecs.open("mnist_hist.json","r",encoding="utf-8") as f:
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
