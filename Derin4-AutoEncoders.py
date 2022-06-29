#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input,Dense
from keras.datasets import fashion_mnist
import json,codecs
from warnings import filterwarnings
filterwarnings("ignore")

#%%

(x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()

#%%

x_train = x_train.astype("float32") / 255.0
x_test= x_test.astype("float32") / 255.0

#%%

x_train = x_train.reshape((len(x_train),x_train.shape[1:][0]*x_train.shape[1:][1]))
x_test = x_test.reshape((len(x_test),x_test.shape[1:][0]*x_test.shape[1:][1]))
#%%

plt.imshow(x_train[150].reshape(28,28),cmap="gray")
plt.axis("off")
plt.show()

#%%

input_img = Input(shape=(784,))

encoded = Dense(units=32,activation="relu")(input_img)

encoded = Dense(units=16,activation="relu")(encoded)

decoded = Dense(units=32,activation="relu")(encoded)

output = Dense(units=784,activation="sigmoid")(decoded)

autoencoder = Model(input_img,output)

autoencoder.compile(optimizer="rmsprop",loss="binary_crossentropy")

#%%

hist = autoencoder.fit(x=x_train,y=x_train,
                       epochs=100,
                       batch_size=256,
                       shuffle=True,
                       validation_data=(x_train,x_train),
                       verbose=2)

#%%

autoencoder.save_weights("fashion_mnist_deneme.h5")

#%%

plt.plot(hist.history["loss"],label="loss")
plt.plot(hist.history["val_loss"],label="val_loss")
plt.legend()
plt.show()

#%%
with open("fashion_mnist_deneme.json","w") as f:
    json.dump(hist.history,f)


#%%

with codecs.open("fashion_mnist_deneme.json","r",encoding="utf-8") as f:
    n = json.loads(f.read())

plt.plot(n["loss"],label="loss")
plt.plot(n["val_loss"],label="val_loss")
plt.legend()
plt.show()

#%%

encoder = Model(input_img,encoded)
encoded_img = encoder.predict(x_test)


#%%
plt.imshow(x_test[1500].reshape(28,28))
plt.axis("off")
plt.show()

plt.imshow(encoded_img[1500].reshape(4,4))
plt.axis("off")
plt.show()
#%%

decoded_img = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.axis("off")
    
    ax = plt.subplot(2,n,i+n+1)
    plt.imshow(decoded_img[i].reshape(28,28))
    plt.axis("off")
plt.show()

#%%

























#%%









#%%
















