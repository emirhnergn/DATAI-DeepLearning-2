#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as f
from PIL import Image
import os
import time

#%%

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

#%%

def read_images(path,num_img):
    array = np.zeros([num_img,64*32])
    i = 0
    for img in os.listdir(path):
        img_path = path + "\\" + img
        img = Image.open(img_path,mode="r")
        data = np.asarray(img,dtype="uint8")
        data = data.flatten()
        array[i,:] = data
        i += 1
    return array

#%%
#train negative
train_negative_path = r"C:\Users\ergin\Desktop\spyder\DATAI-DeepLearning-2\LSIFIR\Classification\Train\neg"
num_train_negative_img = 43390
train_negative_array = read_images(train_negative_path,num_train_negative_img)
x_train_negative_tensor = torch.from_numpy(train_negative_array)
print("x_train_negative_tensor",x_train_negative_tensor.size())
y_train_negative_tensor = torch.zeros(num_train_negative_img,dtype=torch.long)
print("y_train_negative_tensor",y_train_negative_tensor.size())
#train positive
train_positive_path = r"C:\Users\ergin\Desktop\spyder\DATAI-DeepLearning-2\LSIFIR\Classification\Train\pos"
num_train_positive_img = 10208
train_positive_array = read_images(train_positive_path,num_train_positive_img)
x_train_positive_tensor = torch.from_numpy(train_positive_array)
print("x_train_positive_tensor",x_train_positive_tensor.size())
y_train_positive_tensor = torch.zeros(num_train_positive_img,dtype=torch.long)
print("y_train_positive_tensor",y_train_positive_tensor.size())
#test negative
test_negative_path = r"C:\Users\ergin\Desktop\spyder\DATAI-DeepLearning-2\LSIFIR\Classification\Test\neg"
num_test_negative_img = 22050
test_negative_array = read_images(test_negative_path,num_test_negative_img)
x_test_negative_tensor = torch.from_numpy(test_negative_array)
print("x_test_negative_tensor",x_test_negative_tensor.size())
y_test_negative_tensor = torch.zeros(num_test_negative_img,dtype=torch.long)
print("y_test_negative_tensor",y_test_negative_tensor.size())
#test positive
test_positive_path = r"C:\Users\ergin\Desktop\spyder\DATAI-DeepLearning-2\LSIFIR\Classification\Test\pos"
num_test_positive_img = 5944
test_positive_array = read_images(test_positive_path,num_test_positive_img)
x_test_positive_tensor = torch.from_numpy(test_positive_array)
print("x_test_positive_tensor",x_test_positive_tensor.size())
y_test_positive_tensor = torch.zeros(num_test_positive_img,dtype=torch.long)
print("y_test_positive_tensor",y_test_positive_tensor.size())

#%%

x_train = torch.cat((x_train_negative_tensor,x_train_positive_tensor),0)
y_train = torch.cat((y_train_negative_tensor,y_train_positive_tensor),0)

x_test = torch.cat((x_test_negative_tensor,x_test_positive_tensor),0)
y_test = torch.cat((y_test_negative_tensor,y_test_positive_tensor),0)

print("x_train:",x_train.size())
print("y_train:",y_train.size())

print("x_test:",x_test.size())
print("y_test:",y_test.size())

#%%

plt.imshow(x_train[45001].reshape(64,32),cmap="gray")
plt.axis("off")
plt.show()

#%%

num_epoch = 1
num_classes = 2
batch_size = 8933
learning_rate = 0.00001

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        
        self.conv1 = nn.Conv2d(1,10,5)
        self.pool1 = nn.MaxPool2d(2,2)
        #self.pool2 = nn.MaxPool2d(2,2) #AYNIYSA KOYMAYA GEREK YOK 
        self.conv2 = nn.Conv2d(10,16,5)
        
        self.fc1 = nn.Linear(16*13*5,520)
        self.fc2 = nn.Linear(520,130)
        self.fc3 = nn.Linear(130,num_classes)
    
    def forward(self,x):
        
        x = self.pool1(f.relu((self.conv1(x))))
        x = self.pool1(f.relu((self.conv2(x))))
    
        x = x.view(-1,16*13*5)
        
        x = f.relu((self.fc1(x)))
        x = f.relu((self.fc2(x)))
        x = self.fc3(x)
        
        return x

#%%
import torch.utils.data

train = torch.utils.data.TensorDataset(x_train,y_train)
trainloader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                          shuffle=True)

test= torch.utils.data.TensorDataset(x_test,y_test)
testloader = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                          shuffle=False)

#%%

net = Net()
# net = Net().to(device) #FOR GPU
net = net.double()
#%%
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=learning_rate,momentum=0.8)

#%%

start = time.time()

train_accuracy = []
test_accuracy = []
lost_list = []

use_gpu = False #True for GPU

for epoch in range(num_epoch):
    for i,data in enumerate(trainloader,0):
        
        inputs,labels = data
        inputs = inputs.view(batch_size, 1, 64, 32) #reshape
        inputs = inputs.float() #float
        
        #USE GPU
        #if use_gpu:
        #    if torch.cuda.is_available():
        #        inputs,labels = inputs.to(device),labels.to(device)
        
        # zero gradient  
        optimizer.zero_grad()
        
        #forward       
        outputs = net(inputs.double())
        
        #loss  
        loss = criterion(outputs,labels)
        
        #back
        loss.backward()
        
        #update weights
        optimizer.step()
        
    #test
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data 
            images = images.view(batch_size,1,64,32)
            images.float()
            
            #USE GPU
            #if use_gpu:
            #    if torch.cuda.is_available():
            #        images,labels = images.to(device),labels.to(device)
            
            outputs = net(images)
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc1 = 100*correct/total
    print("Accuracy Test: ",acc1)
    test_accuracy.append(acc1)
    
    
    with torch.no_grad():
        for data in trainloader:
            images, labels = data 
            images = images.view(batch_size,1,64,32)
            images.float()
            
            #USE GPU
            #if use_gpu:
            #    if torch.cuda.is_available():
            #        images,labels = images.to(device),labels.to(device)
            
            outputs = net(images)
            _,predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc2 = 100*correct/total
    print("Accuracy Train: ",acc2)
    train_accuracy.append(acc2)
    
print("Train is done.")
    
            

end = time.time()
process_time = (end-start)/60
print("Process Time:",process_time)
#%%























#%%