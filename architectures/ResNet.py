#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function
import numpy as np
import torch
import torch.autograd as ag
from utils import mnist_reader
import MNISTtools
import matplotlib.pyplot as plt
import time
from tqdm import tqdm_notebook
from sklearn.metrics import confusion_matrix
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import data_split
import gc


# In[2]:


# Can also add data augmentation transforms here
train_transform = transforms.Compose([transforms.Resize(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(), transforms.Lambda(lambda x: torch.cat([x,x,x],0)) ])

# transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   #transforms.ToPILImage(),
test_transform = transforms.Compose([ transforms.Resize(224), 
                                     transforms.ToTensor(), transforms.Lambda(lambda x: torch.cat([x,x,x],0)) ])


# In[3]:


# print('Downloading/Checking for data......')
trainset = torchvision.datasets.FashionMNIST(root='data/downloads', train=True,
                                        download=True, transform=train_transform)       # download=True for the 1st time
testset = torchvision.datasets.FashionMNIST(root='data/downloads', train=False,
                                        download=True, transform=test_transform)        # download=True for the 1st time
train, validation = data_split.train_valid_split(trainset)    # separates 10% for validation


# In[4]:


trainloader = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(validation, batch_size=10, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, num_workers=2)


# In[5]:


print('# of batches in training: ',len(trainloader), ',   Total train data: ',len(trainloader)*5)
print('# of batches in test: ',len(testloader), ',   Total test data: ',len(testloader)*5)
img, label = train[100]
print('image size: ', img.size())
# plt.imshow(img[0,:,:])
# print(type(label))


# # Loading the VGG16 trained on imagenet

# In[7]:


vgg16 = torchvision.models.vgg16_bn(pretrained='imagenet')
# print(vgg16.classifier[6].out_features)        # 1000, as it was trained for 1000 classes

#--------------------------------------------------------------------------------------
# REMEMBER: vgg16.features --> convolutional layers, vgg16.classifier --> FC layers
#--------------------------------------------------------------------------------------

# freeze all parameters in covolutional layers
for parameter in vgg16.features.parameters():
    parameter.require_grad = False

in_ftrs = vgg16.classifier[6].in_features
features = list(vgg16.classifier.children())[:-1]       # Removing last layer to add out 10 units layer
features.extend([nn.Linear(in_ftrs, 10)])               # adding out layer with 10 units
vgg16.classifier = nn.Sequential(*features)             # replacing it with the model with new last layer


# In[11]:


# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

vgg16 = torchvision.models.resnet18(pretrained='imagenet')
# print(vgg16.classifier[6].out_features)        # 1000, as it was trained for 1000 classes

#--------------------------------------------------------------------------------------
# REMEMBER: vgg16.features --> convolutional layers, vgg16.classifier --> FC layers
#--------------------------------------------------------------------------------------

# freeze all parameters in covolutional layers
#for parameter in vgg16.features.parameters():
#    parameter.require_grad = False

in_ftrs = vgg16.fc.in_features

num_ftrs = vgg16.fc.in_features
vgg16.fc = nn.Linear(num_ftrs, 10)


#features = list(vgg16.classifier.children())[:-1]       # Removing last layer to add out 10 units layer
#features.extend([nn.Linear(in_ftrs, 10)])               # adding out layer with 10 units
#vgg16.classifier = nn.Sequential(*features)             # replacing it with the model with new last layer


# In[12]:


# print(vgg16)  # prints the architecture


# In[13]:


vgg16=vgg16.cuda()


# In[14]:


def test(vgg16):
    correct = 0
    total = 0
    acc_test = 0.0
    
    vgg16.train(False)
    vgg16.eval()    
    for i,data in enumerate(testloader):
        images, labels = data

        images = images.cuda()
        labels = labels.cuda()

        images = ag.Variable(images, volatile=True)
        labels = ag.Variable(labels, volatile=True)

        outputs = vgg16(images)
        _, predicted = torch.max(outputs.data, 1)
        acc_test += torch.sum(predicted == labels.data)
        del outputs, predicted
        torch.cuda.empty_cache()
    #     gc.collect()

    print(acc_test/len(testset))
    # print('Test Accuracy of the model on the 10000 test images: %.4f %%' % (100 * correct / total))


# In[15]:


test(vgg16)


# # Training the model

# In[16]:


vgg16.train(True)
# vgg16.train()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# In[17]:


start=time.time()
for epoch in range(3):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):       # 0 is just to start i from 0
        # get the inputs
        inputs, labels = data
        
        inputs = ag.Variable(inputs, requires_grad = False)
        labels = ag.Variable(labels, requires_grad = False)
        
        # transformations
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = vgg16(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss[0].data.cpu().numpy()[0]
        # print(type(loss[0].data.cpu().numpy()[0]))
        if i==50:
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 50))
        if i==100:
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 100))
        if i==200:
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 200))
        if i==500:
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 500))
        
        if i % 500 == 499:    # print every 500 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0
        del inputs, labels, loss, outputs
        torch.cuda.empty_cache()
    print('epoch: ', epoch)
    print('Total time taken in training (secs): ',time.time()-start)
print('Total time taken in training (secs): ',time.time()-start)
print('Finished Training')


# In[18]:


torch.cuda.empty_cache()


# In[19]:


# test(vgg16)    # 1 epoch


# In[20]:


test(vgg16)


# In[ ]:




