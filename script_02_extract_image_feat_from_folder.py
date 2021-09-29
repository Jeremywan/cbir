# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 19:03:09 2020

@author: user
"""

from PIL import Image
import torch
import matplotlib.pyplot as plt
import numpy as np
# import torch.nn.functional as F
# from torch import nn
from torchvision import datasets, transforms
import torchvision.models as models
import os
import sys
from time import time
import pickle
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device= torch.device("cpu")
# lab related modules
from ai_pytorch_module import *
from cbir_module import *

#%% Set Path

imgpath = r'.\images'
experimentPath = r'D:\1_Code\FOE\ECE3086\lab1_cbir_student'
os.chdir(experimentPath)
sys.path.append(os.getcwd())
imgPath= os.path.join(experimentPath,'images')
sys.path.append(imgpath)
theCurrentPath = os.path.abspath('.')

#%% Read a selected jpeg images in a given folder
chosenImgId = 3

fileList=[]
files = os.listdir(imgpath)
for f in files:
    if f.endswith('.jpg'):
        (name,ext) =  os.path.splitext(f)
        f = str( '{:03d}'.format(int(name)) ) + ext  # convert string to 3 digit int
        fileList.append(f)

print("\n The number of images in the folder used for image database = " , len(fileList))

fileNameDB = {}
numImages = len(fileList)


#%%

class Database :
    def __init__(self) :
        self.imageName =  None
        self.featCNN = None
        
#db = Database()

#%% select model

model = models.vgg16(pretrained=True)
model.classifier = model.classifier[:4] # until linear (3)before relu (4)
model.to(device)

#%% Try extract feature for one image
filename = '005.jpg'
print(filename)
filename = os.path.join(imgpath,filename )
imgFeature = getCNNFeature(model, filename,  device, showImage = True,)

#%% Run extraction on a list of images in folder
# Init a blank database , empty list
database = [None] * numImages
ctr=0
start = time()

for i in range(numImages):
#for i in [700, 799, 899, 998]:   
    database[i] = Database()
    
    #filename = os.path.join(imgPath,fileList[i])
    filename = fileList[i]
    database[i].imageName = filename
    filename = os.path.join(imgPath,filename )
    database[i].featCNN = getCNNFeature(model,filename,  device,  showImage = False)
    database[i].featureName = 'VGG16 CNN fc layer before Relu'
    database[i].id = int(i)
    database[i].classLabel = (int(i)// 100) + 1 # 100 img in one class
    # print("Extracting feature for image ", i)
    # print("class label = ", database[i].classLabel)
    ctr=ctr+1
    if i%100 ==0:
        print("Extracting feature for image ", i)
        print("class label = ", database[i].classLabel)
        
finish = time()
print("Time taken to extract = (sec) = ", finish - start) # 25 sec using GPU, 1141 sec with cpu


#%%
# save the data
savedData = {'database':database}

with open("CBIR_database.pickle", 'wb') as f:
    pickle.dump(savedData, f)

#%%
# Try load
#del database
with open("CBIR_database.pickle","rb") as f:
    dataDict = pickle.load(f)
database = dataDict['database']

#%% Try preview image
id=99
imFile = database[id].imageName
label = database[id].classLabel
imFile  = os.path.join(imgPath,imFile )
im = Image.open(imFile)
plt.figure(figsize=(8,6))

plt.imshow(im) , plt.axis('off')
titleStr = " Image {}.jpg label = {} Label name = {}".format(str(id), label, LabelDic[label])
plt.title(titleStr,  fontsize=20)





























































