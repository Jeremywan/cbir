# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:49:11 2020

Modules related to AI and Pytorch

Module Name - Date tested


@author: user
"""

from PIL import Image
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models

# This function use a pretrained CNN VGG16 for feature extraction
#    input: image filename , showImage option (bool) 
#    output return featVec dim size = <1,4096> (feature from the last fc layer before Relu)
# Tested on 10 Sept 2020

def getCNNFeature(model, filename, device, showImage):
    #%%
    img = Image.open(filename)
    if showImage:        
        plt.imshow(img)
    
  
    
    #%% Preprocessing 
     
    transform = transforms.Compose([transforms.Resize((224,224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ])
  
    #%% Extract feature  
         
    img = transform(img) 
    image = img.to(device).unsqueeze(0)
    output = model(image)
    featVec = output.detach().cpu().numpy().reshape(1,-1)
    return featVec
