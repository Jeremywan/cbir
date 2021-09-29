# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 11:25:23 2020

@author: haris
"""

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os


LabelDic ={ 1: "African",
        2: "Beach",
        3: "Building",
        4: "Bus",
        5: "Dinasour",
        6: "Elephant",
        7: "Flower",
        8: "House",
        9: "Mountain",
        10: "Food"
       }
#print( LabelDic[1] )

def doRetrieval(featQuery , k, database, imgpath, showImage=True):

    numImages = len(database)
    dist_cnn = []
    idx_k = []
    
    for f in range (0,numImages) :
        dist = np.linalg.norm(featQuery - database[f].featCNN)
        dist_cnn.append(dist)
        
    idx_k = np.argsort(dist_cnn)
        
    return idx_k[1:k+1]

def doRetrieval_hist(featQuery , k, database, imgpath, showImage=True):

    numImages = len(database)
    dist_hist = []
    idx_k = []
    
    for f in range (0,numImages) :
        dist = np.linalg.norm(featQuery - database[f].featColorHist)
        dist_hist.append(dist)
    
    idx_k = np.argsort(dist_hist)
        
    return idx_k[1:k+1]
  
def getPrecisionRank_K(k, queryLabel, retrievedID, database):

    rel_img = 0
    
    for f in retrievedID:
        label = database[f].classLabel
        print(label, end=' ')
        if queryLabel == label:
                rel_img += 1
            
    precision_k = rel_img/k
   
    return precision_k
    #endfunc()














