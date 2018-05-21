# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 00:39:35 2018

@author: Hrid
"""

import numpy as np
from numpy import linalg as LA
import os, os.path
#np.set_printoptions(threshold=np.nan)
import cv2
from PIL import Image
import glob
import tensorflow as tf
from matplotlib import pyplot as plt


data=np.empty((0,2048)) # 2048 is the size of the feature vector/number of pixels after  resizing the image

for filename in glob.glob("C:\\Users\\Hrid\\Desktop\\NEW_NEW_NEW\\training\\*.png"):
    im=cv2.imread(filename,0)
 
    #print('size:',im.shape)
    resized=cv2.reshape(im,(32,64))
    im_ravel=resized.ravel()
    arr = np.append( data,[im_ravel],axis=0)
    data=arr
    
final_data=arr 
mu=np.mean(final_data,axis=0)


plt.figure(1)
 
k=1677
for i in range(0,4):
    
    img1=final_data[k,:]
    ir=np.reshape(img1,(32,64))
    ir=np.uint8(ir)
    plt.subplot(2,2,i+1)
    plt.imshow(ir,cmap='gray')
    k=k+1
    print('k=== ',k)
plt.suptitle('sample image from training dataset')   
plt.show()



data=final_data-mu
covariance=np.cov(data.T)
values,vector=LA.eig(covariance)

pov=np.cumsum(np.divide(values,sum(values)))
plt.figure
plt.plot(pov)
plt.title('percentage of variance explained')




vsort=vector[:,0:301]
scores=np.dot(data,vsort)
projection=np.dot(scores,vsort.T)+mu

%matplotlib qt
plt.figure(2)
k=1677
for i in range(0,4):
    
    img1_train=projection[k,:]
    ir_train=np.reshape(img1_train,(32,64))
    ir=np.uint8(ir_train)
    plt.subplot(2,2,i+1)
    plt.imshow(ir_train,cmap='gray')
    k=k+1
    print('k=== ',k)
plt.suptitle('Image construction using PCA')   
plt.show()      




























