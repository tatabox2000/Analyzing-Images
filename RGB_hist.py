import matplotlib.pyplot as plt
import numpy as np
import cv2
import os 
os.chdir(r'c://Users/analyst')
im = cv2.imread('00008.BMP')

red = np.zeros((im.shape[0],im.shape[1]),np.uint8)
green=np.zeros((im.shape[0],im.shape[1]),np.uint8)
blue =np.zeros((im.shape[0],im.shape[1]),np.uint8)

blue= im[:,:,0]

green= im[:,:,1]
red= im[:,:,2]
im_flat=im.flatten()


plt.subplot(3,1,1),plt.hist(red.flatten(),bins=100,color='r'),plt.xlim(0,256)
plt.subplot(3,1,2),plt.hist(green.flatten(),bins=100,color='g'),plt.xlim(0,256)
plt.subplot(3,1,3),plt.hist(blue.flatten(),bins=100,color='b'),plt.xlim(0,256)

plt.show()

