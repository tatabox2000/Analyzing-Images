import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import django as dj
from skimage import io
import scipy.ndimage as nd

os.chdir(r'c://Users//analyst')

im=cv2.imread('1240.bmp')
im2=im.copy()
imgray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
gray_smooth = cv2.medianBlur(imgray,5)


#imgray[:] = nd.binary_opening(imgray, iterations=100)
#imgray[:] = nd.binary_closing(imgray, iterations=1)

ret,thresh=cv2.threshold(gray_smooth,100,255,0)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#print contours
mask = np.zeros(imgray.shape,np.uint8)
cv2.drawContours(mask,contours,-1,255,-1)
#for h,cnt in enumerate(contours):
#	area = cv2.contourArea(cnt)
#	if area > 20000 :
#		cv2.drawContours(mask,[cnt],0,255,-1)
#		print area
#		plt.subplot(2,3,6),plt.imshow(mask)
#	else:
#		pass
mask2 = np.asarray(mask,np.bool8)
im2[mask2] = 0
plt.subplot(2,3,1),plt.hist(imgray.flatten(),bins=255),plt.xlim(0,255)
plt.subplot(2,3,2),plt.imshow(im)
plt.subplot(2,3,3),plt.imshow(im2)
plt.subplot(2,3,4),plt.imshow(gray_smooth,'gray')
plt.subplot(2,3,5),plt.imshow(mask)
plt.show()

