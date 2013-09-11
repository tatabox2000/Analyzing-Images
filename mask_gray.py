import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import scipy.ndimage as nd
from matplotlib.ticker import FormatStrFormatter
os.chdir(r'c://Users//analyst')

im=cv2.imread('812.bmp')
#im=cv2.imread('balls.png')
im2=im.copy()
im3 = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
im2=im3.copy()
imgray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
gray_smooth = cv2.medianBlur(imgray,5)


#imgray[:] = nd.binary_opening(imgray, iterations=100)
#imgray[:] = nd.binary_closing(imgray, iterations=1)
#ret,thresh=cv2.threshold(gray_smooth,110,255,0)

thresh = np.zeros(imgray.shape,np.uint8)
gr_bool =(gray_smooth>100)&(gray_smooth<130)
thresh[gr_bool]=255

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#print contours
mask = np.zeros(imgray.shape,np.uint8)
#cv2.drawContours(mask,contours,-1,255,-1)
for h,cnt in enumerate(contours):
	area = cv2.contourArea(cnt)
	if area > 32000 :
		pass
	elif area > 100 :
		cv2.drawContours(mask,[cnt],0,255,-1)
	else:
		pass
mask2 = np.asarray(mask,np.bool8)
im2[-mask2] = 0
mask3 = mask2.copy()
#ret,thresh_mask=cv2.threshold(im2,5,255,0)
contours_mask,hierarchy_mask = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
for h2,cnt2 in enumerate(contours_mask):
	area = cv2.contourArea(cnt2)
	M=cv2.moments(cnt2)
	centroid_x = int(M['m10']/M['m00'])
	centroid_y = int(M['m01']/M['m00'])
print centroid_x  , centroid_y
mask3[centroid_y-2.5:centroid_y+2.5,centroid_x-2.5:centroid_x+2.5]=0

pick_up =np.zeros((50,50,3),np.uint8)
pick_up =im3[centroid_y-25:centroid_y+25,centroid_x-25:centroid_x+25]
print pick_up

plt.subplot(3,3,1),plt.imshow(im3)
plt.tick_params(labelbottom='off')
plt.title('picture of a defect')
plt.subplot(3,3,2),plt.imshow(gray_smooth,'gray')
plt.tick_params(labelbottom='off')
plt.title('make a gray picture')
plt.subplot(3,3,3),plt.hist(imgray.flatten(),bins=100,color='m'),plt.xlim(0,255)
plt.title('gray histgram')
plt.subplot(3,3,4),plt.imshow(mask3,'gray')
plt.tick_params(labelbottom='off')
plt.title('mask(100-130) & centroid')
plt.subplot(3,3,5),plt.imshow(im2)
plt.tick_params(labelbottom='off')
plt.title('mask picture')
plt.subplot(3,3,6),plt.imshow(pick_up)
plt.tick_params(labelbottom='off')
plt.title('Area cutting')
plt.subplot(3,3,7),plt.hist(im2[:,:,2].flatten(),bins=50,color='r'),plt.xlim(1,255),plt.ylim(0,80)
plt.title('Red histgram')
plt.subplot(3,3,8),plt.hist(im2[:,:,1].flatten(),bins=50,color='g'),plt.xlim(1,255),plt.ylim(0,80)
plt.title('Gree histgram')
plt.subplot(3,3,9),plt.hist(im2[:,:,0].flatten(),bins=50,color='b'),plt.xlim(1,255),plt.ylim(0,80)
plt.title('Blue histgram')

plt.show()

