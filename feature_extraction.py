import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
import scipy.ndimage as nd
from matplotlib.ticker import FormatStrFormatter
os.chdir(r'c://Users//analyst')

def aspect_circle():
	im=cv2.imread('00011.bmp')
	im1= im.copy()
	im_gray= cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
	im_gray2= im_gray.copy()
	gray_smooth = cv2.medianBlur(im_gray,5)
	thresh = np.zeros(im_gray.shape,np.uint8)
	gr_bool =(gray_smooth>100)&(gray_smooth<130)
	thresh[gr_bool]=255
	contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	for h,cnt in enumerate(contours):
		area = cv2.contourArea(cnt)
		if area >32000:
			pass
		elif area >100:
			rect = cv2.minAreaRect(cnt)
			xy,hw,angle = enumerate(rect)
			box= cv2.cv.BoxPoints(rect)
			box= np.int0(box)
			cv2.drawContours(im1,[box],0,(0,255,0),1)
			x,y,w,h=cv2.boundingRect(cnt)
			cv2.rectangle(im1,(x,y),(x+w,y+h),(255,0,0),1)
			a= cv2.arcLength(cnt,True)
			print a
			approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
			print approx
			print xy
	plt.subplot(1,2,1),plt.imshow(im)
	plt.subplot(1,2,2),plt.imshow(im1)
	plt.show()

if __name__=="__main__":
	aspect_circle()


	
