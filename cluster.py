import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
from matplotlib.ticker import FormatStrFormatter
os.chdir(r'c://Users//analyst')

def hist_cluster():
	im = cv2.imread('1546.bmp')
	im1 = im.copy()
	im2 = im.copy()
	#im_HSV = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
	im_gray= cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
	im_gray2= im_gray.copy()
	gray_smooth = cv2.medianBlur(im_gray,5)
	thresh = np.zeros(im_gray.shape,np.uint8)
	gr_bool =(gray_smooth>90)&(gray_smooth<120)
	thresh[gr_bool]=255
	contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	mask = np.zeros(im_gray.shape,np.uint8)
	for h,cnt in enumerate(contours):
		area = cv2.contourArea(cnt)
		if area > 32000 :
			pass
		elif area > 80 :
			cv2.drawContours(mask,[cnt],0,255,-1)
		else:
			pass
	mask2 = np.asarray(mask,np.bool8)
	im2[-mask2] = 255
	im_lab=cv2.cvtColor(im2,cv2.COLOR_BGR2LAB)
	mask3 = mask2.copy()
	#ret,thresh_mask=cv2.threshold(im2,5,255,0)
	contours_mask,hierarchy_mask = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

	for h,cnt in enumerate(contours):
		area = cv2.contourArea(cnt)
		if area >32000:
			pass
		elif area >80:
			rect = cv2.minAreaRect(cnt)
			xy,hw,angle = enumerate(rect)
			box= cv2.cv.BoxPoints(rect)
			box= np.int0(box)
			cv2.drawContours(im1,[box],0,(0,255,0),1)
			x,y,w,h=cv2.boundingRect(cnt)
			cv2.rectangle(im1,(x,y),(x+w,y+h),(255,0,0),1)
			a= cv2.arcLength(cnt,True)
			approx = cv2.approxPolyDP(cnt,0.1*cv2.arcLength(cnt,True),True)
	
	h,edges =np.histogramdd(im2.reshape(-1,3),8,range=[(0,254),(0,254),(0,254)])
	h_flat=h.flatten()
	print h
	mask_final=np.zeros_like(im_gray,np.uint8)
	mask_final[mask3]=0
	#hist_item =cv2.calcHist([im1],[2],mask_final,[1024],ranges=[1,255])
	#print hist_item
	plt.subplot(2,3,1),plt.plot(h_flat)
	plt.subplot(2,3,2),plt.imshow(im1)
	plt.subplot(2,3,3),plt.imshow(im2)
	plt.subplot(2,3,4),plt.imshow(mask_final,'gray')
#	plt.subplot(2,3,5),plt.imshow(h)
	plt.show()


if __name__ == "__main__":
	hist_cluster()

