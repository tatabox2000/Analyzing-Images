import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as nd
from matplotlib.ticker import FormatStrFormatter
os.chdir(r'c://Users//analyst')

def hist_cluster():
	im = cv2.imread('00011.bmp')
	#im_HSV = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
	h,edges =np.histogramdd(im.reshape(-1,3),8,range=[(0,255),(0,255),(0,255)])
	h_flat=h.flatten()


	plt.subplot(1,2,1),plt.plot(h_flat)

	plt.show()















if __name__ == "__main__":
	hist_cluster()

