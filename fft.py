import scipy as sci
import pylab as plt
import numpy as np
import os

dir = os.getcwd()
os.chdir(dir)
data = np.genfromtxt('data2.csv', delimiter=',')

def fft():
	#rate=1000.00
	#t=np.arange(0,1,1/rate)
	t = np.zeros((0,data.shape[0]),np.float)
	t = data[:,0]
	N=len(t)

	#s=np.sin(15*2*np.pi*t) + np.sin(25*2*np.pi*t+np.pi/4)+0.2*np.random.randn(t.size)
	s = np.zeros((0,data.shape[0]),np.float)
	s = data[:,1] 
	S=sci.fft(s)
	
	rate_np= 1/(data[1:2,0:1] - data[0:1,0:1])
	for rate in rate_np:
		f = 1/rate
		print rate,f
		freq_all=np.fft.fftfreq(N,f)
		pidxs=np.where(freq_all>0)
		freq= freq_all[pidxs]
	
	#freq=rate*np.arange(0,(N/2))/N
	n=len(freq)

	power=abs(S[0:n])/N

	#test = np.vstack([t,s])
	test = np.vstack([freq,power])
	test2 = np.transpose(test)
	S2=S.copy()

	S2[freq >26 ]=0
	S2[(freq>16)&(freq<24)]=0
	S2[freq <14]=0
	main_sig=sci.ifft(S2)


	np.savetxt("fft.csv",test2,fmt="%.5f",delimiter=",")
	plt.subplot(1,3,1),plt.plot(t,s,color='b')
	plt.title("signal")
	plt.subplot(1,3,2),plt.plot(freq,power,color='g'),plt.xlim(0,200)
	plt.title("Fourier transform")
	plt.subplot(1,3,3),plt.plot(t,main_sig,color='r')
	plt.title("Band-pass filter")
	plt.show()
if __name__ == '__main__':
	fft()
