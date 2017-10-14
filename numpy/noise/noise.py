import numpy as np
import time
from scipy import signal
from PIL import Image
from matplotlib import pyplot as plt  
import math

def addNoise(a):
	result = a.copy()
	for i in range(len(a)):
		for j in range(len(a[0])):
			if i %3 == 0 and j %3 ==0:
				result[i,j]=255
	
	return result

def fft(a):
	result = np.fft.fft2(a)
	result = np.fft.fftshift(result)
	result = np.sqrt(result.real**2+result.imag**2)
	#让图形分布好看一点
	result = np.log2(result+1)
	return result

def delNoise(a):
	a = np.fft.fft2(a)
	a = np.fft.fftshift(a)
	x = len(a)/2
	y = len(a[0])/2
	for i in range(len(a)):
		for j in range(len(a[0])):
			r = math.sqrt((i-x)**2+(j-y)**2)
			if r >= 100:
				a[i,j] =0
	a = np.fft.ifftshift(a)
	a = np.fft.ifft2(a)
	return a.real

a = Image.open('./a.jpg').convert('L') 
a = np.array(a)
b = addNoise(a)
c = delNoise(b)

a2 = fft(a)
b2 = fft(b)
c2 = fft(c)

fig = plt.figure()
plt.subplot(231)
plt.imshow(a,cmap='Greys_r') 
plt.subplot(232)
plt.imshow(b,cmap='Greys_r') 
plt.subplot(233)
plt.imshow(c,cmap='Greys_r') 
plt.subplot(234)
plt.imshow(a2,cmap='Greys_r') 
plt.subplot(235)
plt.imshow(b2,cmap='Greys_r') 
plt.subplot(236)
plt.imshow(c2,cmap='Greys_r') 
plt.show() 
