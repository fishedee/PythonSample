import numpy as np
import time
from scipy import signal
from PIL import Image
from matplotlib import pyplot as plt  

def convolve2d(a,b):
	height = len(a) + len(b) - 1
	width = len(a[0])+len(b[0])-1
	FFT_height = 2 ** (int(np.log2(height)) + 1)
	FFT_width = 2 ** (int(np.log2(width)) + 1)
	a2 = np.fft.fft2(a,(FFT_height,FFT_width))
	b2 = np.fft.fft2(b,(FFT_height,FFT_width))
	c2 = np.fft.ifft2(a2*b2)
	data =  c2.real[:height,:width]
	return data

def related2d(a,b):
	b = np.rot90(b,2)
	t1 = time.clock()
	result = convolve2d(a,b)
	t2 = time.clock()
	print("convolve2d time:%fs"%((t2-t1)))
	return result

def related2d_valid(a,b):
	ah = len(a)
	bh = len(b)
	aw = len(a[0])
	bw = len(b[0])
	h = ah-bh+1
	w = aw-bw+1
	t1 = time.clock()
	result = related2d(a,b)
	t2 = time.clock()
	print("related2d time:%fs"%((t2-t1)))

	oh = len(result)
	ow = len(result[0])
	cropH = int((oh-h)/2)
	cropH2 = oh - h - cropH
	cropW = int((ow-w)/2)
	cropW2 = ow - w - cropW
	result = result[cropH:-cropH2,cropW:-cropW2]
	return result

def related2d_valid2(a,b):
	return signal.fftconvolve(a,np.rot90(b,2),mode="valid")

def cumsum(a):
	a = a **2
	b = np.cumsum(a,axis=0)
	c = np.cumsum(b,axis=1)
	return c

def a_cumsum(a,b):
	ah = len(a)
	bh = len(b)
	aw = len(a[0])
	bw = len(b[0])
	asum = cumsum(a)
	result = np.zeros((ah-bh+1,aw-bw+1))
	for i in range (0,len(result)):
		for j in range(0,len(result[0])):
			x = i +bh-1
			y = j +bw-1
			left = 0
			top = 0
			center = 0
			if x -bh >= 0 and y >=0:
				top = asum[x-bh,y]
			if x >= 0 and y - bw >= 0:
				left = asum[x,y-bw]
			if x - bh >= 0 and y -bw >= 0:
				center = asum[x-bh,y-bw]

			result[i,j] = asum[x,y]-top-left+center
	return result

def b_cumsum(a,b):
	bsum = cumsum(b)[-1,-1]
	ah = len(a)
	bh = len(b)
	aw = len(a[0])
	bw = len(b[0])
	result = np.ones((ah-bh+1,aw-bw+1))*bsum
	return result

def template_match(a,b):
	t1 = time.clock()
	asum = a_cumsum(a,b)
	t2 = time.clock()
	print("a_cumsum time:%fs"%((t2-t1)))

	bsum = b_cumsum(a,b)
	mutli = related2d_valid(a,b)
	result = mutli/(np.sqrt(bsum)*np.sqrt(asum))
	minIndex = np.argmax(result)
	index = (int(minIndex/len(result[0])),minIndex%len(result[0]))
	return index,result[index]

a = Image.open('./a2.jpg').convert('L') 
a2 = np.array(a).astype('float32')/255
b = Image.open('./b2.jpg').convert('L')
b2 = np.array(b).astype('float32')/255

t1 = time.clock()

result = template_match(a2,b2)

t2 = time.clock()
print("time:%fs"%((t2-t1)))

print(result)


y = result[0][0]
x = result[0][1]
h = len(b2)
w = len(b2[0])

fig = plt.figure()  
pointx = [x, x+w, x+w, x, x]  
pointy = [y, y, y+h, y+h, y]  
plt.plot(pointx, pointy, 'r')
plt.imshow(a,cmap='Greys_r') 
plt.show() 
