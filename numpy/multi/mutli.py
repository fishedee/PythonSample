import numpy as np

a = np.array([4,4,4,4,5,5])
b = np.array([7,7,7,8,8])

def fft(a,b):
	width = len(a) + len(b) - 1
	FFT_width = 2 ** (int(np.log2(width)) + 1)
	a2 = np.fft.fft(a,FFT_width)
	b2 = np.fft.fft(b,FFT_width)
	c2 = np.fft.ifft(a2*b2)
	data =  np.around(c2.real[:width]).astype('int')
	return data

def mutli(a,b):
	a = a[::-1]
	b = b[::-1]
	result = fft(a,b)
	last = 0
	for i in range(0,len(result)):
		cur = result[i]+last
		result[i] = cur % 10
		last = cur // 10
	result = np.append(result,[cur//10])
	return result[::-1]

print(mutli(a,b))