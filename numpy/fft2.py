import numpy as np
from scipy import signal

# 二维度fft与ifft
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
a2 = np.fft.fft2(a)
a3 = np.fft.ifft2(a2)
print(a2,a3)

b = np.array([[-1,0],[0,1]])
b2 = np.fft.fft2(b)
b3 = np.fft.ifft2(b2)
print(b2,b3)

# 卷积
c = signal.convolve2d(a, b)
YN = len(a) + len(b) - 1
a2 = np.fft.fft2(a,(YN,YN))
b2 = np.fft.fft2(b,(YN,YN))
c2 = np.fft.ifft2(a2*b2)
print(c,c2.real)