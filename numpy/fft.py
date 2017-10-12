import numpy as np

# 一维度fft与ifft
a = np.array([3,4,5,6,7])
a2 = np.fft.fft(a)
a3 = np.fft.ifft(a2)
print(a2,a3)

b = np.array([-1,0,1])
b2 = np.fft.fft(b)
b3 = np.fft.ifft(b2)
print(b2,b3)

# 卷积

c = np.convolve(a,b)
YN = len(a) + len(b) - 1
a2 = np.fft.fft(a,YN)
b2 = np.fft.fft(b,YN)
c2 = np.fft.ifft(a2*b2)
print(c,c2.real)