import numpy as np

a = np.arange(0,12)
a.shape = 3,4
print(a)
a.tofile("__pycache__/.a.bin")

b = np.fromfile("__pycache__/.a.bin",dtype=a.dtype)
b.shape = a.shape
print(b)

np.save("__pycache__/.b.npy",a)
c = np.load("__pycache__/.b.npy")
print(c)

np.savetxt("__pycache__/.c.txt", a)
d = np.loadtxt("__pycache__/.c.txt")
print(d)
