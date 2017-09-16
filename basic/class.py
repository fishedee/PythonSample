# 定义私有属性，私有方法，保护属性，保护方法，公有函数，公有方法，构造函数
class MyObjA:
	contstructCounter = 0
	def __init__(self):
		self.counter = 0
		self.__decCounter = 0
		self._incCounter = 0
		MyObjA.contstructCounter += 1
	def inc(self):
		self.counter += 1
		self._incCounter += 1
	def dec(self):
		self.counter -= 1
		self.__decCounter += 1
	def get(self):
		return self.counter
	def getDecCounter(self):
		return self.__decCounter
	def getContstructCounter():
		return MyObjA.contstructCounter

# 定义派生类，覆盖函数
class MyObjB(MyObjA):
	def __init__(self):
		MyObjA.__init__(self)
	def inc(self):
		self.counter += 2
		self._incCounter += 1
	def print(self):
		print("count %d,decCounter %d,incCounter %d"%(self.counter,self.getDecCounter(),self._incCounter))

a = MyObjA()
print(a.get())
a.dec()
print(a.get())
a.inc()
a.inc()
print(a.get())

b = MyObjB()
b.print()
b.dec()
b.print()
b.inc()
b.inc()
b.print()

print(MyObjA.getContstructCounter())
