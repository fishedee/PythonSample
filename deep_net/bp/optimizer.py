class Optimizer:
	def __init__(self,rate=0.05,momentum=0):
		self._rate = rate
		self._momentum = momentum
		self._last_grade = None
	def update(self,weight,grade):
		if self._last_grade is None:
			self._last_grade = grade
		else:
			resultLastGrade = []
			for i in range(0,len(self._last_grade)):
				resultLastGrade.append(self._last_grade[i]*self._momentum+grade[i])
			self._last_grade = resultLastGrade
		resultWeight = []
		for i in range(0,len(weight)):
			resultWeight.append(weight[i]- self._rate * self._last_grade[i])
		return resultWeight