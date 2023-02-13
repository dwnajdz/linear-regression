import copy
import numpy as np
from matplotlib import pyplot as plt

X = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y = np.array([460, 232, 178])

class LinearRegression:
	w=0; b=0;
	def __init__(self, X, y, alpha, iters):
		#X_normalized = self.zscore(X)
		n = X.shape[1]
		new_w = np.zeros((n,))
		self.w, self.b = self.gd(X, y, new_w, self.b, alpha, iters)

	# standard deviaton
	def calculate_stdd(self, X, mean, m):
		stdd = 0
		for i in range(m):
			stdd += abs((X[i]-mean)**2)
		return stdd/m

	def zscore(self, X):
		m = X.shape[0]
		mean = np.mean(X)
		stdd = self.calculate_stdd(X, mean, m)
		X_normalized = np.zeros(m)
		for i in range(m):
			X_normalized[i] = (X[i]-mean)/stdd 
		return X_normalized

	def mean_squared_error(self, y, yhat):
		m = y.shape[0]
		loss = 0
		for i in range(m):
			diff = (y[i]-yhat[i])**2
			loss += diff
		loss /= 2*m
		return loss

	def calculate_delta(self, X, y, w, b):
		m, n = X.shape
		djw = np.zeros((n,))
		djb = 0.
		for i in range(m):
			error = (np.dot(w, X[i])+b)-y[i]
			for j in range(n):
				djw[j] += X[i, j]*error
			djb += error
		djw /= m
		djb /= m

		return djw, djb

	def gd(self, X, y, w_in, b_in, alpha=0.1, iters=1000):
		w = copy.deepcopy(w_in)
		b = b_in
		#w = w_in
		for i in range(iters):
			w_delta, b_delta = self.calculate_delta(X, y, w, b)
			w -= alpha*w_delta
			b -= alpha*b_delta
		return w, b

	def predict(self, X):
		m = X.shape[0]
		fx = np.zeros(m)
		for i in range(m):
			fx[i] = np.dot(self.w, X[i])+self.b
		return fx

model = LinearRegression(X, y, 5.0e-7, 1000)
print(model.w, model.b)
print(model.predict(X))

plt.scatter(X[:, 0], y, color='blue')
plt.plot(X[:, 0], model.predict(X), color='red')
plt.show()
