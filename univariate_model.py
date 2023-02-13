import copy
import numpy as np
from matplotlib import pyplot as plt

X = np.array([1.0, 2.0])
y = np.array([300.0, 500.0])

class LinearRegressionAlpha:
	# ONLY FOR ONE FEATURE 
	# to do need to make it for more
	w=0; b=0;
	def __init__(self, X, y, alpha, iters, z_normalize=False):
		if z_normalize:
			X = self.zscore(X)
		self.w, self.b = self.gd(X, y, self.w, self.b, alpha, iters)

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
		m = X.shape[0]
		djw = 0
		djb = 0

		for i in range(m):
			err = (w*X[i]+b)-y[i]
			djb += err
			djw += err*X[i]
			
		djw /= m
		djb /= m

		return djw, djb

	def gd(self, X, y, w_in, b_in, alpha=0.1, iters=1000):
		b = b_in
		w = w_in
		for i in range(iters):
			w_delta, b_delta = self.calculate_delta(X, y, w, b)
			w -= alpha*w_delta
			b -= alpha*b_delta

		return w, b

	def predict(self, X):
		fx = (self.w*X)+self.b
		return fx

model = LinearRegressionAlpha(X, y, 1.0e-2, 10000)

plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.show()
