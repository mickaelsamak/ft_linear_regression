import pandas as pd
import matplotlib.pyplot as plt

def plot_it(X, Y, t0, t1):
	Y_pred = t0 + t1 * X

	plt.scatter(X, Y)
	plt.plot([max(X), min(X)], [min(Y_pred), max(Y_pred)], color='red')
	plt.show()

def train():
	# Preprocessing Input data
	data = pd.read_csv('data.csv')
	X = data.iloc[:, 0]
	Y = data.iloc[:, 1]

	teta0 = 0
	teta1 = 0

	L = 0.01  # The learning Rate
	epochs = 10000  # The number of iterations to perform gradient descent

	m = float(len(X)) # Number of elements in X

	#Standardisation :
	X_S = (X - min(X)) / (max(X) - min(X))

	# Performing Gradient Descent
	for i in range(epochs):
		Y_pred = teta0 + (teta1 * X_S)

		# Derivative
		tmp_teta0 = L * (1 / m) * sum(Y_pred - Y)
		tmp_teta1 = L * (1 / m) * sum((Y_pred - Y) * X_S)

		# Update Teta
		teta0 = teta0 - tmp_teta0
		teta1 = teta1 - tmp_teta1

	# Making predictions

	t0 = teta1 * - min(X) / (max(X) - min(X))  + teta0
	t1 = teta1 * (1 - min(X)) /(max(X) - min(X)) + teta0 - t0

	# Save weight
	df = pd.DataFrame({'teta0': [t0], 'teta1' : [t1]} )
	df.to_csv("weight.csv", index = False)

	plot_it(X, Y, t0, t1)

train()
