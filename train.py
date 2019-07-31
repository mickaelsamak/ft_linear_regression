import pandas as pd
import matplotlib.pyplot as plt

def plot_it(X, Y, t0, t1):
	Y_pred = t0 + t1 * X

	plt.figure(2)
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
	epochs = 25000  # The number of iterations to perform gradient descent

	m = float(len(X)) # Number of elements in X

	#Standardisation :
	X_S = (X - min(X)) / (max(X) - min(X))
	accuracy = []

	# Performing Gradient Descent
	for i in range(epochs):
		Y_pred = teta0 + (teta1 * X_S)

		# Derivative
		tmp_teta0 = (1 / m) * sum(Y_pred - Y)
		tmp_teta1 = (1 / m) * sum((Y_pred - Y) * X_S)

		loss = abs((1 / m) * sum(Y_pred - Y))
		print ("Epoch : " + str(i + 1))
		print("Loss : " + str(loss))
		print ("--------------------------")
		accuracy.append(loss)
		# Update Teta
		teta0 -= L * tmp_teta0
		teta1 -= L * tmp_teta1

	# Making predictions
	plt.figure(1)
	plt.plot(accuracy, color='red')
	print ("Loss : " + str(accuracy[-1]))

	t0 = teta1 * - min(X) / (max(X) - min(X))  + teta0
	t1 = teta1 * (1 - min(X)) /(max(X) - min(X)) + teta0 - t0

	# Save weight
	df = pd.DataFrame({'teta0': [t0], 'teta1' : [t1]} )
	df.to_csv("weight.csv", index = False)

	plot_it(X, Y, t0, t1)

train()
