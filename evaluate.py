import pandas as pd

def evaluate():
	x = input("how many kilometers does your car have ? ")
	x = int(x)

	data = pd.read_csv('weight.csv')
	t0 = data.iloc[0]['teta0']
	t1 = data.iloc[0]['teta1']

	price = int(t0 + t1 * x)
	print ("The price is : " + str(price))

evaluate()
