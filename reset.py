import pandas as pd

def reset_data():
	df = pd.DataFrame({'teta0': ['0'], 'teta1' : ['0']} )
	df.to_csv("weight.csv", index = False)

reset_data()
