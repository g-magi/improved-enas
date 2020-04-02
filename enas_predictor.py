from sklearn.svm import SVR
import pandas as pd



def build_prediction_model(data):
	
	### TODO TODO TODO
	# sulla base dei dati addestrare il predittore e ritornarlo
	##
	
	predictor = SVR()
	prepared_data = pd.DataFrame(columns = ['epoch'])
	step_amt = data['acc'].max()
	for i in range(step_amt):
		
		prepared_data = prepared_data.assign()
	
	
	
	return predictor

def get_prediction(source, data, percentage=0.25, target_step):
	prediction = 0.0
	
	### TODO
	# usare il modello passato in [source] e una percentuale dei dati per estrarre una predizione 
	# dell'accuratezza allo step [target_step]
	# 
	#
	
	return prediction