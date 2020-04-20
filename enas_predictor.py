from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


def get_predictor(acc_seqs,final_accs,predictor_type="linear"):
	# data prep
	acc_seqs = np.asarray(acc_seqs)
	predictor = linear_model.LinearRegression()
	if predictor_type is "linear":
		predictor = linear_model.LinearRegression()
	elif predictor_type is "svr-rbf":
		predictor = SVR(kernel='rbf', C=1e4, gamma=0.1)
	elif predictor_type is "svr-linear":
		predictor = SVR(kernel='linear', C=1e4, gamma=0.1)
	elif predictor_type is "linearSVR":
		predictor = SVR(kernel='rbf', C=1e4, gamma=0.1)
	elif predictor_type is "logistic":
		predictor = LogisticRegression(solver='liblinear', random_state=0)
	# type selection
	
	predictor.fit(acc_seqs, final_accs)
	return predictor
	
def get_prediction(acc_seq, predictor):
	acc_seq = np.reshape(acc_seq, (-1,len(acc_seq)))
	prediction = predictor.predict(acc_seq)
	return prediction.item()

def get_untrained_prediction(acc_seq, predictor_type="linear", step_to_be_predicted):

	prediction = 0.0
	if predictor_type is "linear":
		x = np.arange(len(acc_seq))
		y = np.reshape(acc_seq, (-1,len(acc_seq)))
		predictor = linear_model.LinearRegression()
		predictor.fit(x,y)
		prediction = predictor.predict(step_to_be_predicted)
		
	elif predictor_type is "average":
		prediction = np.average(acc_seq)
		
	return prediction