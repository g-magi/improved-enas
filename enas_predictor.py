from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np


def get_predictor(acc_seqs,final_accs,predictor="linear"):
	# data prep
	acc_seqs = np.asarray(acc_seqs)
	if predictor is "linear":
		predictor = linear_model.LinearRegression()
	elif predictor is "svr-rbf":
		predictor = SVR(kernel='rbf', C=1e4, gamma=0.1)
	elif predictor is "svr-linear":
		predictor = SVR(kernel='linear', C=1e4, gamma=0.1)
	elif predictor is "linearSVR":
		predictor = SVR(kernel='rbf', C=1e4, gamma=0.1)
	elif predictor is "logistic":
		predictor = LogisticRegression(solver='liblinear', random_state=0)
	# type selection
	
	predictor.fit(acc_seqs, final_accs)
	return predictor
	
def get_prediction(acc_seq, predictor):
	acc_seq = np.reshape(acc_seq, (-1,len(acc_seq)))
	prediction = predictor.predict(acc_seq)
	return prediction.item()
