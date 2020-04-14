from sklearn.svm import SVR
import pandas as pd
import numpy as np


def get_predictor(acc_seqs,final_accs):
	# TODO creare predittore e ritornarlo
	predictor = SVR(kernel='rbf', C=1e4, gamma=0.1)
	predictor.fit(acc_seqs, final_accs)
	return predictor
	
def get_prediction(acc_seq, predictor):
	prediction = predictor.predict(acc_seq)
	return prediction
