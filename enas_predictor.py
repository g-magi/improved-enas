from sklearn.svm import SVR
import pandas as pd
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


def get_prediction(data, partial_data, percentage=0.25):
	prediction = 0.0
	
	n_steps = data[0].shape[0]
	n_features = 1
	model = Sequential()
	model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
	model.add(Dense(1))
	
	return prediction